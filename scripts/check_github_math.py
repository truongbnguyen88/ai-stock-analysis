#!/usr/bin/env python3
"""Lint Markdown for GitHub-MathJax rendering hazards. Gates docs/*.md via `make check-math`.

GitHub renders math with a MathJax *subset* and unescapes backslash-punctuation inside math,
so equations that render locally can break on github.com. This flags the known hazards.

VENDORED from the user-global `github-math` skill (~/.claude/skills/github-math/scripts/
check-github-math.py) so CI is self-contained. If you improve one, sync the other.

Usage:
    python scripts/check_github_math.py FILE.md [FILE.md ...]

Exit code 0 if clean, 1 if any hazard is found (suitable for pre-commit / CI). Stdlib only.
Heuristics, not a full Markdown/TeX parser: it skips fenced code blocks and inline code spans,
tracks $$ display blocks, and inspects inline $…$ spans. Rare false positives are acceptable —
when in doubt, verify the actual GitHub render.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Macros that only ever appear in math and always break on GitHub — flag anywhere (post code-strip).
ALWAYS_BAD: dict[str, str] = {
    r"\;": r'\; renders as a literal ";" on GitHub — drop it (MathJax auto-spaces relations)',
    r"\,": r'\, renders as a literal "," on GitHub — drop it',
    r"\!": r'\! renders as a literal "!" on GitHub — drop it',
    r"\:": r'\: renders as a literal ":" on GitHub — drop it',
    r"\thickspace": r"\thickspace is unsupported on GitHub MathJax; prints literally — use \quad or drop",
    r"\thinspace": r"\thinspace is unsupported on GitHub MathJax; prints literally — use \quad or drop",
    r"\medspace": r"\medspace is unsupported on GitHub MathJax; prints literally — use \quad or drop",
    r"\negthinspace": r"\negthinspace is unsupported on GitHub MathJax; prints literally — drop it",
}

# Backslash-escaped Markdown punctuation: illegal INSIDE math (GitHub unescapes it before MathJax).
# Each pattern requires a SINGLE preceding backslash — (?<!\\) skips the amsmath row break \\[4pt].
IN_MATH_BAD: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\\)\\#"), r"\# inside math — GitHub unescapes it before MathJax (error)"),
    (re.compile(r"(?<!\\)\\_"), r"\_ inside math — GitHub unescapes it before MathJax (error); use x_i or a code span"),
    (re.compile(r"(?<!\\)\\\*"), r"\* inside math — use {*} or \ast"),
    (re.compile(r"(?<!\\)\\`"), r"\` inside math — GitHub unescapes it before MathJax (error)"),
    (re.compile(r"(?<!\\)\\\["), r"\[ inside math — GitHub unescapes it before MathJax (error)"),
    (re.compile(r"(?<!\\)\\\]"), r"\] inside math — GitHub unescapes it before MathJax (error)"),
]

INLINE_CODE = re.compile(r"`[^`]*`")
INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)")  # $...$ but not $$
TEXT_UNDERSCORE = re.compile(r"\\te?xt(?:tt)?\{[^}]*_[^}]*\}")
# Currency = an unescaped $<number> whose tail looks like prose/currency (a letter, ')', or '|',
# or end-of-line) rather than math. This deliberately does NOT match numeric-leading inline math
# such as $2^1-1$, $1/62$, $0.7$, $60$ (followed by an operator or a closing $), to avoid false
# positives in math-heavy docs. Flag only when 2+ appear on one line (a lone $amount is harmless).
CURRENCY = re.compile(r"(?<!\\)\$\d[\d.,]*(?=\s*[A-Za-z)|]|\s*$)")


def check_file(path: str) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [(0, f"cannot read file: {exc}")]

    in_fence = False
    in_display = False
    total_display_delims = 0

    for lineno, raw in enumerate(lines, start=1):
        if raw.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # Strip inline code spans so documented examples like `\,` aren't flagged.
        line = INLINE_CODE.sub("", raw)

        # (1) math-only macros that always break — anywhere on the line
        for macro, msg in ALWAYS_BAD.items():
            if macro in line:
                findings.append((lineno, msg))

        # (2) underscore inside \text{}/\texttt{}
        if TEXT_UNDERSCORE.search(line):
            findings.append((lineno, r"underscore inside \text{}/\texttt{} breaks math — use a code span or clean symbol"))

        # (3) currency collision: 2+ unescaped $<digit> on one line
        if len(CURRENCY.findall(line)) >= 2:
            findings.append((lineno, r"2+ unescaped $<digit> on one line — GitHub renders the text between them as math; escape each as \$"))

        # (4) backslash-punctuation inside math spans (display line, or inline $…$)
        segments: list[str] = []
        if in_display:
            segments.append(line)
        segments.extend(INLINE_MATH.findall(line))
        for seg in segments:
            for pat, msg in IN_MATH_BAD:
                if pat.search(seg):
                    findings.append((lineno, msg))

        # Track $$ display-block state (toggle on an odd count of $$ on the line).
        n_dd = line.count("$$")
        total_display_delims += n_dd
        if n_dd % 2 == 1:
            in_display = not in_display

    if total_display_delims % 2 == 1:
        findings.append((0, r"odd number of $$ in file — a display-math block is unclosed"))

    # De-duplicate while preserving order.
    seen: set[tuple[int, str]] = set()
    unique: list[tuple[int, str]] = []
    for f in findings:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check-github-math.py FILE.md [FILE.md ...]", file=sys.stderr)
        return 2
    any_bad = False
    for path in argv:
        findings = check_file(path)
        for lineno, msg in sorted(findings):
            any_bad = True
            where = f"{path}:{lineno}" if lineno else path
            print(f"{where}: {msg}")
    if not any_bad:
        print("OK — no GitHub-MathJax hazards found.")
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
