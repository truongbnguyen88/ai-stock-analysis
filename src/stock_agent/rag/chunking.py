"""Section-aware chunking (RAG P3) — sections -> retrieval-sized DocumentChunks.

Pure, deterministic. Splits each parsed section (from P2) into overlapping
fixed-size windows that **never cross a section boundary**, copying the filing's
metadata onto every chunk so the vector store can filter + cite from a chunk alone.

Sizing uses a tokenizer-free **word-count proxy** (≈ 1 token ≈ 0.75 English words)
so P3 stays dependency-free — a real tokenizer can replace ``_target_words`` later
without changing the windowing logic. Also folds in the TOC dedup P2 deferred:
sections whose body is below ``min_chunk_words`` (header-only table-of-contents
entries, trivial cover lines) are dropped.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from stock_agent.documents.parsers import ParsedFiling, Section
from stock_agent.schemas.documents import DocumentChunk, DocumentMetadata

# 1 token ≈ 0.75 words for English prose (~4 chars/token, ~5 chars/word). Used to
# turn the settings' token budget into a word budget without a tokenizer dependency.
_WORDS_PER_TOKEN = 0.75
# Below this many words a "section" is a TOC header line / trivial fragment, not
# content — dropped (dedup) so it never becomes a tiny, near-useless chunk.
_MIN_CHUNK_WORDS = 15

# Defaults mirror settings.rag_chunk_tokens / rag_chunk_overlap (the pipeline passes
# the configured values; these keep the function usable standalone + in tests).
_DEFAULT_CHUNK_TOKENS = 900
_DEFAULT_CHUNK_OVERLAP = 0.15


def _target_words(chunk_tokens: int) -> int:
    """Word-count target for one chunk, from a token budget (proxy, no tokenizer)."""
    return max(1, round(chunk_tokens * _WORDS_PER_TOKEN))


def _overlap_words(target: int, overlap_fraction: float) -> int:
    """Words shared between adjacent chunks; clamped to ``[0, target-1]`` so step >= 1."""
    return min(max(0, round(target * overlap_fraction)), target - 1)


def _windows(words: list[str], target: int, step: int) -> Iterator[list[str]]:
    """Yield overlapping word windows of size ``target`` advancing by ``step``.

    Every non-final window has exactly ``target`` words; the final window has
    ``> target - step`` (= overlap) words — so no window is degenerately tiny when
    the input exceeds ``target`` (a shorter-than-target input yields a single window).
    """
    i = 0
    n = len(words)
    while True:
        yield words[i : i + target]
        if i + target >= n:
            return
        i += step


def chunk_sections(
    sections: Sequence[Section],
    metadata: DocumentMetadata,
    *,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP,
    min_chunk_words: int = _MIN_CHUNK_WORDS,
) -> list[DocumentChunk]:
    """Chunk parsed ``sections`` into ``DocumentChunk``s (one filing's worth).

    Each section is windowed independently (boundaries preserved). Sections with
    fewer than ``min_chunk_words`` words are dropped (TOC dedup). ``chunk_index`` is
    a document-global running counter, so ``chunk_id = document_id:index`` is unique.
    """
    target = _target_words(chunk_tokens)
    step = max(1, target - _overlap_words(target, chunk_overlap))

    chunks: list[DocumentChunk] = []
    index = 0
    for section in sections:
        words = section.text.split()
        if len(words) < min_chunk_words:
            continue  # header-only / trivial section -> drop
        section_label = section.label or None  # "" (no-item fallback) -> None
        for window in _windows(words, target, step):
            chunks.append(
                DocumentChunk.from_metadata(
                    metadata,
                    chunk_index=index,
                    text=" ".join(window),
                    section=section_label,
                )
            )
            index += 1
    return chunks


def chunk_filing(
    parsed: ParsedFiling,
    *,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    chunk_overlap: float = _DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Chunk a parsed filing (P2 ``ParsedFiling``) into retrieval-ready chunks."""
    return chunk_sections(
        parsed.sections,
        parsed.metadata,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )
