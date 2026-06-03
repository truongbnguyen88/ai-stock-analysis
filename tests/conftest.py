"""Shared pytest configuration.

``test_sequence.py`` exercises the experimental LSTM forecaster, which needs the
optional ``[sequence]`` extra (torch). On macOS, torch and lightgbm each bundle
their own OpenMP runtime, and loading both in one interpreter segfaults — so once
torch is imported, the lightgbm-based tests in the same session crash. CI never
installs torch (the tests ``importorskip`` cleanly), but a local dev machine with
the extra installed would collect + import torch and bring the whole gate down.

So we **do not collect** the sequence tests by default. They pass in isolation;
run them deliberately with::

    RUN_SEQUENCE_TESTS=1 pytest tests/unit/test_sequence.py
"""

from __future__ import annotations

import os

collect_ignore: list[str] = []
if not os.environ.get("RUN_SEQUENCE_TESTS"):
    collect_ignore.append("unit/test_sequence.py")
