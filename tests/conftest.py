"""Shared pytest configuration.

``test_sequence.py`` (LSTM forecaster) and ``test_rl_ppo.py`` (A6.2e retrieval PPO) both need an
optional torch extra (``[sequence]`` / ``[rl]``). On macOS, torch and lightgbm each bundle their own
OpenMP runtime, and loading both in one interpreter segfaults — so once torch is imported, the
lightgbm-based tests in the same session crash. CI never installs torch (both files ``importorskip``
cleanly), but a local dev machine with the extra installed would collect + import torch and bring
the whole gate down.

So we **do not collect** these torch tests by default. They pass in isolation; run them deliberately
with::

    RUN_SEQUENCE_TESTS=1 pytest tests/unit/test_sequence.py
    RUN_RL_TORCH_TESTS=1  pytest tests/unit/test_rl_ppo.py
"""

from __future__ import annotations

import os

collect_ignore: list[str] = []
if not os.environ.get("RUN_SEQUENCE_TESTS"):
    collect_ignore.append("unit/test_sequence.py")
if not os.environ.get("RUN_RL_TORCH_TESTS"):
    collect_ignore.append("unit/test_rl_ppo.py")
