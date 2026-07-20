"""Full RL for retrieval (advanced-RAG A6.2) — the agentic loop as a finite-horizon MDP.

This package generalizes the A4 ReAct loop (a fixed LLM prompt) and the A5 entity-bridge (a fixed
heuristic) into ONE learned sequential policy over an MDP whose transitions run the *real* retriever
against the *real* corpus with *templated* (no-LLM) query strings — so rollouts are deterministic,
``$0``, and reproducible. See ``docs/a6_2_plan.md`` for the slice-by-slice plan and
``docs/rl_rag_pre_questions.md`` Q2–Q5 for the MDP design brief.

Modules (built slice-by-slice, default-OFF, CI torch-free):
- ``state``    (A6.2a) — dynamic evidence-summary state featurizer (numpy; label-free).
- ``action``   (A6.2b) — discrete action space + query templates + scope resolution.
- ``env``      (A6.2c) — Gym-style ``reset``/``step`` MDP + a transition cache.
- ``policy``   (A6.2d) — numpy linear-softmax policy + value baseline.
- ``reinforce``(A6.2d) — rollout collection + REINFORCE + behavior-cloning warm-start.
- ``ppo``      (A6.2e) — torch MLP actor-critic (isolated ``[rl]`` extra; lazy import).
- ``train``    (A6.2f) — standardizer + training driver + checkpoint freeze/load.
"""

from __future__ import annotations
