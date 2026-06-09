"""Grounded research synthesis: one LLM call over retrieved evidence + model outputs.

Combines technicals (``indicators/``), forecasts (``forecasting/``), news, and RAG
evidence into a cited research memo. Numbers come from the modules, never the LLM;
citations come from the retrieved set. Built incrementally — see docs/RAG_TODO.md
(P7 grounded QA, P8 research memo).
"""
