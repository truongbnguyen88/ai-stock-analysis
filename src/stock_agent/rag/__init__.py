"""Retrieval-Augmented Generation core: chunk → embed → store → retrieve.

Local-first and LLM-free at retrieval time (the only paid LLM call is the final
synthesis in ``research/``). Embeddings and the vector store each sit behind a
Protocol. Built incrementally — see docs/RAG_TODO.md (P3–P6).
"""
