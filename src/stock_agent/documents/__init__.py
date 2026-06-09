"""Document acquisition + parsing for the RAG layer (SEC filings).

Downloads via the official EDGAR API (``providers/sec_edgar.py``), parses to
normalized ``schemas/documents.py`` shapes. Built incrementally — see
docs/RAG_TODO.md (P1 download, P2 parsing).
"""
