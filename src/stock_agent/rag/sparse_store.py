"""Sparse (BM25) retrieval index (advanced-RAG A3) — the lexical half of hybrid search.

Dense embeddings capture *meaning* but blur **exact terms**: a ticker, a section name ("Item 7A"),
a defined product ("Hopper"), a dollar figure. BM25 is the classic lexical scorer — it ranks a chunk
by how many query *terms* it has, weighted by term rarity (IDF) and saturated by length. Pairing the
two (hybrid search, `rag/hybrid.py`) recovers exact-term hits dense search misses without losing
semantic recall.

Two backends behind one ``SparseStore`` Protocol:
- ``Fts5SparseStore`` (default) — SQLite **FTS5** (Python stdlib ``sqlite3``; **no new dep**),
  persistent on disk, BM25 via the built-in ``bm25()`` function, metadata filters via SQL ``WHERE``.
- ``InMemoryBM25Store`` — a tiny pure-Python Okapi BM25 for parity, deterministic tests, and
  environments whose ``sqlite3`` was built without FTS5.

The store holds the chunk **text + flat metadata** (not vectors), so it can both rank (``search``)
and materialize chunks for fusion (``fetch``). It is written **at ingestion**, alongside the vector
store; querying does no embedding and no LLM call.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Sequence
from datetime import date as Date
from pathlib import Path
from typing import Protocol, runtime_checkable

from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter
from stock_agent.settings import Settings

# Query tokenization: alphanumeric runs, lowercased. Deliberately simple + shared by both backends
# so FTS5 and the in-memory BM25 tokenize queries identically (deterministic parity in tests).
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase → alphanumeric tokens (drops punctuation that would break an FTS5 MATCH string)."""
    return _TOKEN.findall(text.lower())


@runtime_checkable
class SparseStore(Protocol):
    """A lexical (BM25) index over ``DocumentChunk`` text, with metadata filtering.

    Mirrors the vector store's role on the sparse side: ``add`` at ingest, ``search`` returns
    ``(chunk_id, score)`` ranked best-first (higher = more relevant), ``fetch`` materializes chunks
    for fusion, ``existing_ids`` drives the incremental refresh.
    """

    name: str

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        """Index ``chunks`` (idempotent upsert by ``chunk_id``)."""
        ...

    def search(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(chunk_id, score)`` for ``query`` (higher = better)."""
        ...

    def fetch(self, ids: Sequence[str]) -> dict[str, DocumentChunk]:
        """Materialize the stored ``DocumentChunk`` for each id present (for fusion)."""
        ...

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Subset of ``ids`` already indexed (incremental refresh)."""
        ...

    def count(self) -> int:
        """Number of indexed chunks."""
        ...


def _passes(chunk: DocumentChunk, where: ChunkFilter | None) -> bool:
    """In-Python metadata predicate (the in-memory backend; FTS5 does this in SQL)."""
    return where is None or where.matches(chunk)


# --------------------------------------------------------------------------- #
# SQLite FTS5 backend (default, persistent)
# --------------------------------------------------------------------------- #


def fts5_available() -> bool:
    """Whether this Python's ``sqlite3`` was compiled with the FTS5 extension."""
    try:
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
        finally:
            con.close()
        return True
    except sqlite3.OperationalError:
        return False


# Columns carried UNINDEXED in the FTS5 table: stored + filterable, but not full-text searched
# (only ``text`` is). ``filing_date`` is ISO ``YYYY-MM-DD`` so lexical compare == date compare.
_META_COLS = ("chunk_id", "document_id", "chunk_index", "ticker", "document_type", "source",
              "source_url", "filing_date", "section")


class Fts5SparseStore:
    """Persistent BM25 index in a SQLite FTS5 virtual table (one row per chunk).

    ``text`` is the only indexed column; the rest are ``UNINDEXED`` (stored for citation + filtered
    via SQL ``WHERE``). ``add`` deletes-then-inserts by ``chunk_id`` so re-ingesting a chunk is an
    upsert (idempotent). The query is reduced to its alphanumeric tokens OR-ed together — recall
    over a long natural-language question, and robust to FTS5's MATCH operator syntax.
    """

    name = "fts5"

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the agent may query from a worker thread; access is serialized.
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        cols = ", ".join(c if c == "text" else f"{c} UNINDEXED" for c in ("text", *_META_COLS))
        # unicode61 tokenizer; keep '.' and '-' as token chars so "U.S." / "10-K" survive somewhat.
        self._con.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            f"{cols}, tokenize=\"unicode61 tokenchars '.-'\")"
        )
        self._con.commit()

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        placeholders = ",".join("?" * len(ids))
        # Upsert: drop any prior rows for these ids, then insert fresh (idempotent re-ingest).
        self._con.execute(f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})", ids)
        self._con.executemany(
            f"INSERT INTO chunks (text, {', '.join(_META_COLS)}) "
            f"VALUES ({', '.join('?' * (1 + len(_META_COLS)))})",
            [
                (
                    c.text, c.chunk_id, c.document_id, c.chunk_index, c.ticker, c.document_type,
                    c.source, c.source_url, c.filing_date.isoformat(), c.section,
                )
                for c in chunks
            ],
        )
        self._con.commit()

    def _where_sql(self, where: ChunkFilter | None) -> tuple[str, list[object]]:
        """Translate a ``ChunkFilter`` into an SQL ``AND ...`` fragment + bound params."""
        if where is None:
            return "", []
        clauses: list[str] = []
        params: list[object] = []
        if where.ticker is not None:
            clauses.append("ticker = ?")
            params.append(where.ticker)
        if where.document_types is not None:
            qs = ",".join("?" * len(where.document_types))
            clauses.append(f"document_type IN ({qs})")
            params.extend(where.document_types)
        if where.sections is not None:
            qs = ",".join("?" * len(where.sections))
            clauses.append(f"section IN ({qs})")
            params.extend(where.sections)
        if where.date_from is not None:  # ISO strings compare like dates
            clauses.append("filing_date >= ?")
            params.append(where.date_from.isoformat())
        if where.date_to is not None:
            clauses.append("filing_date <= ?")
            params.append(where.date_to.isoformat())
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def search(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> list[tuple[str, float]]:
        tokens = _tokenize(query)
        if not tokens or top_k <= 0:
            return []
        match = " OR ".join(tokens)  # any-term match → recall-friendly for NL questions
        where_sql, params = self._where_sql(where)
        # bm25() is more-negative = more-relevant; negate so higher = better (codebase convention).
        rows = self._con.execute(
            f"SELECT chunk_id, -bm25(chunks) AS score FROM chunks "
            f"WHERE chunks MATCH ?{where_sql} ORDER BY bm25(chunks) LIMIT ?",
            [match, *params, top_k],
        ).fetchall()
        return [(str(cid), float(score)) for cid, score in rows]

    def fetch(self, ids: Sequence[str]) -> dict[str, DocumentChunk]:
        if not ids:
            return {}
        qs = ",".join("?" * len(ids))
        rows = self._con.execute(
            f"SELECT {', '.join(_META_COLS)}, text FROM chunks WHERE chunk_id IN ({qs})", list(ids)
        ).fetchall()
        out: dict[str, DocumentChunk] = {}
        for row in rows:
            md = dict(zip(_META_COLS, row[:-1], strict=True))
            out[str(md["chunk_id"])] = DocumentChunk(
                chunk_id=str(md["chunk_id"]),
                document_id=str(md["document_id"]),
                chunk_index=int(md["chunk_index"]),
                text=str(row[-1]),
                ticker=str(md["ticker"]),
                document_type=str(md["document_type"]),
                source=str(md["source"]),
                source_url=str(md["source_url"]),
                filing_date=Date.fromisoformat(str(md["filing_date"])),
                section=str(md["section"]) if md["section"] is not None else None,
            )
        return out

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        qs = ",".join("?" * len(ids))
        rows = self._con.execute(
            f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({qs})", list(ids)
        ).fetchall()
        return {str(r[0]) for r in rows}

    def count(self) -> int:
        return int(self._con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


# --------------------------------------------------------------------------- #
# In-memory pure-Python BM25 backend (parity / tests / no-FTS5 fallback)
# --------------------------------------------------------------------------- #


class InMemoryBM25Store:
    """Okapi BM25 over an in-memory inverted index — no sqlite, fully deterministic.

    Scores ``s(q,d) = sum_{t in q} IDF(t) * (f*(k1+1)) / (f + k1*(1 - b + b*|d|/avgdl))`` with the
    standard ``k1=1.5``, ``b=0.75`` and ``IDF(t) = ln((N - n_t + 0.5)/(n_t + 0.5) + 1)`` (the ``+1``
    keeps IDF non-negative). Used for tests, backend parity, and when sqlite lacks FTS5; not
    persistent, so production uses ``Fts5SparseStore``.
    """

    name = "bm25-mem"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: dict[str, DocumentChunk] = {}
        self._tokens: dict[str, list[str]] = {}  # chunk_id -> token list

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        for c in chunks:  # dict assignment = idempotent upsert by chunk_id
            self._chunks[c.chunk_id] = c
            self._tokens[c.chunk_id] = _tokenize(c.text)

    def _idf(self, term: str, doc_freq: dict[str, int], n_docs: int) -> float:
        n_t = doc_freq.get(term, 0)
        return math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def search(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> list[tuple[str, float]]:
        q_terms = _tokenize(query)
        if not q_terms or top_k <= 0:
            return []
        # Restrict the corpus to chunks passing the filter (BM25 stats computed over that subset,
        # mirroring how the FTS5 backend ranks only the filtered rows).
        cids = [cid for cid, c in self._chunks.items() if _passes(c, where)]
        if not cids:
            return []
        n_docs = len(cids)
        avgdl = sum(len(self._tokens[cid]) for cid in cids) / n_docs
        doc_freq: dict[str, int] = {}
        for term in set(q_terms):
            doc_freq[term] = sum(1 for cid in cids if term in self._tokens[cid])
        scored: list[tuple[str, float]] = []
        for cid in cids:
            toks = self._tokens[cid]
            dl = len(toks)
            score = 0.0
            for term in q_terms:
                f = toks.count(term)
                if f == 0:
                    continue
                idf = self._idf(term, doc_freq, n_docs)
                denom = f + self._k1 * (1 - self._b + self._b * dl / avgdl)
                score += idf * (f * (self._k1 + 1)) / denom
            if score > 0:
                scored.append((cid, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def fetch(self, ids: Sequence[str]) -> dict[str, DocumentChunk]:
        return {cid: self._chunks[cid] for cid in ids if cid in self._chunks}

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        return {cid for cid in ids if cid in self._chunks}

    def count(self) -> int:
        return len(self._chunks)


def _slug(text: str) -> str:
    """Filesystem-safe slug of an embedder namespace (mirrors the vector store's collection)."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def build_sparse_store(settings: Settings) -> SparseStore:
    """Select the sparse backend: persistent FTS5 if available, else in-memory BM25 (fallback).

    The DB is namespaced per embedder (parallel to the vector store's collection) so corpora chunked
    by different configs never mix.
    """
    from stock_agent.rag.embeddings import embedding_namespace

    if not fts5_available():  # pragma: no cover - environment-dependent
        return InMemoryBM25Store()
    return Fts5SparseStore(settings.sparse_store_dir / f"{_slug(embedding_namespace(settings))}.db")
