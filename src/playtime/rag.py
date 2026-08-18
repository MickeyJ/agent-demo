"""Stage 6: mini RAG — the whole pipeline in one readable file.

    ingest:  text ──chunk──▶ pieces ──embed──▶ vectors ──▶ sqlite (var/rag.sqlite)
    search:  query ──embed──▶ vector ──cosine vs all──▶ top-k chunks

That's all RAG is: nearest-neighbor search over embedded text, whose hits get
pasted into the model's context as a tool result. No vector DB — numpy cosine
over a few hundred chunks is instant; you graduate to a real index (pgvector,
qdrant) when the corpus outgrows RAM, not before.

Ingest is idempotent by title: re-submitting a doc replaces it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from playtime.llm import make_embeddings
from playtime.toolkit import ToolSpec

DB = Path("var/rag.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    title TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    Path("var").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Paragraph-based chunking: split on blank lines, greedily merge small
    paragraphs, hard-split monsters. Chunk size is THE RAG tuning knob — too
    big and hits drown the context; too small and they lose their meaning."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > max_chars:  # single paragraph over budget
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def ingest(title: str, text: str) -> dict:
    """The write path: chunk → embed (one batched call) → store. Returns a
    small report so callers (CLI, web UI) can show what happened."""
    title = title.strip() or "untitled"
    chunks = chunk_text(text)
    if not chunks:
        return {"title": title, "chunks": 0}
    vectors = make_embeddings().embed_documents(chunks)
    with _conn() as conn:
        conn.execute("DELETE FROM docs WHERE title = ?", (title,))  # replace-by-title
        doc_id = conn.execute("INSERT INTO docs (title) VALUES (?)", (title,)).lastrowid
        conn.executemany(
            "INSERT INTO chunks (doc_id, seq, text, embedding) VALUES (?, ?, ?, ?)",
            [
                (doc_id, i, c, np.asarray(v, dtype=np.float32).tobytes())
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ],
        )
    return {"title": title, "chunks": len(chunks)}


def search(query: str, k: int = 3) -> list[dict]:
    """The read path: embed the query, cosine against every stored chunk."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT d.title, c.seq, c.text, c.embedding FROM chunks c JOIN docs d ON d.id = c.doc_id"
        ).fetchall()
    if not rows:
        return []
    q = np.asarray(make_embeddings().embed_query(query), dtype=np.float32)
    mat = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    scores = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
    top = np.argsort(scores)[::-1][:k]
    return [
        {"title": rows[i][0], "seq": rows[i][1], "text": rows[i][2], "score": float(scores[i])}
        for i in top
    ]


def list_docs() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT d.title, d.created_at, COUNT(c.id) FROM docs d "
            "LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.created_at"
        ).fetchall()
    return [{"title": t, "created_at": ts, "chunks": n} for t, ts, n in rows]


def delete_doc(title: str) -> bool:
    with _conn() as conn:
        return conn.execute("DELETE FROM docs WHERE title = ?", (title,)).rowcount > 0


# --- the tool the master agent sees -----------------------------------------


class SearchDocsArgs(BaseModel):
    query: str = Field(description="What to look for — a natural-language question or keywords.")


def _search_docs(a: SearchDocsArgs) -> str:
    hits = search(a.query, k=3)
    if not hits:
        return "The knowledge base has no documents matching this query."
    return "\n---\n".join(
        f"[{h['title']} §{h['seq']}] (score {h['score']:.2f})\n{h['text']}" for h in hits
    )


SEARCH_DOCS = ToolSpec(
    name="search_docs",
    description=(
        "Search the local knowledge base of stored documents (company wiki, notes, "
        "anything uploaded). Use for ANY question about facts you don't know — "
        "always search before saying you don't know. Returns the best-matching excerpts."
    ),
    args_model=SearchDocsArgs,
    fn=_search_docs,
    timeout_s=30.0,
)
