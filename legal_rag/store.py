"""SQLite store — docs, chunks, bbox, version counter."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import CFG

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    sha256 TEXT UNIQUE NOT NULL,
    pages INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    added_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'corpus'
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    boxes TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    p = db_path or CFG.db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# ---- version counter (cache invalidation) ----
def version(conn) -> int:
    r = conn.execute("SELECT v FROM meta WHERE k='version'").fetchone()
    return int(r[0]) if r else 0


def bump(conn) -> None:
    conn.execute(
        "INSERT INTO meta(k,v) VALUES('version','1') "
        "ON CONFLICT(k) DO UPDATE SET v = CAST(CAST(v AS INTEGER)+1 AS TEXT)"
    )
    conn.commit()


# ---- docs ----
def insert_doc(conn, path, title, sha, pages, size, source="corpus") -> int:
    cur = conn.execute(
        "INSERT INTO docs(path,title,sha256,pages,size_bytes,added_at,source) "
        "VALUES (?,?,?,?,?,?,?)",
        (path, title, sha, pages, size, datetime.now(timezone.utc).isoformat(), source),
    )
    return cur.lastrowid


def doc_by_sha(conn, sha):
    return conn.execute("SELECT id FROM docs WHERE sha256=?", (sha,)).fetchone()


def list_docs(conn):
    rows = conn.execute(
        "SELECT d.id, d.title, d.pages, d.size_bytes, d.added_at, d.source, "
        "COUNT(c.id) FROM docs d LEFT JOIN chunks c ON c.doc_id=d.id "
        "GROUP BY d.id ORDER BY d.added_at DESC"
    ).fetchall()
    return [
        {"id": r[0], "title": r[1], "pages": r[2], "size_mb": r[3] / 1e6,
         "added_at": r[4], "source": r[5], "chunks": r[6]}
        for r in rows
    ]


def get_doc(conn, doc_id):
    r = conn.execute("SELECT id, path, title, sha256 FROM docs WHERE id=?", (doc_id,)).fetchone()
    return {"id": r[0], "path": r[1], "title": r[2], "sha256": r[3]} if r else None


def doc_chunk_ids(conn, doc_id) -> list[int]:
    return [r[0] for r in conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]


def delete_doc(conn, doc_id) -> list[int]:
    ids = doc_chunk_ids(conn, doc_id)
    conn.execute("DELETE FROM docs WHERE id=?", (doc_id,))
    conn.commit()
    return ids


# ---- chunks ----
def insert_chunk(conn, doc_id, page, ordinal, heading, text, boxes) -> int:
    cur = conn.execute(
        "INSERT INTO chunks(doc_id,page,ordinal,heading,text,boxes) VALUES (?,?,?,?,?,?)",
        (doc_id, page, ordinal, heading, text, json.dumps(boxes)),
    )
    return cur.lastrowid


def _row(r):
    return {"id": r[0], "text": r[1], "page": r[2], "boxes": json.loads(r[3]),
            "path": r[4], "title": r[5], "doc_id": r[6], "ordinal": r[7], "heading": r[8]}


BASE = ("SELECT c.id, c.text, c.page, c.boxes, d.path, d.title, c.doc_id, c.ordinal, c.heading "
        "FROM chunks c JOIN docs d ON d.id=c.doc_id ")


def all_chunks(conn, doc_ids=None):
    if doc_ids:
        marks = ",".join("?" * len(doc_ids))
        rows = conn.execute(BASE + f"WHERE c.doc_id IN ({marks}) ORDER BY c.id", list(doc_ids))
    else:
        rows = conn.execute(BASE + "ORDER BY c.id")
    return [_row(r) for r in rows.fetchall()]


def get_chunks(conn, ids):
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    found = {r[0]: _row(r) for r in conn.execute(BASE + f"WHERE c.id IN ({marks})", list(ids))}
    return [found[i] for i in ids if i in found]


def neighbours(conn, doc_id, ordinal, span=1):
    rows = conn.execute(
        BASE + "WHERE c.doc_id=? AND c.ordinal BETWEEN ? AND ? AND c.id NOT IN (?) ORDER BY c.ordinal",
        (doc_id, ordinal - span, ordinal + span, -1),
    ).fetchall()
    return [_row(r) for r in rows]