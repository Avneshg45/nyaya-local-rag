"""Hybrid retrieval: multi-query + BM25 + dense + RRF + rerank + dedupe + neighbours."""
import re
import threading

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import store
from .config import CFG
from .models import get_embedder, get_reranker

TOKEN = re.compile(r"[\u0900-\u097F]+|[A-Za-z0-9]+")
DEVA = re.compile(r"[\u0900-\u097F]")

_c = {"v": None, "bm25": None, "ids": None, "index": None, "docs": None}
_lock = threading.Lock()


def tokenize(t):
    return [x.lower() for x in TOKEN.findall(t)]


def has_devanagari(q):
    return bool(DEVA.search(q))


def _load(doc_ids=None):
    with _lock:
        key = (store.version(store.connect()), tuple(doc_ids or ()))
        if _c["v"] == key:
            return
        conn = store.connect()
        rows = store.all_chunks(conn, doc_ids)
        conn.close()
        _c["ids"] = [r["id"] for r in rows]
        _c["docs"] = {r["id"]: r for r in rows}
        _c["bm25"] = BM25Okapi([tokenize(r["text"]) for r in rows]) if rows else None
        _c["index"] = faiss.read_index(str(CFG.faiss_path)) if CFG.faiss_path.exists() else None
        _c["v"] = key


def invalidate():
    _c["v"] = None


def dense_search(query, k, allowed=None):
    _load()
    if _c["index"] is None or _c["index"].ntotal == 0:
        return []
    vec = get_embedder().encode(["query: " + query], normalize_embeddings=True,
                                convert_to_numpy=True).astype("float32")
    scores, idx = _c["index"].search(vec, min(k * 3, _c["index"].ntotal))
    out = []
    for s, i in zip(scores[0], idx[0]):
        if i < 0:
            continue
        cid = int(i)
        if allowed is not None and cid not in allowed:
            continue
        out.append((cid, float(s)))
        if len(out) >= k:
            break
    return out


def bm25_search(query, k, allowed=None):
    _load()
    if _c["bm25"] is None:
        return []
    scores = _c["bm25"].get_scores(tokenize(query))
    order = np.argsort(scores)[::-1]
    out = []
    for i in order:
        if scores[i] <= 0:
            break
        cid = _c["ids"][i]
        if allowed is not None and cid not in allowed:
            continue
        out.append((cid, float(scores[i])))
        if len(out) >= k:
            break
    return out


def rrf(lists, k=60):
    fused = {}
    for lst in lists:
        for rank, (cid, _) in enumerate(lst, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def _jaccard(a, b):
    A, B = set(tokenize(a)), set(tokenize(b))
    return len(A & B) / max(len(A | B), 1)


def _dedupe(chunks, limit):
    out = []
    for c in chunks:
        if all(_jaccard(c["text"], k["text"]) < CFG.dedupe_jaccard for k in out):
            out.append(c)
        if len(out) >= limit:
            break
    return out


def search(query, lexical_query=None, doc_ids=None, final_k=None):
    """Returns (chunks, top_dense). chunks me neighbour context add hai."""
    _load(doc_ids)
    allowed = set(_c["ids"]) if doc_ids else None
    k_final = min(final_k or CFG.final_k, CFG.final_k_max)

    queries = [query]
    if lexical_query:
        queries.append(lexical_query)

    lists = []
    for q in queries:
        lists.append(dense_search(q, CFG.top_k_dense, allowed))
        lists.append(bm25_search(q, CFG.top_k_bm25, allowed))

    pool_ids = [cid for cid, _ in rrf(lists, CFG.rrf_k)[: CFG.rerank_pool]]
    dense_map = {cid: s for lst in lists[:2] for cid, s in lst}
    top_dense = max(dense_map.values(), default=0.0)

    conn = store.connect()
    chunks = store.get_chunks(conn, pool_ids)
    if not chunks:
        conn.close()
        return [], top_dense

    if CFG.use_reranker and len(chunks) > 1:
        pairs = [(query, c["text"][:900]) for c in chunks]
        scores = get_reranker().predict(pairs, show_progress_bar=False)
        for c, s in zip(chunks, scores):
            c["rerank"] = float(s)
        chunks.sort(key=lambda c: c["rerank"], reverse=True)

    chunks = _dedupe(chunks, k_final)

    for c in chunks:
        c["dense_score"] = dense_map.get(c["id"], 0.0)
        ctx = store.neighbours(conn, c["doc_id"], c["ordinal"])
        c["context"] = " ".join(n["text"] for n in ctx)[: CFG.neighbour_chars] if ctx else ""
    conn.close()
    return chunks, top_dense