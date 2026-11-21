"""Index build / incremental add / delete.

CLI:
  python -m legal_rag.ingest build  ./corpus      # fresh
  python -m legal_rag.ingest add    a.pdf b.pdf   # incremental
  python -m legal_rag.ingest delete 7             # doc id
  python -m legal_rag.ingest rebuild              # DB se poora index
  python -m legal_rag.ingest list
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import faiss
import fitz
import numpy as np
from tqdm import tqdm

from . import store
from .config import CFG
from .models import embed_dim, get_embedder

WS = re.compile(r"\s+")
SECTION = re.compile(
    r"^(section\s+\d|sec\.\s*\d|धारा\s*\d|article\s+\d|अनुच्छेद\s*\d|"
    r"rule\s+\d|नियम\s*\d|chapter\s+[ivx\d]|अध्याय\s+)", re.I)
HEADING = re.compile(r"^(\d+(\.\d+)*\.?\s+)?[A-Z\u0900-\u097F][^.]{2,90}$")


def norm(s: str) -> str:
    return WS.sub(" ", s).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_blocks(path: Path):
    doc = fitz.open(path)
    blocks = []
    for pno in range(doc.page_count):
        for b in doc[pno].get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            spans = [sp["text"] for ln in b["lines"] for sp in ln["spans"] if sp["text"].strip()]
            if not spans:
                continue
            text = norm(" ".join(spans))
            if len(text) < 2:
                continue
            blocks.append({"page": pno, "text": text, "bbox": [round(v, 2) for v in b["bbox"]]})
    pages = doc.page_count
    doc.close()
    return pages, blocks


def chunk_blocks(blocks, chunk_chars: int, overlap: int):
    """Structure-aware: section markers aur headings pe split, overlap carry-over."""
    raw, cur, cur_len = [], [], 0
    heading = ""

    def flush(keep_overlap: bool):
        nonlocal cur, cur_len
        if cur:
            raw.append((heading, cur))
        keep, klen = [], 0
        if keep_overlap:
            for prev in reversed(cur):
                if klen + len(prev["text"]) > overlap:
                    break
                keep.insert(0, prev)
                klen += len(prev["text"])
        cur, cur_len = keep, klen

    for b in blocks:
        t = b["text"]
        is_heading = len(t) < 100 and HEADING.match(t) and not SECTION.match(t)
        if is_heading and len(t) < 100:
            heading = t
        starts_section = bool(SECTION.match(t))
        over = cur_len + len(t) > chunk_chars
        if cur and (over or (starts_section and cur_len > chunk_chars * 0.4)):
            flush(keep_overlap=True)
        cur.append(b)
        cur_len += len(t)

    if cur:
        raw.append((heading, cur))

    out = []
    for head, group in raw:
        text = norm(" ".join(x["text"] for x in group))
        if len(text) < CFG.min_chunk_chars:
            continue
        boxes = {}
        for x in group:
            boxes.setdefault(x["page"], []).append(x["bbox"])
        out.append({"text": text, "pages": sorted(boxes), "boxes": boxes, "heading": head})
    return out


def load_index(create: bool = False):
    if CFG.faiss_path.exists():
        return faiss.read_index(str(CFG.faiss_path))
    if not create:
        return None
    CFG.index_dir.mkdir(parents=True, exist_ok=True)
    return faiss.IndexIDMap2(faiss.IndexFlatIP(embed_dim()))


def save_index(index) -> None:
    CFG.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(CFG.faiss_path))


def _validate(data: bytes, filename: str):
    if not data.startswith(b"%PDF"):
        return f"{filename}: PDF magic bytes nahi mile"
    if len(data) > CFG.max_file_mb * 1_000_000:
        return f"{filename}: {CFG.max_file_mb}MB se bada"
    return None


def add_pdf(data: bytes, filename: str, source: str = "upload", dest_dir: Path | None = None):
    """Ek PDF ko index me add karo. Returns dict(ok, doc_id, chunks, reason)."""
    err = _validate(data, filename)
    if err:
        return {"ok": False, "reason": err}

    digest = sha256_bytes(data)
    conn = store.connect()
    if store.doc_by_sha(conn, digest):
        conn.close()
        return {"ok": False, "reason": f"{filename}: already indexed (same file)"}

    dest = dest_dir or CFG.upload_dir
    dest.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name) or "doc.pdf"
    path = dest / f"{digest[:10]}_{safe}"
    path.write_bytes(data)

    try:
        pages, blocks = extract_blocks(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        conn.close()
        return {"ok": False, "reason": f"{filename}: parse fail ({exc})"}

    if pages > CFG.max_pages:
        path.unlink(missing_ok=True)
        conn.close()
        return {"ok": False, "reason": f"{filename}: {pages} pages (limit {CFG.max_pages})"}
    if len(blocks) < 3:
        path.unlink(missing_ok=True)
        conn.close()
        return {"ok": False, "reason": f"{filename}: text layer nahi (scanned?) — OCR chahiye"}

    title = Path(filename).stem[:120]
    doc_id = store.insert_doc(conn, str(path), title, digest, pages, len(data), source)

    chunks = chunk_blocks(blocks, CFG.chunk_chars, CFG.chunk_overlap)
    if not chunks:
        store.delete_doc(conn, doc_id)
        path.unlink(missing_ok=True)
        conn.close()
        return {"ok": False, "reason": f"{filename}: chunk nahi bana"}

    ids, embed_texts = [], []
    for i, ch in enumerate(chunks):
        cid = store.insert_chunk(conn, doc_id, ch["pages"][0], i, ch["heading"],
                                 ch["text"], ch["boxes"])
        ids.append(cid)
        # contextual retrieval: title + heading chunk ke saath embed
        prefix = " | ".join(x for x in (title, ch["heading"]) if x)
        embed_texts.append(f"passage: {prefix} | {ch['text']}")
    conn.commit()

    index = load_index(create=True)
    vecs = get_embedder().encode(
        embed_texts, batch_size=16, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype("float32")
    index.add_with_ids(vecs, np.array(ids, dtype="int64"))
    save_index(index)

    store.bump(conn)
    conn.close()
    return {"ok": True, "doc_id": doc_id, "chunks": len(ids), "title": title}


def delete_doc(doc_id: int):
    conn = store.connect()
    doc = store.get_doc(conn, doc_id)
    if not doc:
        conn.close()
        return {"ok": False, "reason": "doc nahi mila"}
    ids = store.delete_doc(conn, doc_id)

    index = load_index()
    if index is not None and ids:
        index.remove_ids(np.array(ids, dtype="int64"))
        save_index(index)

    Path(doc["path"]).unlink(missing_ok=True)
    store.bump(conn)
    conn.close()
    return {"ok": True, "chunks": len(ids), "title": doc["title"]}


def rebuild():
    """DB se poora FAISS dobara — delete crash ke baad recovery."""
    conn = store.connect()
    rows = store.all_chunks(conn)
    if not rows:
        conn.close()
        return {"ok": False, "reason": "DB khaali"}
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embed_dim()))
    vecs = get_embedder().encode(
        ["passage: " + r["text"] for r in rows], batch_size=16,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
    ).astype("float32")
    index.add_with_ids(vecs, np.array([r["id"] for r in rows], dtype="int64"))
    save_index(index)
    store.bump(conn)
    conn.close()
    return {"ok": True, "chunks": len(rows)}


def build_dir(corpus: Path):
    # A fresh build must reset both the SQLite metadata and the FAISS index.
    # Resetting only FAISS while leaving SQLite would make every PDF look
    # "already indexed" on the next build.
    CFG.index_dir.mkdir(parents=True, exist_ok=True)
    if CFG.faiss_path.exists():
        CFG.faiss_path.unlink()
    if CFG.db_path.exists():
        CFG.db_path.unlink()
    conn = store.connect()
    conn.close()
    pdfs = sorted(corpus.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"PDF nahi mile: {corpus}")
    for p in tqdm(pdfs, desc="ingest"):
        r = add_pdf(p.read_bytes(), p.name, source="corpus", dest_dir=CFG.corpus_dir)
        if not r["ok"]:
            tqdm.write("  skip: " + r["reason"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "add", "delete", "rebuild", "list"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()

    if a.cmd == "build":
        build_dir(Path(a.args[0]))
    elif a.cmd == "add":
        for p in a.args:
            print(add_pdf(Path(p).read_bytes(), Path(p).name))
    elif a.cmd == "delete":
        for i in a.args:
            print(delete_doc(int(i)))
    elif a.cmd == "rebuild":
        print(rebuild())
    else:
        conn = store.connect()
        for d in store.list_docs(conn):
            print(f"{d['id']:>4}  {d['chunks']:>4} chunks  {d['title'][:60]}")