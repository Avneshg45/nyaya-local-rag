"""Streamlit UI: upload, doc manager, delete, scoped chat, page highlight."""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

default_gguf = ROOT / "legalrag_data" / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
if not os.environ.get("GGUF_PATH") and default_gguf.exists():
    os.environ["GGUF_PATH"] = str(default_gguf)

import fitz
import streamlit as st

from legal_rag import guard, ingest, retrieve, store
from legal_rag.config import CFG
from legal_rag.generate import generate, gloss_to_english, paraphrase

st.set_page_config(
    page_title="Nyaya Local",
    page_icon="⚖",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background: #f7f8fb; border-right: 1px solid #e7e9ef; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #1f2937; }
    [data-testid="stAppViewContainer"] { background: #fbfcfe; }
    [data-testid="stHeader"] { background: rgba(251,252,254,0.8); }
    [data-testid="stChatMessage"] {
        border: 1px solid #e4e7ed;
        border-radius: 14px;
        background: white;
        box-shadow: 0 2px 12px rgba(30, 41, 59, 0.04);
        padding: 0.35rem 0.6rem;
    }
    .brand { display:flex; align-items:center; gap:.65rem; margin-bottom:.15rem; }
    .brand-mark {
        width:42px; height:42px; border-radius:12px;
        display:grid; place-items:center; background:#1d4ed8; color:white;
        font-size:22px;
    }
    .brand-title { font-size:1.5rem; font-weight:750; color:#172033; line-height:1.1; }
    .brand-sub { color:#64748b; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="brand"><div class="brand-mark">⚖</div>'
    '<div><div class="brand-title">Nyaya Local</div>'
    '<div class="brand-sub">Offline legal document assistant</div></div></div>',
    unsafe_allow_html=True,
)
st.caption("Answers are generated only from your uploaded PDFs, with matching PDF passages highlighted.")

for k, v in [("history", []), ("flash", None)]:
    st.session_state.setdefault(k, v)


def refresh():
    retrieve.invalidate()
    st.cache_data.clear()


def render_page(path, page_no, boxes, highlight_text=""):
    try:
        doc = fitz.open(path)
        page_idx = min(max(int(page_no), 0), doc.page_count - 1)
        page = doc[page_idx]
        for r in boxes.get(str(page_idx), boxes.get(page_idx, [])):
            page.add_highlight_annot(fitz.Rect(r))
        if highlight_text:
            clean = re.sub(r"\[[^\]]+\]", " ", highlight_text)
            clean = re.sub(r"\s+", " ", clean).strip()
            terms = []
            if clean:
                terms.append(clean)
            terms.extend(re.findall(r"[A-Za-z][A-Za-z'-]{5,}", clean))
            terms.extend(re.findall(r"[\u0900-\u097F][\u0900-\u097F\s]{7,}", clean))
            for term in list(dict.fromkeys([t.strip() for t in terms if len(t.strip()) >= 8]))[:50]:
                for rect in page.search_for(term):
                    page.add_highlight_annot(rect)
        st.image(
            page.get_pixmap(dpi=130).tobytes("png"),
            caption=f"{Path(path).name} — page {page_idx + 1}",
        )
        doc.close()
    except Exception as exc:
        st.warning(f"page render fail: {exc}")


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Documents")

    ups = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if ups and st.button(f"Index {len(ups)} file{'s' if len(ups) != 1 else ''}", type="primary"):
        if len(ups) > CFG.max_batch_files:
            st.error(f"Maximum {CFG.max_batch_files} files can be indexed at once")
        else:
            bar = st.progress(0.0, "Indexing PDFs...")
            ok = bad = 0
            for i, f in enumerate(ups, 1):
                r = ingest.add_pdf(f.getvalue(), f.name)
                if r["ok"]:
                    ok += 1
                else:
                    bad += 1
                    st.warning(r["reason"])
                bar.progress(i / len(ups), f"{i}/{len(ups)}")
            bar.empty()
            refresh()
            st.session_state.flash = f"{ok} document(s) indexed, {bad} failed"
            st.rerun()

    conn = store.connect()
    docs = store.list_docs(conn)
    conn.close()

    if docs:
        st.caption(f"{len(docs)} documents · {sum(d['chunks'] for d in docs)} passages")
        for d in docs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"**{d['title'][:34]}**  \n<sub>{d['pages']}p · "
                f"{d['chunks']} chunks · {d['source']}</sub>",
                unsafe_allow_html=True,
            )
            if c2.button("🗑", key=f"del{d['id']}", help="Delete document"):
                r = ingest.delete_doc(d["id"])
                refresh()
                st.session_state.flash = (
                    f"Deleted: {r.get('title', '?')} ({r.get('chunks', 0)} passages)"
                )
                st.rerun()

        scope = st.multiselect(
            "Search scope (blank = all documents)",
            options=[d["id"] for d in docs],
            format_func=lambda i: next(d["title"] for d in docs if d["id"] == i),
        )
        if st.button("Rebuild index"):
            st.info(ingest.rebuild())
            refresh()
    else:
        scope = []
        st.info("No documents yet. Upload a PDF to begin.")

    with st.expander("settings"):
        st.write(f"Reranker: {'on' if CFG.use_reranker else 'off'} · Multi-query: {'on' if CFG.use_multiquery else 'off'}")
        st.write(f"Minimum dense score: {CFG.min_dense_score} · Entailment: {CFG.entail_threshold}")
        st.write(f"Answer passages: {CFG.final_k}")

if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

# ---------------- chat ----------------
with st.form("nyaya_form", clear_on_submit=True):
    q_input = st.text_input("Ask a question (Hindi or English)")
    submitted = st.form_submit_button("Search documents", type="primary")

q = q_input.strip() if (submitted and q_input and q_input.strip()) else None

if q:
    if not docs:
        st.warning("Upload a PDF before asking questions.")
    else:
        with st.spinner("Searching the PDFs..."):
            gloss = gloss_to_english(q) if retrieve.has_devanagari(q) else None
            extras = paraphrase(q) if CFG.use_multiquery else []
            chunks, top_dense = retrieve.search(
                q,
                lexical_query=" ".join([gloss or ""] + extras).strip() or None,
                doc_ids=scope or None,
            )
            if extras:
                more, _ = retrieve.search(extras[0], doc_ids=scope or None)
                seen = {c["id"] for c in chunks}
                chunks += [c for c in more if c["id"] not in seen][:3]

        with st.spinner("Creating an answer in the question's language from PDF passages..."):
            res = generate(q, chunks)

        with st.spinner("Checking grounding and citations..."):
            kept, dropped, stats = guard.check(res["answer"], chunks)

        if not chunks:
            answer, cites = "**Information not found** — the PDF index has no matching passages.", []
        elif guard.should_refuse(res["found"], top_dense, stats) or not kept:
            answer = "**Information not found** — no grounded answer was found in the uploaded documents."
            cites = []
        else:
            answer = "\n".join(s["text"] for s in kept)
            cites = sorted({c for s in kept for c in s["citations"]})

        st.session_state.history.append({
            "q": q,
            "a": answer,
            "chunks": chunks,
            "kept": kept,
            "dropped": dropped,
            "stats": stats,
            "top": top_dense,
            "cites": cites,
        })

for t in st.session_state.history:
    with st.chat_message("user"):
        st.write(t["q"])
    with st.chat_message("assistant"):
        st.write(t["a"])
        m = t["stats"]
        st.caption(f"Top similarity {t['top']:.3f} · grounded statements {m['kept']}/{m['total']}")
        if t["dropped"]:
            with st.expander(f"{m['dropped']} unsupported statement(s) removed"):
                for d in t["dropped"]:
                    st.write(f"- {d['text']} (entailment {d['entail']})")
        for i in t["cites"]:
            if 1 <= i <= len(t["chunks"]):
                c = t["chunks"][i - 1]
                with st.expander(f"[{i}] {c['title']} — p{c['page'] + 1}"):
                    st.write(c["text"])
                    matching_text = "\n".join(
                        s["text"] for s in t["kept"] if i in s["citations"]
                    )
                    render_page(c["path"], c["page"], c["boxes"], matching_text)
