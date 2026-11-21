"""Model singletons â€” ek hi baar load, sab modules share karte hain."""
import threading

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import CFG

_lock = threading.Lock()
_embed = _rerank = _nli = _nli_tok = _llm = None


def get_embedder() -> SentenceTransformer:
    global _embed
    with _lock:
        if _embed is None:
            _embed = SentenceTransformer(CFG.embed_model, device="cpu")
    return _embed


def embed_dim() -> int:
    return get_embedder().get_sentence_embedding_dimension()


def get_reranker() -> CrossEncoder:
    global _rerank
    with _lock:
        if _rerank is None:
            _rerank = CrossEncoder(CFG.rerank_model, max_length=512, device="cpu")
    return _rerank


def get_nli():
    global _nli, _nli_tok
    with _lock:
        if _nli is None:
            _nli_tok = AutoTokenizer.from_pretrained(CFG.nli_model)
            _nli = AutoModelForSequenceClassification.from_pretrained(CFG.nli_model).eval()
    return _nli_tok, _nli


def get_llm():
    global _llm
    with _lock:
        if _llm is None:
            from llama_cpp import Llama
            if not CFG.gguf_path:
                raise RuntimeError("GGUF_PATH set nahi hai")
            _llm = Llama(model_path=CFG.gguf_path, n_ctx=CFG.n_ctx,
                         n_threads=CFG.n_threads, n_batch=256, verbose=False)
    return _llm


def free_reranker():
    global _rerank
    _rerank = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
