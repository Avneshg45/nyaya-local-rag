"""Central config. Sab env var se override."""
import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "legalrag_data"
ROOT = Path(os.environ.get("LEGALRAG_HOME", _DEFAULT_ROOT)).expanduser().resolve()


@dataclass(frozen=True)
class Config:
    data_dir: Path = ROOT
    corpus_dir: Path = ROOT / "corpus"
    upload_dir: Path = ROOT / "uploads"
    index_dir: Path = ROOT / "index"
    db_path: Path = ROOT / "index" / "chunks.sqlite3"
    faiss_path: Path = ROOT / "index" / "dense.faiss"

    embed_model: str = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
    nli_model: str = os.environ.get("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    rerank_model: str = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    gguf_path: str = os.environ.get(
        "GGUF_PATH",
        str(ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"),
    )

    chunk_chars: int = int(os.environ.get("CHUNK_CHARS", "900"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "150"))
    min_chunk_chars: int = 80

    # upload limits
    max_pages: int = int(os.environ.get("MAX_PAGES", "1200"))
    max_file_mb: int = int(os.environ.get("MAX_FILE_MB", "80"))
    max_batch_files: int = 15

    # retrieval
    top_k_dense: int = int(os.environ.get("TOP_K_DENSE", "40"))
    top_k_bm25: int = int(os.environ.get("TOP_K_BM25", "40"))
    rerank_pool: int = int(os.environ.get("RERANK_POOL", "40"))
    rrf_k: int = 60
    final_k: int = int(os.environ.get("FINAL_K", "6"))
    final_k_max: int = 10
    min_dense_score: float = float(os.environ.get("MIN_DENSE", "0.70"))
    dedupe_jaccard: float = 0.85
    neighbour_chars: int = 400

    use_reranker: bool = os.environ.get("RERANKER", "1") == "1"
    use_multiquery: bool = os.environ.get("MULTIQUERY", "1") == "1"

    n_ctx: int = 4096
    max_tokens: int = 300
    temperature: float = 0.1
    n_threads: int = int(os.environ.get("N_THREADS", str(os.cpu_count() or 4)))

    entail_threshold: float = float(os.environ.get("ENTAIL_THRESHOLD", "0.15"))
    max_drop_ratio: float = 0.5


CFG = Config()