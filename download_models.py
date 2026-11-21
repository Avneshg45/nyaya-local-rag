"""Download the model artifacts used by Nyaya Local.

Usage:
    python download_models.py

Set HF_ENDPOINT or HF_TOKEN as usual for the Hugging Face client.
The GGUF file is left optional because it is large and has several quantizations.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "legalrag_data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "intfloat/multilingual-e5-small"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def main() -> None:
    for repo_id in (EMBED_MODEL, NLI_MODEL, RERANK_MODEL):
        print(f"Downloading {repo_id}...")
        snapshot_download(repo_id=repo_id)

    gguf = MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
    if not gguf.exists():
        print("Downloading GGUF model...")
        hf_hub_download(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            local_dir=MODELS_DIR,
        )
    else:
        print(f"GGUF already present: {gguf}")


if __name__ == "__main__":
    main()