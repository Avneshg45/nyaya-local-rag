#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d ".venv" ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
export LEGALRAG_HOME="$PWD/legalrag_data"
export GGUF_PATH="$LEGALRAG_HOME/models/qwen2.5-3b-instruct-q4_k_m.gguf"
mkdir -p "$LEGALRAG_HOME/models"
[ -f "$GGUF_PATH" ] || .venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='$LEGALRAG_HOME/models')"
.venv/bin/streamlit run legal_rag/app.py --server.port 8510
