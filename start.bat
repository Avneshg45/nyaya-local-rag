@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
set "LEGALRAG_HOME=%CD%\legalrag_data"
set "MODEL_DIR=%LEGALRAG_HOME%\models"
set "GGUF_PATH=%MODEL_DIR%\qwen2.5-3b-instruct-q4_k_m.gguf"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%GGUF_PATH%" ".venv\Scripts\python.exe" -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir=r'%MODEL_DIR%')"
".venv\Scripts\python.exe" -m streamlit run legal_rag/app.py --server.port 8510
pause
