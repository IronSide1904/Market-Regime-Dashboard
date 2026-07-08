$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
