# MR-1 Lite

Searchable retail market regime and swing-trading dashboard built with Streamlit.
Position regime analysis, volume, correlation, and performance comparison.

## Requirements

- Python 3.10 or newer
- Internet access for `yfinance` market data

## Install

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Then open:

```text
http://SERVER_IP:8501
```

For local use:

```text
http://127.0.0.1:8501
```

## Optional Finviz Elite Data

Finviz is optional. The dashboard continues to work with yfinance if Finviz is not configured.

Store a rotated Finviz Elite token locally in `.env`:

```env
FINVIZ_AUTH_TOKEN=PASTE_NEW_FINVIZ_TOKEN_HERE
```

For Streamlit deployment, use `.streamlit/secrets.toml`:

```toml
FINVIZ_AUTH_TOKEN = "PASTE_NEW_FINVIZ_TOKEN_HERE"
```

Do not commit either file.

## Files

- `app.py` - Streamlit entry point
- `dashboard.py` - UI and chart rendering
- `data.py` - yfinance data loading
- `indicators.py` - regime indicators
- `scoring.py` - MR-1 regime scoring
- `swing.py` - swing-trading engine
- `performance.py` - multi-window performance calculations
- `metadata.py` - editable sector, industry, and peer mappings
- `peers.py` - peer helpers
- `config.py` - app configuration, score weights, and defaults

## Notes

The dashboard downloads market data at runtime through `yfinance`. If a server has no internet access, the app will load but market data will be unavailable.
