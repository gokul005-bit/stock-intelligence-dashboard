# Stock Data Intelligence Dashboard (JarNox Internship Assignment)

Mini financial data platform to collect, clean, store, and visualize stock data with clean REST APIs and a simple dashboard UI.

## Highlights (what this demonstrates)
- **Real-world data handling**: ingestion, cleaning, typing, missing values, deduplication
- **Metrics**: Daily Return, 7-day Moving Average, 52-week High/Low
- **Custom insight**: **Volatility Score (30d)** = rolling standard deviation of daily returns (risk signal)
- **APIs**: `/companies`, `/data/{symbol}`, `/summary/{symbol}`, bonus `/compare`
- **Dashboard**: left-side company list, click-to-chart, optional compare mode

## Tech Stack
- Python, FastAPI
- SQLite (local, zero-config)
- Pandas + NumPy (cleaning + metrics)
- yfinance (public market data)
- Chart.js (frontend charts)

## Quickstart (Windows / PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```
Open: `http://127.0.0.1:8000`

FastAPI docs (Swagger): `http://127.0.0.1:8000/docs`

## Data Notes
- App seeds a few symbols on startup (best-effort). If a symbol isn’t present, the API **ingests on-demand** the first time you request it.
- For NSE symbols, use the **`.NS`** suffix (example: `INFY.NS`, `TCS.NS`).

## Endpoints
### GET `/companies`
Returns all available companies in the local database.

### GET `/data/{symbol}?days=30`
Returns last N days of cleaned data including computed fields:
- `daily_return = (close - open) / open`
- `ma7` (7-day moving average of close)

### GET `/summary/{symbol}`
Returns:
- 52-week high, low (computed over last ~252 trading sessions available)
- average close
- **volatility_score_30d** (custom metric)

### GET `/compare?symbol1=INFY.NS&symbol2=TCS.NS&days=90` (Bonus)
Returns:
- aligned date series
- normalized performance curves (start at 1.0)
- correlation of daily returns (when sufficient data)

## Project Structure
```
.
├─ main.py
├─ requirements.txt
├─ README.md
└─ static/
   └─ index.html
```

## Submission Checklist (recommended)
- Screenshots:
  - Swagger docs page
  - `/data/{symbol}` response sample
  - dashboard showing chart + cards
  - compare chart + correlation
- Optional: short 2–3 minute demo video walking through the UI + endpoints.

