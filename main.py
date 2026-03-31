from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "stocks.db"
STATIC_DIR = APP_DIR / "static"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS companies (
              symbol TEXT PRIMARY KEY,
              name TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
              symbol TEXT NOT NULL,
              date TEXT NOT NULL,
              open REAL,
              high REAL,
              low REAL,
              close REAL,
              volume INTEGER,
              daily_return REAL,
              ma7 REAL,
              PRIMARY KEY (symbol, date)
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)")


def _to_iso_date(d: Any) -> str:
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, (datetime, date)):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _clean_prices(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.reset_index(inplace=True)

    # yfinance returns Date column as DatetimeIndex or column name varies
    if "Date" not in df.columns:
        # fallback: first column might be the date
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Date": "date",
    }
    df.rename(columns=rename_map, inplace=True)

    # Keep only required columns (adj_close not needed for metrics here)
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    df["date"] = df["date"].apply(_to_iso_date)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

    df = df.dropna(subset=["date", "open", "close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Required metrics
    df["daily_return"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["daily_return"] = df["daily_return"].replace([np.inf, -np.inf], np.nan)
    df["ma7"] = df["close"].rolling(7, min_periods=1).mean()

    df.insert(0, "symbol", symbol.upper())
    return df[
        ["symbol", "date", "open", "high", "low", "close", "volume", "daily_return", "ma7"]
    ]


@dataclass(frozen=True)
class IngestResult:
    symbol: str
    rows_upserted: int
    first_date: Optional[str]
    last_date: Optional[str]


def ingest_symbol(symbol: str, name: Optional[str] = None, period: str = "2y") -> IngestResult:
    symbol = symbol.upper().strip()
    ticker = yf.Ticker(symbol)

    hist = ticker.history(period=period, auto_adjust=False)
    cleaned = _clean_prices(hist, symbol=symbol)

    if cleaned.empty:
        raise ValueError(f"No data returned for {symbol}.")

    init_db()
    with _connect() as con:
        con.execute(
            "INSERT OR IGNORE INTO companies(symbol, name) VALUES (?, ?)",
            (symbol, name or symbol),
        )
        rows = cleaned.to_dict(orient="records")
        con.executemany(
            """
            INSERT INTO prices(symbol, date, open, high, low, close, volume, daily_return, ma7)
            VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :daily_return, :ma7)
            ON CONFLICT(symbol, date) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              volume=excluded.volume,
              daily_return=excluded.daily_return,
              ma7=excluded.ma7
            """,
            rows,
        )

    return IngestResult(
        symbol=symbol,
        rows_upserted=len(cleaned),
        first_date=str(cleaned["date"].iloc[0]),
        last_date=str(cleaned["date"].iloc[-1]),
    )


def ensure_seed_data() -> None:
    init_db()
    with _connect() as con:
        n = con.execute("SELECT COUNT(*) AS n FROM companies").fetchone()["n"]
    if n:
        return

    # Seed a few popular symbols (works internationally; user can change later)
    for sym, nm in [
        ("INFY.NS", "Infosys"),
        ("TCS.NS", "Tata Consultancy Services"),
        ("RELIANCE.NS", "Reliance Industries"),
        ("HDFCBANK.NS", "HDFC Bank"),
    ]:
        try:
            ingest_symbol(sym, name=nm, period="2y")
        except Exception:
            # If a symbol fails (network / region), ignore and continue
            continue

    # If internet/data source is blocked, generate a small deterministic mock dataset
    with _connect() as con:
        n2 = con.execute("SELECT COUNT(*) AS n FROM companies").fetchone()["n"]
    if not n2:
        _seed_mock_data()


def _seed_mock_data() -> None:
    rng = np.random.default_rng(7)
    init_db()

    symbols = [
        ("INFY.NS", "Infosys (Mock)"),
        ("TCS.NS", "TCS (Mock)"),
        ("RELIANCE.NS", "Reliance (Mock)"),
        ("HDFCBANK.NS", "HDFC Bank (Mock)"),
    ]

    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=420)  # ~1.6y of sessions

    all_rows: list[dict[str, Any]] = []
    for sym, _nm in symbols:
        base = float(rng.uniform(900, 2200))
        drift = float(rng.uniform(-0.0002, 0.0006))
        vol = float(rng.uniform(0.008, 0.02))

        rets = rng.normal(loc=drift, scale=vol, size=len(dates))
        close = base * np.cumprod(1 + rets)
        openp = close * (1 + rng.normal(0, vol / 3, size=len(dates)))
        high = np.maximum(openp, close) * (1 + rng.uniform(0.0005, 0.01, size=len(dates)))
        low = np.minimum(openp, close) * (1 - rng.uniform(0.0005, 0.01, size=len(dates)))
        volume = rng.integers(5_000_00, 25_000_000, size=len(dates))

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": openp,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
        cleaned = _clean_prices(df, symbol=sym)
        all_rows.extend(cleaned.to_dict(orient="records"))

        with _connect() as con:
            con.execute("INSERT OR IGNORE INTO companies(symbol, name) VALUES (?, ?)", (sym, _nm))

    with _connect() as con:
        con.executemany(
            """
            INSERT INTO prices(symbol, date, open, high, low, close, volume, daily_return, ma7)
            VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :daily_return, :ma7)
            ON CONFLICT(symbol, date) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              volume=excluded.volume,
              daily_return=excluded.daily_return,
              ma7=excluded.ma7
            """,
            all_rows,
        )


def _fetch_last_n(symbol: str, n_days: int) -> list[dict[str, Any]]:
    symbol = symbol.upper().strip()
    init_db()
    with _connect() as con:
        rows = con.execute(
            """
            SELECT symbol, date, open, high, low, close, volume, daily_return, ma7
            FROM prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (symbol, int(n_days)),
        ).fetchall()
    out = [dict(r) for r in reversed(rows)]
    return out


def _summary(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    init_db()

    # 52-week ≈ 252 trading sessions; if fewer, use what we have
    with _connect() as con:
        rows = con.execute(
            """
            SELECT date, close, daily_return
            FROM prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 252
            """,
            (symbol,),
        ).fetchall()

    if not rows:
        raise KeyError(symbol)

    closes = np.array([r["close"] for r in rows if r["close"] is not None], dtype=float)
    rets = np.array(
        [r["daily_return"] for r in rows if r["daily_return"] is not None and not math.isnan(r["daily_return"])],
        dtype=float,
    )

    high_52w = float(np.max(closes)) if closes.size else None
    low_52w = float(np.min(closes)) if closes.size else None
    avg_close = float(np.mean(closes)) if closes.size else None

    # Custom metric: volatility score (rolling std dev of returns over up to last 30 sessions)
    last30 = rets[:30] if rets.size else np.array([], dtype=float)
    vol_score = float(np.std(last30, ddof=1)) if last30.size >= 2 else (float(np.std(last30)) if last30.size else None)

    return {
        "symbol": symbol,
        "window_days_used": int(len(rows)),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "avg_close": avg_close,
        "volatility_score_30d": vol_score,
    }


def _insights(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    data = _fetch_last_n(symbol, n_days=90)
    if len(data) < 10:
        raise KeyError(symbol)

    df = pd.DataFrame(data)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ma7"] = pd.to_numeric(df["ma7"], errors="coerce")
    df["daily_return"] = pd.to_numeric(df["daily_return"], errors="coerce")
    df = df.dropna(subset=["close"])

    last_close = float(df["close"].iloc[-1])
    first_close = float(df["close"].iloc[0])
    perf = (last_close / first_close - 1.0) if first_close else 0.0

    ma_short = float(df["close"].rolling(7, min_periods=1).mean().iloc[-1])
    ma_long = float(df["close"].rolling(30, min_periods=5).mean().iloc[-1])

    rets = df["daily_return"].dropna().tail(30).to_numpy(dtype=float)
    vol = float(np.std(rets, ddof=1)) if rets.size >= 2 else (float(np.std(rets)) if rets.size else 0.0)

    labels: list[str] = []
    if ma_short > ma_long and perf > 0:
        labels.append("Uptrend detected")
    elif ma_short < ma_long and perf < 0:
        labels.append("Downtrend detected")
    else:
        labels.append("Sideways / mixed trend")

    if vol >= 0.02:
        labels.append("High volatility")
    elif vol <= 0.01:
        labels.append("Low volatility")
    else:
        labels.append("Moderate volatility")

    if perf >= 0.08:
        labels.append("Strong performance (90d)")
    elif perf <= -0.08:
        labels.append("Weak performance (90d)")
    else:
        labels.append("Stable performance (90d)")

    return {
        "symbol": symbol,
        "performance_90d": float(perf),
        "ma7": float(ma_short),
        "ma30": float(ma_long),
        "volatility_score_30d": float(vol),
        "signals": labels,
    }


def _predict_next_close(symbol: str, lookback_days: int = 60) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    rows = _fetch_last_n(symbol, n_days=max(lookback_days, 20))
    if len(rows) < 20:
        raise KeyError(symbol)

    df = pd.DataFrame(rows)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.shape[0] < 20:
        raise KeyError(symbol)

    y = df["close"].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    # linear regression (least squares)
    a, b = np.polyfit(x, y, 1)  # y = a*x + b
    next_pred = float(a * (len(y)) + b)
    last = float(y[-1])

    return {
        "symbol": symbol,
        "lookback_days_used": int(len(y)),
        "model": "linear_regression_close_vs_time",
        "last_close": last,
        "predicted_next_close": next_pred,
        "predicted_change_pct": float((next_pred / last - 1.0) if last else 0.0),
    }


def _movers() -> dict[str, Any]:
    init_db()
    with _connect() as con:
        latest = con.execute("SELECT MAX(date) AS d FROM prices").fetchone()["d"]
        if not latest:
            return {"date": None, "top_gainers": [], "top_losers": []}
        rows = con.execute(
            """
            SELECT p.symbol, c.name, p.date, p.daily_return
            FROM prices p
            LEFT JOIN companies c ON c.symbol = p.symbol
            WHERE p.date = ? AND p.daily_return IS NOT NULL
            """,
            (latest,),
        ).fetchall()
    items = [dict(r) for r in rows]
    items.sort(key=lambda r: (r["daily_return"] if r["daily_return"] is not None else -999), reverse=True)
    top_gainers = items[:3]
    top_losers = list(reversed(items[-3:])) if len(items) >= 3 else list(reversed(items))
    return {"date": latest, "top_gainers": top_gainers, "top_losers": top_losers}


app = FastAPI(title="Stock Data Intelligence Dashboard", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    ensure_seed_data()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="static/index.html missing")
    return FileResponse(str(index))


@app.get("/companies")
def companies() -> list[dict[str, Any]]:
    init_db()
    with _connect() as con:
        rows = con.execute("SELECT symbol, name FROM companies ORDER BY symbol").fetchall()
    return [dict(r) for r in rows]


@app.get("/data/{symbol}")
def data(symbol: str, days: int = Query(30, ge=1, le=365)) -> dict[str, Any]:
    init_db()
    rows = _fetch_last_n(symbol, n_days=days)
    if not rows:
        # try ingest on-demand
        try:
            ingest_symbol(symbol, name=symbol, period="2y")
            rows = _fetch_last_n(symbol, n_days=days)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Symbol not available: {symbol}. ({e})")
    return {"symbol": symbol.upper().strip(), "days": int(days), "rows": rows}


@app.get("/summary/{symbol}")
def summary(symbol: str) -> dict[str, Any]:
    try:
        return _summary(symbol)
    except KeyError:
        # try ingest on-demand
        try:
            ingest_symbol(symbol, name=symbol, period="2y")
            return _summary(symbol)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Symbol not available: {symbol}. ({e})")


@app.get("/insights/{symbol}")
def insights(symbol: str) -> dict[str, Any]:
    try:
        return _insights(symbol)
    except KeyError:
        try:
            ingest_symbol(symbol, name=symbol, period="2y")
            return _insights(symbol)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Symbol not available: {symbol}. ({e})")


@app.get("/predict/{symbol}")
def predict(symbol: str, lookback_days: int = Query(60, ge=20, le=252)) -> dict[str, Any]:
    try:
        return _predict_next_close(symbol, lookback_days=int(lookback_days))
    except KeyError:
        try:
            ingest_symbol(symbol, name=symbol, period="2y")
            return _predict_next_close(symbol, lookback_days=int(lookback_days))
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Symbol not available: {symbol}. ({e})")


@app.get("/movers")
def movers() -> dict[str, Any]:
    return _movers()


@app.get("/compare")
def compare(
    symbol1: str = Query(..., min_length=1),
    symbol2: str = Query(..., min_length=1),
    days: int = Query(90, ge=7, le=365),
) -> dict[str, Any]:
    s1 = symbol1.upper().strip()
    s2 = symbol2.upper().strip()

    d1 = _fetch_last_n(s1, n_days=days)
    d2 = _fetch_last_n(s2, n_days=days)
    if not d1:
        ingest_symbol(s1, name=s1, period="2y")
        d1 = _fetch_last_n(s1, n_days=days)
    if not d2:
        ingest_symbol(s2, name=s2, period="2y")
        d2 = _fetch_last_n(s2, n_days=days)

    if not d1 or not d2:
        raise HTTPException(status_code=404, detail="One or both symbols not available")

    # Align by date
    df1 = pd.DataFrame(d1)[["date", "close", "daily_return"]].rename(
        columns={"close": "close1", "daily_return": "ret1"}
    )
    df2 = pd.DataFrame(d2)[["date", "close", "daily_return"]].rename(
        columns={"close": "close2", "daily_return": "ret2"}
    )
    merged = pd.merge(df1, df2, on="date", how="inner").sort_values("date")

    # Normalize for charting (start at 1.0)
    merged["norm1"] = merged["close1"] / merged["close1"].iloc[0]
    merged["norm2"] = merged["close2"] / merged["close2"].iloc[0]

    corr = None
    if merged[["ret1", "ret2"]].dropna().shape[0] >= 3:
        corr = float(merged[["ret1", "ret2"]].corr().iloc[0, 1])

    return {
        "symbols": [s1, s2],
        "days_requested": int(days),
        "days_aligned": int(len(merged)),
        "correlation_daily_returns": corr,
        "series": merged[["date", "close1", "close2", "norm1", "norm2"]].to_dict(orient="records"),
        "summary": {"symbol1": _summary(s1), "symbol2": _summary(s2)},
    }

