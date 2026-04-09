"""
PoliSim Stock API — Production Ready
Deploy to Railway, Render, or any VPS.

Local dev:
    uvicorn main:app --reload --port 8000

Deploy (Railway / Render):
    Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="PoliSim Stock API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
DATA_FILE = Path(os.getenv("DATA_PATH", "/tmp/stocks_data.json"))

_cache: dict = {}
_fetch_status: dict = {"running": False, "total": 0, "done": 0, "failed": []}


def _load_from_disk() -> dict:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_to_disk(data: dict):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _is_complete(sym_data: dict) -> bool:
    """A record is complete when it has price history, financials, and beta."""
    has_history = "priceHistory" in sym_data and len(sym_data["priceHistory"]) > 0
    has_fin     = sym_data.get("pb", 0.0) != 0.0
    has_beta    = sym_data.get("beta_calculated", False)
    return has_history and has_fin and has_beta


def _get_financials_radar(sym: str):
    """
    Try multiple vnstock sources then Yahoo Finance to extract P/E, P/B, EPS.
    Returns (pe, pb, eps) floats — all 0.0 on total failure.
    """
    pe_keywords  = ["p/e", "pe", "pricetoearning"]
    pb_keywords  = ["p/b", "pb", "pricetobook"]
    eps_keywords = ["eps", "earning"]

    for src in ["SSI", "VCI", "MSN", "DNSE"]:
        try:
            df = Vnstock().stock(symbol=sym, source=src).finance.ratio()
            if df is None or df.empty:
                continue
            row = {str(k).lower().replace(" ", ""): v for k, v in df.iloc[0].items()}

            pe, pb, eps = 0.0, 0.0, 0.0
            for k, v in row.items():
                if any(kw in k for kw in pe_keywords):
                    try: pe = round(float(v), 2); break
                    except: pass
            for k, v in row.items():
                if any(kw in k for kw in pb_keywords):
                    try: pb = round(float(v), 2); break
                    except: pass
            for k, v in row.items():
                if any(kw in k for kw in eps_keywords):
                    try: eps = round(float(v), 2); break
                    except: pass

            if pb != 0.0 or pe != 0.0:
                return pe, pb, eps
        except Exception:
            continue

    # Fallback: Yahoo Finance
    try:
        info = yf.Ticker(f"{sym}.VN").info
        pe  = round(float(info.get("trailingPE",  info.get("forwardPE",  0.0))), 2)
        pb  = round(float(info.get("priceToBook", 0.0)), 2)
        eps = round(float(info.get("trailingEps", info.get("forwardEps", 0.0))), 2)
        return pe, pb, eps
    except Exception:
        pass

    return 0.0, 0.0, 0.0

def _fetch_all_stocks(target: int = 500):
    global _cache, _fetch_status

    if not VNSTOCK_AVAILABLE:
        _fetch_status["running"] = False
        _fetch_status["failed"].append("vnstock not installed on server")
        return

    result = _load_from_disk()
    _cache = result.copy()
    _fetch_status.update({"running": True, "failed": []})

    try:
        # ----------------------------------------------------------------
        # Step 1 — Load VNINDEX for Beta calculation
        # ----------------------------------------------------------------
        vn_returns = pd.Series(dtype=float)
        try:
            vn_ticker   = Vnstock().stock(symbol="VNINDEX", source="VCI")
            vn_hist     = vn_ticker.quote.history(start="2023-01-01", end="2026-03-20", interval="1D")
            tc          = "time" if "time" in vn_hist.columns else vn_hist.columns[0]
            cc          = "close" if "close" in vn_hist.columns else next(c for c in vn_hist.columns if "close" in c.lower())
            vn_hist[tc] = pd.to_datetime(vn_hist[tc]).dt.tz_localize(None).dt.normalize()
            vn_hist     = vn_hist.set_index(tc).sort_index()
            vn_returns  = vn_hist[cc].astype(float).pct_change().dropna()
            vn_returns.name = "VNINDEX"
        except Exception as e:
            _fetch_status["failed"].append(f"VNINDEX load failed: {e}")

        # ----------------------------------------------------------------
        # Step 2 — Get full symbol listing (try multiple sources)
        # ----------------------------------------------------------------
        all_df = None
        for src in ["MSN", "TCBS", "VCI", "SSI"]:
            try:
                all_df = Vnstock().stock(symbol="SSI", source=src).listing.all_symbols()
                break
            except Exception:
                continue
        if all_df is None:
            raise RuntimeError("Cannot load symbol listing from any source")

        symbol_col  = "symbol" if "symbol" in all_df.columns else all_df.columns[0]
        all_symbols = all_df.dropna(subset=[symbol_col])

        completed_count        = sum(1 for d in result.values() if _is_complete(d))
        _fetch_status["total"] = target
        _fetch_status["done"]  = completed_count

        # ----------------------------------------------------------------
        # Step 3 — Iterate and enrich each symbol
        # ----------------------------------------------------------------
        for _, row in all_symbols.iterrows():
            if completed_count >= target:
                break

            sym = str(row.get("symbol", row.iloc[0])).strip().upper()

            # Skip already-complete records
            if sym in result and _is_complete(result[sym]):
                continue

            exchange = str(row.get("exchange", "HOSE")).upper()
            if exchange == "UPCOM":
                continue

            try:
                # A — Price history via Yahoo Finance
                yf_symbol = f"{sym}.HN" if exchange == "HNX" else f"{sym}.VN"
                hist = yf.Ticker(yf_symbol).history(period="1y")
                if hist.empty or len(hist) < 60:
                    _fetch_status["failed"].append(f"{sym}: insufficient price data")
                    continue

                close  = hist["Close"]
                volume = hist["Volume"]

                last_close    = round(float(close.iloc[-1]))
                price_history = close.tail(60).round(2).tolist()
                avg_vol       = max(1000, int(volume.tail(60).mean()))
                market_cap    = round(last_close * avg_vol * 10 / 1e9, 2)
                ema_60        = round(float(close.ewm(span=60).mean().iloc[-1]))

                # B — Beta via covariance with VNINDEX
                calculated_beta = 1.0
                if not vn_returns.empty:
                    stock_returns = close.pct_change().dropna()
                    stock_returns.index = stock_returns.index.tz_localize(None).normalize()
                    stock_returns.name  = "STOCK"
                    aligned = pd.concat([stock_returns, vn_returns], axis=1, join="inner").dropna()
                    if len(aligned) > 30:
                        cov = np.cov(aligned["STOCK"], aligned["VNINDEX"])
                        calculated_beta = round(cov[0, 1] / cov[1, 1], 2)

                # C — Financials via multi-source radar
                pe, pb, eps = _get_financials_radar(sym)

                fundamental_value = ema_60
                if pb > 0:
                    fundamental_value = round(last_close / pb)

                # D — Persist record
                if sym not in result:
                    result[sym] = {}

                result[sym].update({
                    "symbol":           sym,
                    "name":             str(row.get("organ_name", sym)),
                    "exchange":         exchange,
                    "sector":           str(row.get("icb_name2", "Unknown")),
                    "initialPrice":     last_close,
                    "fundamentalValue": fundamental_value,
                    "pe":               pe,
                    "pb":               pb,
                    "eps":              eps,
                    "priceHistory":     price_history,
                    "avgDailyVolume":   avg_vol,
                    "marketCap":        market_cap,
                    "priceLimit":       0.07,
                    "beta":             calculated_beta,
                    "beta_calculated":  True,
                })

                completed_count       += 1
                _fetch_status["done"]  = completed_count
                _cache = result.copy()
                _save_to_disk(result)

            except Exception as e:
                _fetch_status["failed"].append(f"{sym}: {e}")

            time.sleep(0.5)

    except Exception as e:
        _fetch_status["failed"].append(f"Pipeline error: {e}")
    finally:
        _fetch_status["running"] = False


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    global _cache
    _cache = _load_from_disk()
    print(f"[startup] {len(_cache)} stocks loaded from disk")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service":      "PoliSim Stock API",
        "stocks_ready": len(_cache),
        "fetch_status": _fetch_status,
        "endpoints": {
            "all_stocks":   "GET /api/stocks",
            "single_stock": "GET /api/stocks/{symbol}",
            "fetch_status": "GET /api/stocks/status",
            "clear_cache":  "DELETE /api/stocks",
        },
    }


@app.get("/api/stocks/status")
def fetch_status():
    """Live progress of the background fetch pipeline."""
    return _fetch_status


@app.get("/api/stocks")
def get_all_stocks(length: int = 500):
    """
    Return all cached stocks as a JSON object keyed by symbol.
    The JSON file on disk is the source of truth — fetched incrementally,
    so even a partial run returns whatever has been collected so far.
    Use ?length=N to set the target (default 500).
    """
    global _cache
    data = _cache or _load_from_disk()

    if not data:
        if not VNSTOCK_AVAILABLE:
            raise HTTPException(status_code=503, detail="vnstock not installed on server")
        _fetch_all_stocks(target=length)
        data = _cache

    if not data:
        raise HTTPException(status_code=503, detail="Failed to fetch stock data")

    return data


@app.get("/api/stocks/{symbol}")
def get_stock(symbol: str):
    """Return a single stock by ticker symbol, e.g. /api/stocks/HPG."""
    global _cache
    data = _cache or _load_from_disk()

    if not data:
        if not VNSTOCK_AVAILABLE:
            raise HTTPException(status_code=503, detail="vnstock not installed on server")
        _fetch_all_stocks(target=250)
        data = _cache

    sym = symbol.strip().upper()
    if sym not in data:
        raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found")
    return data[sym]


@app.delete("/api/stocks")
def clear_stocks():
    """Wipe in-memory cache and JSON file so the next GET re-fetches from scratch."""
    global _cache
    _cache = {}
    if DATA_FILE.exists():
        DATA_FILE.unlink()
    return {"message": "Cache cleared"}