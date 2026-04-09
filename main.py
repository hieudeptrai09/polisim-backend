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

# CORS — allow all origins so any deployed frontend can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Storage — prefer /tmp on serverless, fallback to local file
# ---------------------------------------------------------------------------
DATA_FILE = Path(os.getenv("DATA_PATH", "/tmp/stocks_data.json"))

_cache: dict = {}
_fetch_status: dict = {
    "running": False,
    "total": 0,
    "done": 0,
    "failed": [],
}


def _load_from_disk() -> dict:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_to_disk(data: dict):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Background fetch task
# ---------------------------------------------------------------------------
def _fetch_all_stocks(target: int = 250):
    global _cache, _fetch_status

    if not VNSTOCK_AVAILABLE:
        _fetch_status["running"] = False
        _fetch_status["failed"].append("vnstock not installed on server")
        return

    result = _load_from_disk()
    _cache = result.copy()
    _fetch_status.update({"running": True, "failed": []})

    try:
        # New API: use Vnstock().stock() to get a listing of all symbols
        client = Vnstock()
        listing = client.stock(symbol="VNM", source="VCI")  # symbol required but ignored for listing
        all_df = listing.listing.all_symbols(show_log=False)
        symbol_col = "symbol" if "symbol" in all_df.columns else all_df.columns[0]
        all_symbols = all_df.dropna(subset=[symbol_col])

        existing = set(result.keys())
        remaining = all_symbols[~all_symbols[symbol_col].isin(existing)]
        needed = max(0, target - len(existing))
        stocks_df = remaining.head(needed)

        _fetch_status["total"] = len(existing) + len(stocks_df)
        _fetch_status["done"] = len(existing)

        for _, row in stocks_df.iterrows():
            sym = str(row.get("symbol", row.iloc[0])).strip().upper()
            try:
                # New API: initialise per-symbol ticker object, then call quote.history
                ticker = Vnstock().stock(symbol=sym, source="VCI")
                hist = ticker.quote.history(
                    start="2023-01-01", end="2026-03-20",
                    interval="1D",
                )
                if hist is None or len(hist) < 10:
                    raise ValueError("Không đủ dữ liệu lịch sử")

                time_col  = "time"   if "time"   in hist.columns else hist.columns[0]
                close_col = "close"  if "close"  in hist.columns else next(c for c in hist.columns if "close"  in c.lower())
                vol_col   = "volume" if "volume" in hist.columns else next(c for c in hist.columns if "vol"    in c.lower())

                hist      = hist.set_index(time_col).sort_index()
                close     = hist[close_col].astype(float)
                volume    = hist[vol_col].astype(float)

                last_raw   = float(close.iloc[-1])
                mult       = 1000 if last_raw < 1000 else 1
                last_close = round(last_raw * mult)
                fundamental = round(float(close.ewm(span=60).mean().iloc[-1]) * mult)
                avg_vol    = max(1000, int(volume.tail(60).mean()))
                market_cap = round(last_close * avg_vol * 10 / 1e9, 2)
                shares_outstanding = int(market_cap * 1e9 / last_close) if last_close > 0 else 0

                # Optional: try to fetch a real fundamental value via finance.ratio
                try:
                    ratios = ticker.finance.ratio()
                    if ratios is not None and not ratios.empty:
                        # Use book value per share (bvps) as fundamental proxy if available
                        bvps_col = next(
                            (c for c in ratios.columns if "bvps" in c.lower() or "book" in c.lower()),
                            None,
                        )
                        if bvps_col:
                            bvps = float(ratios[bvps_col].dropna().iloc[-1])
                            if bvps > 0:
                                fundamental = round(bvps * mult)
                except Exception:
                    pass  # fall back to EMA-based fundamental

                result[sym] = {
                    "symbol":            sym,
                    "name":              str(row.get("organ_name", sym)),
                    "exchange":          str(row.get("exchange", "HOSE")),
                    "sector":            str(row.get("icb_name2", "Unknown")),
                    "initialPrice":      last_close,
                    "fundamentalValue":  fundamental,
                    "avgDailyVolume":    avg_vol,
                    "marketCap":         market_cap,
                    "priceLimit":        0.07,
                    "beta":              1.0,
                    "beta_done":         False,
                    "sharesOutstanding": shares_outstanding,
                    "mcap_real":         True,
                }
                _fetch_status["done"] += 1

            except Exception as e:
                _fetch_status["failed"].append(f"{sym}: {e}")

            _cache = result.copy()
            _save_to_disk(result)
            time.sleep(0.5)          # be polite to the data source

    except Exception as e:
        _fetch_status["failed"].append(f"Listing error: {e}")
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
        "service": "PoliSim Stock API",
        "stocks_ready": len(_cache),
        "endpoints": {
            "all_stocks":   "GET /api/stocks",
            "single_stock": "GET /api/stocks/{symbol}",
            "clear_cache":  "DELETE /api/stocks",
        },
    }


@app.get("/api/stocks")
def get_all_stocks(length: int = 250):
    """
    Return stocks as a JSON object keyed by symbol.
    Auto-fetches from vnstock on first call if cache is empty.
    Use ?length=50 to fetch fewer stocks and avoid timeout on slow servers.
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
    """Return a single stock by ticker symbol (e.g. /api/stocks/HPG)."""
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
    """Wipe cache and JSON file so the next GET will re-fetch from scratch."""
    global _cache
    _cache = {}
    if DATA_FILE.exists():
        DATA_FILE.unlink()
    return {"message": "Cache cleared"}