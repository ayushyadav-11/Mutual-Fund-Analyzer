"""
app.py — FastAPI backend for Mutual Fund Portfolio Analyzer
Run: uvicorn app:app --reload --port 8000
"""

import os
import re
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, Any
from datetime import date

import asyncio
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.parser import parse_cas, save_session, load_session, recompute_xirr
from core.risk import compute_risk_metrics
from data.database import initialize_database
from data.data_collector import fetch_and_populate_mfapi_data, prefetch_deep_dive_for_user
from core.portfolio_overlap import fetch_fund_holdings, compute_overlap, _scraper
from core.advanced_analytics import calculate_taxes_and_loads, calculate_goal_strategy, calculate_sip_step_up, run_monte_carlo_simulation, calculate_stress_test, calculate_rebalance, calculate_dividend_cashflow

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from pydantic import BaseModel

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

app = FastAPI(
    title="Mutual Fund Portfolio Analyzer",
    description="Parse CAS PDF → Analyze Portfolio with risk metrics",
    version="1.0.0",
)
logger = logging.getLogger(__name__)

DEBUG_BYPASS_DEEP_DIVE_CACHE = False
DEBUG_DISABLE_FUNDAMENTALS_FALLBACK = False

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.on_event("startup")
def startup_event():
    import threading
    def _init():
        logger.info("Initializing database schemas in background...")
        try:
            initialize_database()
            logger.info("Database schemas verified successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    t = threading.Thread(target=_init, daemon=True)
    t.start()

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# ── JWKS cache for ECC (P-256) / ES256 token verification ────────────────────
_jwks_cache: dict = {}

async def _get_jwks() -> dict:
    """Fetch and cache Supabase JWKS public keys (used for ES256 / ECC P-256 tokens)."""
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    if not SUPABASE_URL:
        return {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
            resp.raise_for_status()
            _jwks_cache = resp.json()
            logger.info("JWKS fetched successfully: %d key(s)", len(_jwks_cache.get("keys", [])))
    except Exception as exc:
        logger.warning("Failed to fetch JWKS: %s", exc)
    return _jwks_cache


def _decode_token(token: str, jwks: dict) -> dict:
    """Try JWKS public keys first (RS256/ES256), then fall back to legacy HS256 secret."""
    import jwt
    from jwt import PyJWK

    # 1. Try each public key from JWKS (RS256 or ES256)
    for key_data in jwks.get("keys", []):
        try:
            jwk = PyJWK.from_dict(key_data)
            public_key = jwk.key
            return jwt.decode(
                token, public_key,
                algorithms=["RS256", "ES256"],
                options={"verify_aud": False}
            )
        except Exception:
            continue

    # 2. Fallback: legacy HS256 shared secret (for old tokens still in circulation)
    if SUPABASE_JWT_SECRET:
        return jwt.decode(
            token, SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False}
        )

    raise ValueError("No valid signing key found")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Unauthenticated endpoints
        if request.method == "OPTIONS" or request.url.path in [
            "/", "/favicon.ico", "/dashboard.html", "/family.html",
            "/reset-password.html", "/api/config"
        ]:
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized access"},
                    headers={"Access-Control-Allow-Origin": "*"}
                )
            token = auth_header.split(" ")[1]
            try:
                jwks = await _get_jwks()
                payload = _decode_token(token, jwks)
                request.state.user_id = payload.get("sub")
            except Exception as e:
                logger.warning("JWT verification failed: %s", e)
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid token"},
                    headers={"Access-Control-Allow-Origin": "*"}
                )
        return await call_next(request)

app.add_middleware(AuthMiddleware)

@app.get("/api/config")
def get_config():
    return {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY")
    }


# Session file location (persists between requests)
SESSION_FILE = os.path.join(os.path.dirname(__file__), "session_data.json")

# In-memory cache for Yahoo Finance benchmark series
_benchmark_cache = {}


# ── MFAPI Live NAV Fetcher ───────────────────────────────────────────────────
async def fetch_latest_navs_from_mfapi(holdings: list) -> None:
    """
    Mutate holdings in-place to attach `live_nav` and `live_nav_date`
    by fetching from mfapi.in async.
    """
    if not holdings:
        return

    # Fetch master scheme list
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            res = await client.get("https://api.mfapi.in/mf")
            res.raise_for_status()
            master_list = res.json()
        except Exception:
            return  # Silently fail, fallback to last transaction NAV

        # Build ISIN -> SchemeCode map
        isin_to_code = {}
        for fund in master_list:
            code = fund.get("schemeCode")
            isin1 = fund.get("isinGrowth")
            isin2 = fund.get("isinDivReinvestment")
            if code:
                if isin1 and isin1 not in isin_to_code:
                    isin_to_code[isin1] = code
                if isin2 and isin2 not in isin_to_code:
                    isin_to_code[isin2] = code

        # Fetch NAVs for each holding concurrently
        async def fetch_nav(holding: dict):
            isin = holding.get("isin")
            code = isin_to_code.get(isin)
            if not code:
                return
            try:
                r = await client.get(f"https://api.mfapi.in/mf/{code}")
                r.raise_for_status()
                data = r.json().get("data", [])
                if data:
                    holding["live_nav"] = float(data[0]["nav"])
                    holding["live_nav_date"] = data[0]["date"]
                    
                    from datetime import datetime
                    month_map = {}
                    for entry in data:
                        try:
                            dt = datetime.strptime(entry["date"], "%d-%m-%Y")
                            ym = dt.strftime("%Y-%m")
                            if ym not in month_map or dt > month_map[ym]["dt"]:
                                month_map[ym] = {"dt": dt, "nav": float(entry["nav"])}
                        except ValueError:
                            pass
                    
                    hist = [{"date": month_map[ym]["dt"].strftime("%Y-%m-%d"), "nav": month_map[ym]["nav"]} 
                            for ym in sorted(month_map.keys())]
                    holding["historical_nav_series"] = hist[-60:]
            except Exception:
                pass

        await asyncio.gather(*(fetch_nav(h) for h in holdings))


# ── Mount frontend static files ───────────────────────────────────────────────
frontend_dir = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
def root():
    response = FileResponse(str(frontend_dir / "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = "0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    response = FileResponse(str(frontend_dir / "dashboard.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = "0"
    response.headers["Pragma"] = "no-cache"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/parse
# Upload CAS PDF + optional password → parse and store session
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/parse")
async def parse_pdf(
    request: Request,
    file: UploadFile = File(...),
    password: str = Form(default=""),
    merge: bool = Query(False, description="Merge with existing session instead of overwriting")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        data = parse_cas(tmp_path, password=password)
        # Fetch live NAVs before saving
        await fetch_latest_navs_from_mfapi(data.get("holdings", []))
        # Important: Recompute XIRR now that live NAVs are populated
        data = recompute_xirr(data)
        save_session(data, request.state.user_id, merge=merge)
        
        # If merged, reload to return the combined dataset
        if merge:
            data = load_session(request.state.user_id)
            
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass

    # Fire-and-forget: pre-fetch deep-dive data for all holdings in the background
    # so the first Risk tab click is instant instead of waiting 10-20s per fund
    user_id = request.state.user_id
    asyncio.create_task(prefetch_deep_dive_for_user(user_id, data.get("holdings", [])))

    return {
        "status": "ok",
        "investor": data["investor_info"]["name"],
        "period": data["statement_period"],
        "funds": len(data["holdings"]),
        "transactions": len(data.get("all_transactions", data.get("transactions", []))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/summary
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/summary")
def get_summary(request: Request):
    data = _load_or_404(request.state.user_id)
    holdings = data["holdings"]

    total_invested = 0.0
    current_value = 0.0

    for h in holdings:
        # Sum of all BUY / SIP amounts = invested
        invested = sum(
            abs(t["amount"])
            for t in h["transactions"]
            if t["type"] in ("BUY", "SIP", "SWITCH_IN", "DIVR")
        )
        # Subtract redemptions
        redeemed = sum(
            abs(t["amount"])
            for t in h["transactions"]
            if t["type"] in ("SELL", "SWITCH_OUT")
        )
        if h["units"] <= 0.001:
            net_invested = 0.0
        else:
            net_invested = invested - redeemed
            
        total_invested += net_invested

        # Current value = units × live NAV (fallback to last known)
        curr_nav = h.get("live_nav") or _get_last_nav(h["transactions"])
        curr_val = h["units"] * curr_nav if curr_nav else 0.0
        h["current_value"] = round(curr_val, 2)
        h["invested"] = round(net_invested, 2)
        current_value += curr_val

    abs_return_pct = (
        ((current_value - total_invested) / total_invested * 100)
        if total_invested > 0
        else 0.0
    )

    return {
        "investor": data["investor_info"],
        "period": data["statement_period"],
        "total_invested": round(total_invested, 2),
        "current_value": round(current_value, 2),
        "total_gain": round(current_value - total_invested, 2),
        "abs_return_pct": round(abs_return_pct, 2),
        "xirr": data.get("portfolio_xirr"),
        "fund_count": len(holdings),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/holdings
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/holdings")
def get_holdings(request: Request):
    data = _load_or_404(request.state.user_id)
    result = []
    for h in data["holdings"]:
        curr_nav = h.get("live_nav") or _get_last_nav(h["transactions"])
        
        if h["units"] <= 0.001:
            invested = 0.0
            current_value = 0.0
            gain = 0.0
            gain_pct = 0.0
        else:
            invested = sum(
                abs(t["amount"])
                for t in h["transactions"]
                if t["type"] in ("BUY", "SIP", "SWITCH_IN", "DIVR")
            ) - sum(
                abs(t["amount"])
                for t in h["transactions"]
                if t["type"] in ("SELL", "SWITCH_OUT")
            )
            current_value = h["units"] * curr_nav if curr_nav else 0.0
            gain = current_value - invested
            gain_pct = (gain / invested * 100) if invested > 0 else 0.0

        # Avg buy NAV
        buy_txns = [t for t in h["transactions"] if t["type"] in ("BUY", "SIP") and t["nav"]]
        avg_nav = (
            sum(t["nav"] for t in buy_txns) / len(buy_txns) if buy_txns else None
        )

        result.append({
            "isin": h["isin"],
            "name": h["name"],
            "category": h["category"],
            "units": h["units"],
            "avg_nav": round(avg_nav, 4) if avg_nav else None,
            "current_nav": curr_nav,
            "live_nav_date": h.get("live_nav_date"),
            "invested": round(invested, 2),
            "current_value": round(current_value, 2),
            "gain": round(gain, 2),
            "gain_pct": round(gain_pct, 2),
            "xirr": h.get("xirr") if h["units"] > 0.001 else None,
            "benchmark": h.get("benchmark"),
        })

    result.sort(key=lambda x: x["current_value"], reverse=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/allocation
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/allocation")
def get_allocation(request: Request):
    data = _load_or_404(request.state.user_id)
    category_map = {}
    for h in data["holdings"]:
        cat = _normalize_category(h.get("category", "Other"))
        curr_nav = h.get("live_nav") or _get_last_nav(h["transactions"])
        val = h["units"] * curr_nav if curr_nav else 0.0
        category_map[cat] = category_map.get(cat, 0) + val

    total = sum(category_map.values())
    allocation = [
        {
            "category": cat,
            "value": round(val, 2),
            "pct": round(val / total * 100, 2) if total > 0 else 0,
        }
        for cat, val in sorted(category_map.items(), key=lambda x: x[1], reverse=True)
    ]
    return allocation


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/transactions
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/transactions")
def get_transactions(
    request: Request,
    isin: Optional[str] = None,
    txn_type: Optional[str] = None,
    limit: int = 200,
):
    data = _load_or_404(request.state.user_id)
    txns = data["all_transactions"]

    if isin:
        txns = [t for t in txns if t.get("isin") == isin]
    if txn_type:
        txns = [t for t in txns if t.get("type", "").upper() == txn_type.upper()]

    txns = sorted(txns, key=lambda x: x.get("date") or "", reverse=True)
    return txns[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/risk
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/risk")
async def get_risk(request: Request):
    data = _load_or_404(request.state.user_id)
    
    # Offload the NAV synchronization strictly to the asynchronous SQLite collector worker
    from data.data_collector import fetch_and_populate_mfapi_data
    await fetch_and_populate_mfapi_data(data["holdings"])
        
    # Hot-reload the latest category-to-benchmark assignments dynamically
    from core.parser import get_benchmark_ticker
    for h in data["holdings"]:
        h["benchmark"] = get_benchmark_ticker(h.get("name", ""), h.get("category", ""))
        
    try:
        metrics = compute_risk_metrics(data["holdings"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk computation failed: {str(e)}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/growth  — portfolio growth over time (invested vs value)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/growth")
def get_growth(request: Request):
    data = _load_or_404(request.state.user_id)
    all_txns = sorted(data["all_transactions"], key=lambda x: x.get("date") or "")

    running_invested = 0.0
    growth = []
    seen_months = {}

    for txn in all_txns:
        if not txn.get("date"):
            continue
        d = txn["date"][:7]  # YYYY-MM
        amt = abs(txn.get("amount") or 0)
        if txn.get("type") in ("BUY", "SIP", "SWITCH_IN", "DIVR"):
            running_invested += amt
        elif txn.get("type") in ("SELL", "SWITCH_OUT"):
            running_invested -= amt

        seen_months[d] = max(seen_months.get(d, 0), running_invested)

    for month, val in sorted(seen_months.items()):
        growth.append({"month": month, "invested": round(val, 2)})

    return growth


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/overlap
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/overlap")
async def get_overlap(request: Request, refresh: bool = False):
    """
    Fetch AMC monthly portfolio Excel files and compute pairwise overlap.
    Results are cached in session_data.json under 'overlap_cache'.
    Pass ?refresh=true to force a re-fetch.
    """
    data = _load_or_404(request.state.user_id)
    holdings_list = [h for h in data.get("holdings", []) if h.get("units", 0) > 0.001]

    if not holdings_list:
        raise HTTPException(status_code=400, detail="No active holdings found")

    # Return cached result if available and not stale
    if not refresh and data.get("overlap_cache"):
        return data["overlap_cache"]

    # Build fund → AMC name mapping from fund name
    def guess_amc(fund_name: str) -> str:
        """
        Extract AMC name from scheme name:
        e.g. 'Parag Parikh Flexi Cap...' -> 'PPFAS Mutual Fund'
        We map known prefixes.
        """
        name_lower = fund_name.lower()
        if "parag parikh" in name_lower or "ppfas" in name_lower:
            return "PPFAS Mutual Fund"
        if "mirae" in name_lower:
            return "Mirae Asset Mutual Fund"
        if "sbi" in name_lower:
            return "SBI Mutual Fund"
        if "hdfc" in name_lower:
            return "HDFC Mutual Fund"
        if "icici" in name_lower:
            return "ICICI Prudential Mutual Fund"
        if "axis" in name_lower:
            return "Axis Mutual Fund"
        if "kotak" in name_lower:
            return "Kotak Mahindra Mutual Fund"
        if "nippon" in name_lower or "reliance" in name_lower:
            return "Nippon India Mutual Fund"
        if "motilal" in name_lower:
            return "Motilal Oswal Mutual Fund"
        if "dsp" in name_lower:
            return "DSP Mutual Fund"
        if "uti" in name_lower:
            return "UTI Mutual Fund"
        if "aditya birla" in name_lower or "absl" in name_lower:
            return "Aditya Birla Sun Life Mutual Fund"
        if "franklin" in name_lower:
            return "Franklin Templeton Mutual Fund"
        if "tata" in name_lower:
            return "Tata Mutual Fund"
        if "bandhan" in name_lower:
            return "Bandhan Mutual Fund"
        if "quant" in name_lower:
            return "Quant Mutual Fund"
        if "canara" in name_lower:
            return "Canara Robeco Mutual Fund"
        if "invesco" in name_lower:
            return "Invesco Mutual Fund"
        if "whiteoak" in name_lower or "white oak" in name_lower:
            return "WhiteOak Capital Mutual Fund"
        if "edelweiss" in name_lower:
            return "Edelweiss Mutual Fund"
        return fund_name  # fallback

    # Only process equity/hybrid funds (debt/liquid funds have no stock overlap to speak of)
    def is_equity_like(category: str) -> bool:
        cat = (category or "").lower()
        skip_keywords = ["debt", "liquid", "overnight", "money market", "gilt", "bond", "credit risk", "banking and psu"]
        return not any(kw in cat for kw in skip_keywords)

    equity_holdings = [h for h in holdings_list if is_equity_like(h.get("category", ""))]

    if len(equity_holdings) < 2:
        return {
            "fund_count": len(equity_holdings),
            "message": "Need at least 2 equity/hybrid funds to compute overlap",
            "all_pairs": [],
            "top_pairs": [],
            "per_fund": {},
        }

    # Fetch holdings concurrently
    raw_holdings_map: dict[str, dict] = {}
    mstar_id_map: dict[str, str] = {} # fund_name -> mstar_id

    async def _fetch(h: dict):
        fn = h["name"]
        mstar_fund = await asyncio.get_event_loop().run_in_executor(
            None, _scraper.search_fund, fn
        )
        if mstar_fund:
            try:
                mid = mstar_fund['id']
                mstar_id_map[fn] = mid
                result = await asyncio.get_event_loop().run_in_executor(
                    None, _scraper.get_portfolio, mid
                )
                raw_holdings_map[fn] = result
            except Exception as exc:
                logger.warning("Failed to get portfolio for %s: %s", fn, exc)
                raw_holdings_map[fn] = {}
        else:
            raw_holdings_map[fn] = {}

    for h in equity_holdings:
        await _fetch(h)
        await asyncio.sleep(0.3)

    result = compute_overlap(raw_holdings_map)

    # Cache in session
    data["overlap_cache"] = result
    save_session(data, SESSION_FILE)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/taxes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/taxes")
async def get_taxes(request: Request):
    """
    Computes current tax liability and exit loads based on FIFO sales of all holdings today.
    """
    data = _load_or_404(request.state.user_id)
    txns = data.get("all_transactions", [])
    holdings = data.get("holdings", [])
    
    if not txns:
        return {"error": "No transaction history to compute taxes"}
        
    # Build dictionary of current NAVs using live_nav from session data
    current_navs = {h["name"]: h.get("live_nav") or _get_last_nav(h.get("transactions", [])) or 0 for h in holdings}
    
    result = calculate_taxes_and_loads(txns, current_navs)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/goal-strategy
# ─────────────────────────────────────────────────────────────────────────────
class GoalRequest(BaseModel):
    target_amount: float
    horizon_years: int
    include_current_portfolio: bool = True

@app.post("/api/goal-strategy")
async def get_goal_strategy(request: Request, req: GoalRequest):
    """
    Calculates SIP required to reach a goal and suggests asset allocation.
    """
    current_value = 0.0
    if req.include_current_portfolio:
        try:
            data = _load_or_404(request.state.user_id)
            holdings = data.get("holdings", [])
            current_value = sum(h.get("current_value", 0) for h in holdings)
        except Exception:
            pass # Ignore if no session
            
    result = calculate_goal_strategy(req.target_amount, req.horizon_years, current_value)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/simulations
# ─────────────────────────────────────────────────────────────────────────────
class SimulationRequest(BaseModel):
    monthly_sip: float
    step_up_pct: float
    horizon_years: int
    mean_return_pct: float
    volatility_pct: float

@app.post("/api/simulations")
async def run_simulations(request: Request, req: SimulationRequest):
    """
    Runs both SIP Step-up deterministic model and Monte Carlo stochastic model.
    """
    try:
        data = _load_or_404(request.state.user_id)
        holdings = data.get("holdings", [])
        current_value = 0.0
        for h in holdings:
            curr_nav = h.get("live_nav") or _get_last_nav(h.get("transactions", []))
            current_value += h.get("units", 0) * curr_nav if curr_nav else 0.0
    except Exception:
        current_value = 0.0

    step_up_res = calculate_sip_step_up(
        req.monthly_sip, req.step_up_pct, req.horizon_years, req.mean_return_pct
    )
    
    mc_res = run_monte_carlo_simulation(
        current_value, req.monthly_sip, req.horizon_years, 
        req.mean_return_pct, req.volatility_pct, num_paths=500
    )
    
    return {
        "step_up_analysis": step_up_res,
        "monte_carlo_analysis": mc_res
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/stress-test
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/stress-test")
async def get_stress_test(request: Request):
    """
    Calculates portfolio drop in historical crash scenarios based on current allocation.
    """
    data = _load_or_404(request.state.user_id)
    holdings = data.get("holdings", [])
    
    current_value = 0.0
    equity_value = 0.0
    
    for h in holdings:
        curr_nav = h.get("live_nav") or _get_last_nav(h.get("transactions", []))
        val = h.get("units", 0) * curr_nav if curr_nav else 0.0
        current_value += val
        
        cat = (h.get("category") or "").lower()
        if "equity" in cat or "flexi" in cat or "cap" in cat or "sectoral" in cat:
            equity_value += val
        elif "hybrid" in cat or "balanced" in cat:
            equity_value += val * 0.65 # Assume 65% equity for hybrid
            
    equity_pct = (equity_value / current_value * 100) if current_value > 0 else 0
    
    result = calculate_stress_test(current_value, equity_pct)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/rebalance
# ─────────────────────────────────────────────────────────────────────────────
class RebalanceRequest(BaseModel):
    target_equity_pct: float

@app.post("/api/rebalance")
async def get_rebalance(request: Request, req: RebalanceRequest):
    """
    Calculates trades required to reach target equity allocation.
    """
    data = _load_or_404(request.state.user_id)
    holdings = data.get("holdings", [])
    
    # Needs updated current value from summary endpoint caching
    result = calculate_rebalance(holdings, req.target_equity_pct)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/dividends
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/dividends")
async def get_dividends(request: Request):
    """
    Extracts total dividends from the transaction history.
    """
    data = _load_or_404(request.state.user_id)
    txns = data.get("all_transactions", [])
    
    result = calculate_dividend_cashflow(txns)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/fund/{isin} Deep-Dive Integration
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/fund/{isin}")
async def get_fund_details(request: Request, isin: str):
    """Return fund details using Moneycontrol for metrics and Morningstar for portfolio composition."""
    from data.database import get_nav_series, DB_PATH, get_cached_fund_deep_dive, cache_fund_deep_dive, get_connection

    # ── 1. Base Scheme from DB ─────────────────────────────────────────────────
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT scheme_name, category, benchmark, scheme_code FROM schemes WHERE isin = ?", (isin,))
    scheme = c.fetchone()
    conn.close()

    # Preserve CAS-based XIRR for this individual holding.
    holding_xirr = None
    active_holding = None
    try:
        session_data = _load_or_404(request.state.user_id)
        active_holding = next((h for h in session_data.get("holdings", []) if h.get("isin") == isin), None)
        if active_holding:
            holding_xirr = active_holding.get("xirr")
    except HTTPException:
        session_data = None
    except Exception:
        session_data = None

    if not scheme:
        if active_holding:
            # Fallback to session data if not in DB
            from core.parser import get_benchmark_ticker
            name = active_holding.get("name", "Unknown Fund")
            cat = active_holding.get("category", "")
            scheme = {
                "scheme_name": name,
                "category": cat,
                "benchmark": get_benchmark_ticker(name, cat),
                "scheme_code": None
            }
        else:
            raise HTTPException(status_code=404, detail="Fund ISIN not indexed in Database.")

    scheme_name = scheme["scheme_name"]
    category = scheme["category"]
    scheme_code = scheme["scheme_code"]
    benchmark_symbol = scheme["benchmark"]

    # Map Yahoo Finance / MC ticker symbols to human-readable index names
    _TICKER_TO_NAME = {
        "^NSEI":         "Nifty 50",
        "^BSESN":        "BSE Sensex",
        "^NSEMDCP50":    "Nifty Midcap 50",
        "^CNX100":       "Nifty 100",
        "^CNX200":       "Nifty 200",
        "^CRSLDX":       "Nifty 500",
        "^NSMIDCP":      "Nifty Midcap 150",
        "HDFCSML250.NS": "Nifty Smallcap 250",
        "NIFTYSMALLCAP250.NS": "Nifty Smallcap 250",
        "CRSLDX": "Nifty 500",
        "NSMIDCP":       "Nifty Midcap 150",
        "^NIFTY_MID_SELECT": "Nifty Midcap Select",
    }
    readable_benchmark = _TICKER_TO_NAME.get(benchmark_symbol, benchmark_symbol)

    # ── 2. Check SQLite Cache (1-Hour Expiration) ─────────────────────────────
    # Debug toggle: keep the live path hot while we inspect fundamentals parsing.
    # cached_fund = get_cached_fund_deep_dive(isin, max_age_hours=1)
    cached_fund = None if DEBUG_BYPASS_DEEP_DIVE_CACHE else get_cached_fund_deep_dive(isin, max_age_hours=24)
    
    if cached_fund:
        fallback_benchmark = cached_fund.get("risk", {}).get("benchmark_name") or scheme["benchmark"]
        risk_data = cached_fund.get("risk", {})
        returns_data = cached_fund.get("returns", {})
        benchmark_returns = cached_fund.get("benchmark_cagr", {})
        sorted_holdings = cached_fund.get("holdings", [])
        sector_allocation = cached_fund.get("sectors", [])
        
        fundms = cached_fund.get("fundamentals", {})
        # Stale cache detection: if price_sale is missing, this is an old cache entry — re-fetch
        if fundms.get("price_sale") is None and fundms.get("pe") is None:
            cached_fund = None  # Force cache miss path below
        else:
            mfapi_data = {
                "aum_cr": fundms.get("aum_cr"),
                "expense_ratio": fundms.get("expense_ratio"),
                "exit_load": fundms.get("exit_load"),
                "current_nav": None,
                "nav_date": None,
            }
            portfolio_turnover = fundms.get("portfolio_turnover")

    if not cached_fund:
        # ── 3. Moneycontrol & Morningstar Fetching (Cache Miss) ──────────────────
        from scrapers.morningstar import MorningstarScraper
        from scrapers.moneycontrol import MoneyControlScraper
        ms = MorningstarScraper()
        mc = MoneyControlScraper()
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ms_fund, mc_risk, mc_perf, mc_perf_yearly, mc_perf_sip, mc_fund, mc_overview = await asyncio.gather(
                loop.run_in_executor(pool, ms.search_fund, scheme_name),
                loop.run_in_executor(pool, mc.get_risk_metrics, isin),
                loop.run_in_executor(pool, mc.get_performance, isin),
                loop.run_in_executor(pool, mc.get_performance_yearly, isin),
                loop.run_in_executor(pool, mc.get_performance_sip, isin),
                loop.run_in_executor(pool, mc.get_fundamentals, isin),
                loop.run_in_executor(pool, mc.get_overview, isin),
            )
        # Merge overview data (AUM, expense, turnover) into mc_fund for _mc_extract_fundamentals
        if mc_overview:
            mc_fund = {**mc_overview, **(mc_fund or {})}

    
        # ── 4. Process Deep Dive Data ────────────────────────────────────────────────
        # Merge performance and risk return streams to combat MC API zeroing anomalies (0.00%)
        returns_data, benchmark_returns = _mc_extract_period_returns(mc_perf)
        risk_returns, fallback_benchmark = _mc_extract_period_returns(mc_risk)
        
        for k in returns_data:
            if returns_data.get(k) in (0.0, None) and risk_returns.get(k) not in (0.0, None):
                returns_data[k] = risk_returns[k]
                
        for k in benchmark_returns:
            if benchmark_returns.get(k) in (0.0, None) and fallback_benchmark.get(k) not in (0.0, None):
                benchmark_returns[k] = fallback_benchmark[k]

        fallback_benchmark_name = (
            _mc_find_first(mc_perf, "benchmark_name", "benchmark", "benchmarklabel")
            or _mc_find_first(mc_risk, "benchmark_name", "benchmark", "benchmarklabel")
            or scheme["benchmark"]
        )
        risk_data = _mc_extract_risk(mc_risk, fallback_benchmark_name)
        if mc_risk and all(value is None for key, value in risk_data.items() if key != "benchmark_name"):
            logger.warning(
                "Moneycontrol risk payload parsed empty for %s. Top-level keys: %s. Periods: %s. Sharpe sample: %s",
                isin,
                list(mc_risk.keys())[:20] if isinstance(mc_risk, dict) else type(mc_risk).__name__,
                mc_risk.get("period") if isinstance(mc_risk, dict) else None,
                mc_risk.get("sharpe_ratio") if isinstance(mc_risk, dict) else None,
            )
        if mc_perf and all(value is None for value in returns_data.values()):
            logger.warning("Moneycontrol performance payload parsed empty for %s. Top-level keys: %s", isin, list(mc_perf.keys())[:20] if isinstance(mc_perf, dict) else type(mc_perf).__name__)
    
        mc_fundamentals = _mc_extract_fundamentals(mc_fund)
        sorted_holdings: list[dict] = []
        sector_allocation: list = []
        portfolio_turnover: Optional[float] = mc_fundamentals["portfolio_turnover"]
    
        if ms_fund:
            try:
                ms_id = ms_fund["id"]
                raw_portfolio, fund_info = await asyncio.gather(
                    loop.run_in_executor(None, ms.get_portfolio, ms_id),
                    loop.run_in_executor(None, ms.get_fund_info, ms_id),
                )
                sorted_holdings = [
                    {"asset": asset, "weight": round(weight * 100, 2)}
                    for asset, weight in sorted(raw_portfolio.items(), key=lambda item: item[1], reverse=True)
                ][:20]
                sector_allocation = fund_info.get("sector_allocation", []) or []
                if not DEBUG_DISABLE_FUNDAMENTALS_FALLBACK and mc_fundamentals["aum_cr"] is None:
                    mc_fundamentals["aum_cr"] = fund_info.get("aum_cr")
                if not DEBUG_DISABLE_FUNDAMENTALS_FALLBACK and mc_fundamentals["expense_ratio"] is None:
                    mc_fundamentals["expense_ratio"] = fund_info.get("expense_ratio")
                if not DEBUG_DISABLE_FUNDAMENTALS_FALLBACK and portfolio_turnover is None:
                    portfolio_turnover = fund_info.get("portfolio_turnover_pct")
            except Exception:
                pass
                
        fundms = {
            "aum_cr": mc_fundamentals["aum_cr"],
            "expense_ratio": mc_fundamentals["expense_ratio"],
            "exit_load": mc_fundamentals["exit_load"],
            "portfolio_turnover": portfolio_turnover,
            "pe": mc_fundamentals.get("pe"),
            "cat_avg_pe": mc_fundamentals.get("cat_avg_pe"),
            "pb": mc_fundamentals.get("pb"),
            "cat_avg_pb": mc_fundamentals.get("cat_avg_pb"),
            "price_sale": mc_fundamentals.get("price_sale"),
            "cat_avg_price_sale": mc_fundamentals.get("cat_avg_price_sale"),
            "price_cash_flow": mc_fundamentals.get("price_cash_flow"),
            "cat_avg_price_cash_flow": mc_fundamentals.get("cat_avg_price_cash_flow"),
            "dividend_yield": mc_fundamentals.get("dividend_yield"),
            "cat_avg_dividend_yield": mc_fundamentals.get("cat_avg_dividend_yield"),
            "roe": mc_fundamentals.get("roe"),
            "cat_avg_roe": mc_fundamentals.get("cat_avg_roe")
        }

        mfapi_data = {
            "aum_cr": fundms["aum_cr"],
            "expense_ratio": fundms["expense_ratio"],
            "exit_load": fundms["exit_load"],
            # Seed NAV from MC overview immediately — will be overwritten by mfapi if available
            "current_nav": mc_overview.get("latest_nav") if mc_overview else None,
            "nav_date": mc_overview.get("nav_date") if mc_overview else None,
        }

        logger.info("Parsed Moneycontrol fundamentals for %s: %s", isin, fundms)

        # Debug toggle: skip cache writes so each request reflects live scraper output.
        if not DEBUG_BYPASS_DEEP_DIVE_CACHE:
            cache_fund_deep_dive(
                isin=isin,
                fundamentals=fundms,
                risk=risk_data,
                returns=returns_data,
                bench_returns=benchmark_returns,
                holdings=sorted_holdings,
                sectors=sector_allocation
            )


    # Seed NAV from locally stored history so the UI does not go blank when live fetch fails.
    navs = get_nav_series(isin)
    if navs:
        latest_nav = navs[-1]
        try:
            mfapi_data["current_nav"] = float(latest_nav["nav"])
            mfapi_data["nav_date"] = latest_nav["date"]
        except (TypeError, ValueError, KeyError):
            pass

    try:
        import requests as _requests, time as _time
        from data.database import insert_or_update_scheme as _insert_or_update_scheme

        def _resolve_scheme_code_for_nav(code, fund_name, fund_isin):
            if code:
                return code

            try:
                query = _requests.utils.quote((fund_name or "")[:60])
                r = _requests.get(f"https://api.mfapi.in/mf/search?q={query}", timeout=10)
                r.raise_for_status()
                results = r.json() or []
                for entry in results:
                    if fund_isin in (entry.get("isinGrowth"), entry.get("isinDivReinvestment")):
                        return entry.get("schemeCode")
                if results:
                    return results[0].get("schemeCode")
            except Exception:
                return None
            return None

        def _fetch_nav(code):
            for attempt in range(3):
                try:
                    r = _requests.get(f"https://api.mfapi.in/mf/{code}", timeout=10)
                    r.raise_for_status()
                    return r.json()
                except Exception:
                    if attempt < 2:
                        _time.sleep(1)
            raise Exception(f"mfapi failed for scheme {code} after 3 attempts")

        resolved_scheme_code = await asyncio.get_event_loop().run_in_executor(
            None, _resolve_scheme_code_for_nav, scheme_code, scheme_name, isin
        )
        if resolved_scheme_code and resolved_scheme_code != scheme_code:
            scheme_code = resolved_scheme_code
            await asyncio.get_event_loop().run_in_executor(
                None, _insert_or_update_scheme, isin, scheme_name, str(scheme_code)
            )

        if scheme_code:
            detail = await asyncio.get_event_loop().run_in_executor(None, _fetch_nav, scheme_code)
            nav_data = detail.get("data", [])
            if nav_data:
                mfapi_data["current_nav"] = float(nav_data[0]["nav"])
                mfapi_data["nav_date"] = nav_data[0]["date"]
                # Convert mfapi date format (DD-MM-YYYY) → YYYY-MM-DD and store + expose for chart
                from datetime import datetime as _dt
                from data.database import batch_insert_navs as _batch_insert_navs
                converted = []
                for pt in nav_data:
                    try:
                        d_str = pt["date"]
                        # Handle both DD-MM-YYYY and DD-MMM-YYYY formats
                        try:
                            d_parsed = _dt.strptime(d_str, "%d-%m-%Y")
                        except ValueError:
                            d_parsed = _dt.strptime(d_str, "%d-%b-%Y")
                        converted.append({"date": d_parsed.strftime("%Y-%m-%d"), "nav": float(pt["nav"])})
                    except Exception:
                        continue
                if converted:
                    # Persist to DB for future calls
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, _batch_insert_navs, isin, sorted(converted, key=lambda x: x["date"])
                        )
                    except Exception as _bi_err:
                        logger.warning("batch_insert_navs failed for %s: %s", isin, _bi_err)
                    # Override navs so chart uses fresh data even on first view
                    navs = sorted(converted, key=lambda x: x["date"])
    except Exception as _nav_err:
        import logging as _log
        _log.getLogger("app").warning("NAV fetch failed for %s: %s", isin, _nav_err)


    # ── 4. Chart: SIP wealth (if txns) else ₹10k NAV growth vs benchmark ─────
    sip_chart: dict = {}
    _chart_logger = __import__("logging").getLogger("app.chart")
    try:
        holding = next((h for h in session_data.get("holdings", []) if h.get("isin") == isin), None) if session_data else None
        benchmark_label = risk_data.get("benchmark_name") or fallback_benchmark
        import pandas as pd
        from datetime import date, timedelta
        from core.risk import _fetch_benchmark_returns

        if not navs:
            raise ValueError("No NAV data available; skipping chart")

        nav_df = pd.DataFrame(navs)
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df = nav_df.set_index("date").sort_index()
        monthly_nav = nav_df["nav"].resample("ME").last().dropna()

        if monthly_nav.empty:
            raise ValueError("Monthly NAV series is empty")

        # Try to fetch benchmark (optional — chart still works without it)
        bench_s = None
        bench_monthly_nav = None
        if benchmark_symbol:
            bench_start = (date.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
            bench_end   = date.today().strftime("%Y-%m-%d")
            cache_key   = f"{benchmark_symbol}_{bench_start}_{bench_end}"
            try:
                if cache_key in _benchmark_cache:
                    bench_s = _benchmark_cache[cache_key]
                else:
                    bench_s = await asyncio.get_event_loop().run_in_executor(
                        None, _fetch_benchmark_returns, benchmark_symbol, bench_start, bench_end
                    )
                    if bench_s is not None:
                        _benchmark_cache[cache_key] = bench_s
                if bench_s is not None:
                    bench_monthly_nav = (1 + bench_s).cumprod()
                else:
                    _chart_logger.warning("Benchmark %s returned None for %s; rendering fund-only chart", benchmark_symbol, isin)
            except Exception as _be:
                _chart_logger.warning("Benchmark fetch failed for %s / %s: %s", isin, benchmark_symbol, _be)

        # Collect SIP transactions (fund must be in portfolio)
        sip_txns = []
        if holding:
            sip_txns = [t for t in holding.get("transactions", [])
                        if t["type"] in ("SIP", "BUY", "SWITCH_IN") and t.get("nav") and t.get("amount")]
            sip_txns.sort(key=lambda x: x["date"])

        labels, fund_vals, bench_vals = [], [], []

        if sip_txns and bench_monthly_nav is not None:
            # ── Path A: actual SIP wealth chart ─────────────────────────────
            fund_units = 0.0
            bench_units = 0.0
            for txn in sip_txns:
                d = pd.Timestamp(txn["date"])
                amount = abs(txn["amount"])
                fund_units += amount / txn["nav"]
                ym = d.to_period("M")
                matches = bench_monthly_nav[bench_monthly_nav.index.to_period("M") == ym]
                if not matches.empty:
                    bench_units += amount / (100.0 * float(matches.iloc[-1]))
            for ts, nav_val in monthly_nav[-60:].items():
                labels.append(ts.strftime("%b %Y"))
                fund_vals.append(round(fund_units * float(nav_val), 2))
                ym = ts.to_period("M")
                bm = bench_monthly_nav[bench_monthly_nav.index.to_period("M") == ym]
                bench_vals.append(round(bench_units * 100.0 * float(bm.iloc[-1]) if not bm.empty else 0, 2))
            chart_type = "sip"
        else:
            # ── Path B: ₹10,000 lumpsum NAV growth (benchmark optional) ─────
            start_nav = None
            start_bench = None
            for ts, nav_val in monthly_nav[-60:].items():
                if start_nav is None:
                    start_nav = float(nav_val)
                    if bench_monthly_nav is not None:
                        ym0 = ts.to_period("M")
                        bm0 = bench_monthly_nav[bench_monthly_nav.index.to_period("M") == ym0]
                        start_bench = float(bm0.iloc[-1]) if not bm0.empty else None
                labels.append(ts.strftime("%b %Y"))
                fund_vals.append(round(10000 * float(nav_val) / start_nav, 2))
                if bench_monthly_nav is not None and start_bench:
                    ym = ts.to_period("M")
                    bm = bench_monthly_nav[bench_monthly_nav.index.to_period("M") == ym]
                    bench_vals.append(round(float(bm.iloc[-1]) / start_bench * 10000 if not bm.empty else 0, 2))
                else:
                    bench_vals.append(None)
            chart_type = "growth"

        if labels:
            sip_chart = {
                "labels":          labels,
                "fund_value":      fund_vals,
                "benchmark_value": bench_vals,
                "benchmark_name":  benchmark_label if bench_s is not None else None,
                "chart_type":      chart_type,
            }
            _chart_logger.info("Chart built for %s: type=%s labels=%d bench_available=%s",
                               isin, chart_type, len(labels), bench_s is not None)
    except Exception as _chart_err:
        _chart_logger.warning("Chart build failed for %s: %s", isin, _chart_err, exc_info=True)



    return {
        "isin":               isin,
        "name":               scheme_name,
        "category":           category,
        "benchmark":          risk_data.get("benchmark_name") or readable_benchmark,
        "aum_cr":             mfapi_data["aum_cr"],
        "expense_ratio":      mfapi_data["expense_ratio"],
        "exit_load":          mfapi_data["exit_load"],
        "current_nav":        mfapi_data["current_nav"],
        "nav_date":           mfapi_data["nav_date"],
        "portfolio_turnover": portfolio_turnover,
        "fundamentals":       fundms,
        "risk":               risk_data,
        "returns":            returns_data,
        "fund_trailing":      returns_data,
        "benchmark_cagr":     benchmark_returns,
        "performance_annualised": (
            # Extract lumpsum.annualised from mc_perf (which is a list like [{lumpsum:{annualised:[...],...}},...])
            (mc_perf[0].get("lumpsum", {}).get("annualised", []) if isinstance(mc_perf, list) and mc_perf else [])
            if not cached_fund else []
        ),
        "performance_yearly":     mc_perf_yearly if not cached_fund else [],
        "performance_sip":        mc_perf_sip if not cached_fund else [],

        "sector_allocation":  sector_allocation,
        "holdings":           sorted_holdings,
        "sip_vs_benchmark":   sip_chart,
        "xirr":               holding_xirr,
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat_with_advisor(request: Request, req: ChatRequest):
    if not UPSTAGE_API_KEY:
        raise HTTPException(status_code=500, detail="Upstage API key not configured.")
        
    try:
        data = _load_or_404(request.state.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="No portfolio data available context.")

    # Build context
    total_invested = 0.0
    current_value = 0.0
    fund_details = []
    
    for h in data.get("holdings", []):
        curr_nav = h.get("live_nav") or _get_last_nav(h.get("transactions", []))
        
        if h.get("units", 0) <= 0.001:
            invested = 0.0
            val = 0.0
            gain_pct = 0.0
        else:
            invested = sum(abs(t["amount"]) for t in h.get("transactions", []) if t["type"] in ("BUY", "SIP", "SWITCH_IN", "DIVR")) - sum(abs(t["amount"]) for t in h.get("transactions", []) if t["type"] in ("SELL", "SWITCH_OUT"))
            val = h.get("units", 0) * curr_nav if curr_nav else 0.0
            gain = val - invested
            gain_pct = (gain / invested * 100) if invested > 0 else 0.0
            
        total_invested += invested
        current_value += val
        
        if h.get("units", 0) > 0.001:
            fund_details.append(
                f"- {h['name']} ({h.get('category', 'Unknown')}): "
                f"Value: ₹{val:,.2f}, Gain: {gain_pct:.2f}%"
            )

    xirr_val = data.get("portfolio_xirr", "N/A")
    
    system_prompt = f"""You are an expert personalized AI Financial Advisor.
You are currently advising a user on their real mutual fund portfolio in India.

PORTFOLIO SNAPSHOT:
- Total Invested: ₹{total_invested:,.2f}
- Current Value: ₹{current_value:,.2f}
- Active Portfolio XIRR: {xirr_val}%

ACTIVE FUNDS:
{chr(10).join(fund_details)}

Given this exact data, provide concise, professional, and personalized financial advice to the user's questions. 
Do not hallucinate funds they don't own. Frame your answers around their actual performance. Format in markdown."""

    client = OpenAI(
        api_key=UPSTAGE_API_KEY,
        base_url="https://api.upstage.ai/v1/solar"
    )
    
    try:
        response = client.chat.completions.create(
            model="solar-pro3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message}
            ],
            max_tokens=800,
            stream=False
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_or_404(user_id: str) -> dict:
    try:
        return load_session(user_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No portfolio data found. Please upload your CAS PDF first."
        )


def _get_last_nav(transactions: list) -> Optional[float]:
    """Return the most recent non-null NAV from a transaction list."""
    navs = [(t["date"], t["nav"]) for t in transactions if t.get("nav") and t.get("date")]
    if not navs:
        return None
    navs.sort(key=lambda x: x[0], reverse=True)
    return navs[0][1]


def _normalize_category(category: str) -> str:
    """Map raw casparser category strings to clean labels."""
    cat = category.lower()
    if "large" in cat:
        return "Large Cap"
    if "mid" in cat:
        return "Mid Cap"
    if "small" in cat:
        return "Small Cap"
    if "flexi" in cat or "multi" in cat:
        return "Flexi/Multi Cap"
    if "elss" in cat or "tax" in cat:
        return "ELSS / Tax Saver"
    if "hybrid" in cat or "balanced" in cat:
        return "Hybrid"
    if "debt" in cat or "bond" in cat or "gilt" in cat or "liquid" in cat or "money" in cat or "overnight" in cat:
        return "Debt / Liquid"
    if "index" in cat or "etf" in cat:
        return "Index / ETF"
    if "sector" in cat or "thematic" in cat or "bank" in cat or "technology" in cat or "pharma" in cat:
        return "Sectoral / Thematic"
    return "Other"


from core.mc_helpers import _mc_to_float, _mc_find_first, _mc_extract_metric_value, _mc_extract_period_returns, _mc_extract_risk, _mc_extract_fundamentals, _mc_extract_holdings


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# ── Family Dashboard Routes ──────────────────────────────────────────────────
class FamilyInviteRequest(BaseModel):
    username: str

@app.post("/api/family/invite")
async def invite_family(request: Request, req: FamilyInviteRequest):
    from data.database import get_user_by_username, send_family_invite
    target_user = get_user_by_username(req.username)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    if target_user['id'] == request.state.user_id:
        raise HTTPException(status_code=400, detail="Cannot invite yourself")
    send_family_invite(request.state.user_id, target_user['id'])
    return {"status": "ok", "message": "Invite sent successfully"}

@app.get("/api/family/pending")
async def get_pending_invites(request: Request):
    from data.database import get_pending_invites
    return get_pending_invites(request.state.user_id)

class AcceptInviteRequest(BaseModel):
    from_user_id: str

@app.post("/api/family/accept")
async def accept_family_invite(request: Request, req: AcceptInviteRequest):
    from data.database import accept_family_invite
    accept_family_invite(req.from_user_id, request.state.user_id)
    return {"status": "ok", "message": "Invite accepted"}

@app.get("/api/family/portfolios")
async def get_family_portfolios(request: Request):
    from data.database import get_family_members, get_user_by_id
    from core.parser import load_session
    
    user_ids = [request.state.user_id] + get_family_members(request.state.user_id)
    portfolios = []
    
    for uid in user_ids:
        try:
            data = load_session(uid)
            user_info = get_user_by_id(uid)
            total_invested = 0.0
            current_value = 0.0
            for h in data.get("holdings", []):
                curr_nav = h.get("live_nav")
                if not curr_nav:
                    navs = [(t["date"], t["nav"]) for t in h.get("transactions", []) if t.get("nav")]
                    if navs:
                        navs.sort(key=lambda x: x[0], reverse=True)
                        curr_nav = navs[0][1]
                val = h.get("units", 0) * (curr_nav if curr_nav else 0.0)
                current_value += val
                
                invested = sum(abs(t["amount"]) for t in h.get("transactions", []) if t["type"] in ("BUY", "SIP", "SWITCH_IN", "DIVR")) - sum(abs(t["amount"]) for t in h.get("transactions", []) if t["type"] in ("SELL", "SWITCH_OUT"))
                total_invested += invested
                
            portfolios.append({
                "user_id": uid,
                "username": user_info["username"] if user_info else "Unknown",
                "total_invested": round(total_invested, 2),
                "current_value": round(current_value, 2),
                "xirr": data.get("portfolio_xirr")
            })
        except Exception:
            pass
            
    return portfolios
