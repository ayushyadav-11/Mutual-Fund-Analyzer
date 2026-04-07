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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.parser import parse_cas, save_session, load_session, recompute_xirr
from core.risk import compute_risk_metrics
from data.database import initialize_database
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

@app.on_event("startup")
def startup_event():
    logger.info("Initializing database schemas...")
    try:
        initialize_database()
        logger.info("Database schemas verified successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MASTER_PASSWORD = os.getenv("MASTER_PASSWORD", "admin123")

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Exclude frontend root and login endpoints from auth logic
        if request.method == "OPTIONS" or request.url.path in ["/", "/api/login", "/favicon.ico"]:
            return await call_next(request)
        
        # Guard internal API routes
        if request.url.path.startswith("/api/"):
            # Check Bearer Token
            auth_header = request.headers.get("Authorization")
            if not auth_header or auth_header != f"Bearer {MASTER_PASSWORD}":
                return JSONResponse(
                    status_code=401, 
                    content={"detail": "Unauthorized access. Please login."},
                    headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true"}
                )
                
        return await call_next(request)

app.add_middleware(AuthMiddleware)

class LoginRequest(BaseModel):
    password: str

@app.post("/api/login")
async def login(req: LoginRequest):
    if req.password == MASTER_PASSWORD:
        return {"token": MASTER_PASSWORD, "message": "Authenticated successfully"}
    raise HTTPException(status_code=401, detail="Incorrect password")

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
    return FileResponse(str(frontend_dir / "index.html"))


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
        save_session(data, SESSION_FILE, merge=merge)
        
        # If merged, reload to return the combined dataset
        if merge:
            data = load_session(SESSION_FILE)
            
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass

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
def get_summary():
    data = _load_or_404()
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
def get_holdings():
    data = _load_or_404()
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
def get_allocation():
    data = _load_or_404()
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
    isin: Optional[str] = None,
    txn_type: Optional[str] = None,
    limit: int = 200,
):
    data = _load_or_404()
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
async def get_risk():
    data = _load_or_404()
    
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
def get_growth():
    data = _load_or_404()
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
async def get_overlap(refresh: bool = False):
    """
    Fetch AMC monthly portfolio Excel files and compute pairwise overlap.
    Results are cached in session_data.json under 'overlap_cache'.
    Pass ?refresh=true to force a re-fetch.
    """
    data = _load_or_404()
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
async def get_taxes():
    """
    Computes current tax liability and exit loads based on FIFO sales of all holdings today.
    """
    data = _load_or_404()
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
async def get_goal_strategy(req: GoalRequest):
    """
    Calculates SIP required to reach a goal and suggests asset allocation.
    """
    current_value = 0.0
    if req.include_current_portfolio:
        try:
            data = _load_or_404()
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
async def run_simulations(req: SimulationRequest):
    """
    Runs both SIP Step-up deterministic model and Monte Carlo stochastic model.
    """
    try:
        data = _load_or_404()
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
async def get_stress_test():
    """
    Calculates portfolio drop in historical crash scenarios based on current allocation.
    """
    data = _load_or_404()
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
async def get_rebalance(req: RebalanceRequest):
    """
    Calculates trades required to reach target equity allocation.
    """
    data = _load_or_404()
    holdings = data.get("holdings", [])
    
    # Needs updated current value from summary endpoint caching
    result = calculate_rebalance(holdings, req.target_equity_pct)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/dividends
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/dividends")
async def get_dividends():
    """
    Extracts total dividends from the transaction history.
    """
    data = _load_or_404()
    txns = data.get("all_transactions", [])
    
    result = calculate_dividend_cashflow(txns)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/fund/{isin} Deep-Dive Integration
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/fund/{isin}")
async def get_fund_details(isin: str):
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
        session_data = _load_or_404()
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

    # ── 2. Check SQLite Cache (1-Hour Expiration) ─────────────────────────────
    cached_fund = get_cached_fund_deep_dive(isin, max_age_hours=1)
    
    if cached_fund:
        fallback_benchmark = cached_fund.get("risk", {}).get("benchmark_name") or scheme["benchmark"]
        risk_data = cached_fund.get("risk", {})
        returns_data = cached_fund.get("returns", {})
        benchmark_returns = cached_fund.get("benchmark_cagr", {})
        sorted_holdings = cached_fund.get("holdings", [])
        sector_allocation = cached_fund.get("sectors", [])
        
        fundms = cached_fund.get("fundamentals", {})
        mfapi_data = {
            "aum_cr": fundms.get("aum_cr"),
            "expense_ratio": fundms.get("expense_ratio"),
            "exit_load": fundms.get("exit_load"),
            "current_nav": None,
            "nav_date": None,
        }
        portfolio_turnover = fundms.get("portfolio_turnover")

    else:
        # ── 3. Moneycontrol & Morningstar Fetching (Cache Miss) ──────────────────
        from scrapers.morningstar import MorningstarScraper
        from scrapers.moneycontrol import MoneyControlScraper
        ms = MorningstarScraper()
        mc = MoneyControlScraper()
        loop = asyncio.get_event_loop()
        ms_fund, mc_risk, mc_perf, mc_fund = await asyncio.gather(
            loop.run_in_executor(None, ms.search_fund, scheme_name),
            loop.run_in_executor(None, mc.get_risk_metrics, isin),
            loop.run_in_executor(None, mc.get_performance, isin),
            loop.run_in_executor(None, mc.get_fundamentals, isin),
        )
    
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
                if mc_fundamentals["aum_cr"] is None:
                    mc_fundamentals["aum_cr"] = fund_info.get("aum_cr")
                if mc_fundamentals["expense_ratio"] is None:
                    mc_fundamentals["expense_ratio"] = fund_info.get("expense_ratio")
                if portfolio_turnover is None:
                    portfolio_turnover = fund_info.get("portfolio_turnover_pct")
            except Exception:
                pass
                
        mfapi_data = {
            "aum_cr": mc_fundamentals["aum_cr"],
            "expense_ratio": mc_fundamentals["expense_ratio"],
            "exit_load": mc_fundamentals["exit_load"],
            "current_nav": None,
            "nav_date": None,
        }
        
        # Save to Cache
        cache_fund_deep_dive(
            isin=isin,
            fundamentals={"aum_cr": mfapi_data["aum_cr"], "expense_ratio": mfapi_data["expense_ratio"], "exit_load": mfapi_data["exit_load"], "portfolio_turnover": portfolio_turnover},
            risk=risk_data,
            returns=returns_data,
            bench_returns=benchmark_returns,
            holdings=sorted_holdings,
            sectors=sector_allocation
        )


    try:
        if scheme_code:
            import requests as _requests, time as _time

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

            detail = await asyncio.get_event_loop().run_in_executor(None, _fetch_nav, scheme_code)
            nav_data = detail.get("data", [])
            if nav_data:
                mfapi_data["current_nav"] = float(nav_data[0]["nav"])
                mfapi_data["nav_date"]    = nav_data[0]["date"]
    except Exception as _nav_err:
        import logging as _log
        _log.getLogger("app").warning("NAV fetch failed for %s: %s", isin, _nav_err)

    # ── 4. Derived XIRR charting is preserved for SIP analysis only ───────────
    navs = get_nav_series(isin)
    sip_chart: dict = {}
    try:
        holding = next((h for h in session_data.get("holdings", []) if h.get("isin") == isin), None) if session_data else None
        benchmark_label = risk_data.get("benchmark_name") or fallback_benchmark
        if holding and navs and benchmark_symbol:
            import pandas as pd
            from datetime import date, timedelta
            from core.risk import _fetch_benchmark_returns

            nav_df = pd.DataFrame(navs)
            nav_df["date"] = pd.to_datetime(nav_df["date"])
            nav_df = nav_df.set_index("date").sort_index()

            sip_txns = [t for t in holding["transactions"]
                        if t["type"] in ("SIP", "BUY") and t.get("nav") and t.get("amount")]
            sip_txns.sort(key=lambda x: x["date"])

            if sip_txns:
                bench_start = (date.today() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")
                bench_end = date.today().strftime("%Y-%m-%d")
                cache_key = f"{benchmark_symbol}_{bench_start}_{bench_end}"
                
                if cache_key in _benchmark_cache:
                    bench_s = _benchmark_cache[cache_key]
                else:
                    bench_s = await asyncio.get_event_loop().run_in_executor(
                        None, _fetch_benchmark_returns, benchmark_symbol,
                        bench_start, bench_end
                    )
                    if bench_s is not None:
                        _benchmark_cache[cache_key] = bench_s

                if bench_s is not None:
                    # Build monthly SIP wealth index
                    labels, fund_vals, bench_vals = [], [], []
                    fund_units = 0.0
                    bench_units = 0.0
                    bench_monthly_nav = (1 + bench_s).cumprod()

                    for txn in sip_txns:
                        d = pd.Timestamp(txn["date"])
                        nav_val = txn["nav"]
                        amount  = abs(txn["amount"])
                        fund_units += amount / nav_val
                        # Simulate equal SIP in benchmark
                        ym = d.to_period("M")
                        matches = bench_monthly_nav[bench_monthly_nav.index.to_period("M") == ym]
                        if not matches.empty:
                            bench_units += amount / (100.0 * float(matches.iloc[-1]))

                    # Evaluate current value for each month
                    monthly_nav = nav_df["nav"].resample("ME").last().dropna()
                    for ts, nav_val in monthly_nav[-60:].items():
                        label = ts.strftime("%b %Y")
                        labels.append(label)
                        fund_vals.append(round(fund_units * float(nav_val), 2))
                        ym = ts.to_period("M")
                        bm = bench_monthly_nav[bench_monthly_nav.index.to_period("M") == ym]
                        bench_vals.append(round(bench_units * 100.0 * float(bm.iloc[-1]) if not bm.empty else 0, 2))

                    sip_chart = {
                        "labels": labels,
                        "fund_value": fund_vals,
                        "benchmark_value": bench_vals,
                        "benchmark_name": benchmark_label
                    }
    except Exception:
        pass  # sip_chart stays empty — non-critical

    return {
        "isin":               isin,
        "name":               scheme_name,
        "category":           category,
        "benchmark":          risk_data.get("benchmark_name") or benchmark_symbol,
        "aum_cr":             mfapi_data["aum_cr"],
        "expense_ratio":      mfapi_data["expense_ratio"],
        "exit_load":          mfapi_data["exit_load"],
        "current_nav":        mfapi_data["current_nav"],
        "nav_date":           mfapi_data["nav_date"],
        "portfolio_turnover": portfolio_turnover,
        "risk":               risk_data,
        "returns":            returns_data,
        "fund_trailing":      returns_data,
        "benchmark_cagr":     benchmark_returns,
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
def chat_with_advisor(req: ChatRequest):
    if not UPSTAGE_API_KEY:
        raise HTTPException(status_code=500, detail="Upstage API key not configured.")
        
    try:
        data = _load_or_404()
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
def _load_or_404() -> dict:
    try:
        return load_session(SESSION_FILE)
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


def _mc_to_float(value: Any) -> Optional[float]:
    """Best-effort numeric parsing for Moneycontrol payload values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned in {"--", "-", "N/A", "NA", "null", "None"}:
            return None
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("%", "")
        cleaned = cleaned.replace("₹", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _mc_find_first(payload: Any, *keys: str) -> Any:
    """Recursively find the first non-empty value for any matching key."""
    def _norm(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    normalized = {_norm(k) for k in keys}

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in node.items():
                if _norm(key) in normalized and value not in (None, "", [], {}):
                    return value
            for value in node.values():
                found = _walk(value)
                if found not in (None, "", [], {}):
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found not in (None, "", [], {}):
                    return found
        return None

    return _walk(payload)


def _mc_period_label(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    token = re.sub(r"[^a-z0-9]", "", str(raw).strip().lower())
    mapping = {
        "1y": "1Y", "1yr": "1Y", "1year": "1Y", "oneyear": "1Y",
        "1years": "1Y", "1yrs": "1Y",
        "3y": "3Y", "3yr": "3Y", "3year": "3Y", "threeyear": "3Y",
        "3years": "3Y", "3yrs": "3Y",
        "5y": "5Y", "5yr": "5Y", "5year": "5Y", "fiveyear": "5Y",
        "5years": "5Y", "5yrs": "5Y",
        "10y": "10Y", "10yr": "10Y", "10year": "10Y", "tenyear": "10Y",
        "10years": "10Y", "10yrs": "10Y",
    }
    return mapping.get(token)


def _mc_pick_period_value(periods: Any, values: Any, preferred: tuple[str, ...] = ("3Y", "5Y", "1Y", "10Y")) -> Optional[float]:
    if isinstance(periods, dict) and isinstance(values, dict):
        normalized_periods = {str(key).lower(): _mc_period_label(val) for key, val in periods.items()}
        for target in preferred:
            for raw_key, label in normalized_periods.items():
                if label == target and raw_key in values:
                    numeric = _mc_to_float(values.get(raw_key))
                    if numeric is not None:
                        return numeric
        for raw_key in periods.keys():
            numeric = _mc_to_float(values.get(raw_key))
            if numeric is not None:
                return numeric
        return None
    if not isinstance(periods, list) or not isinstance(values, list):
        return None
    labels = [_mc_period_label(p) for p in periods]
    for target in preferred:
        for idx, label in enumerate(labels):
            if label == target and idx < len(values):
                numeric = _mc_to_float(values[idx])
                if numeric is not None:
                    return numeric
    for value in values:
        numeric = _mc_to_float(value)
        if numeric is not None:
            return numeric
    return None


def _mc_extract_metric_value(payload: Any, *keys: str) -> Optional[float]:
    value = _mc_find_first(payload, *keys)
    numeric = _mc_to_float(value)
    if numeric is not None:
        return numeric
    if isinstance(payload, dict) and isinstance(value, (list, dict)):
        period_list = _mc_find_first(payload, "period", "tenure", "duration")
        numeric = _mc_pick_period_value(period_list, value)
        if numeric is not None:
            return numeric
    return None


def _mc_extract_period_returns(payload: Any) -> tuple[dict, dict]:
    """Extract fund and benchmark/category returns from varied Moneycontrol shapes."""
    fund = {"1Y": None, "3Y": None, "5Y": None, "10Y": None}
    bench = {"1Y": None, "3Y": None, "5Y": None, "10Y": None}

    if isinstance(payload, dict):
        periods = _mc_find_first(payload, "period", "tenure", "duration")
        direct_returns = _mc_find_first(payload, "returns", "return", "fund_return", "fund returns")
        benchmark_direct = _mc_find_first(payload, "benchmark", "benchmark_return", "benchmark returns", "category_return", "category returns")
        if isinstance(periods, dict) and isinstance(direct_returns, dict):
            normalized_periods = {str(key).lower(): _mc_period_label(val) for key, val in periods.items()}
            for raw_key, label in normalized_periods.items():
                if label and fund[label] is None:
                    fund[label] = _mc_to_float(direct_returns.get(raw_key))
        if isinstance(periods, dict) and isinstance(benchmark_direct, dict):
            normalized_periods = {str(key).lower(): _mc_period_label(val) for key, val in periods.items()}
            for raw_key, label in normalized_periods.items():
                if label and bench[label] is None:
                    bench[label] = _mc_to_float(benchmark_direct.get(raw_key))
        if isinstance(periods, list) and isinstance(direct_returns, list):
            for period, value in zip(periods, direct_returns):
                label = _mc_period_label(period)
                if label and fund[label] is None:
                    fund[label] = _mc_to_float(value)
        if isinstance(periods, list) and isinstance(benchmark_direct, list):
            for period, value in zip(periods, benchmark_direct):
                label = _mc_period_label(period)
                if label and bench[label] is None:
                    bench[label] = _mc_to_float(value)

    def _assign(label: Optional[str], node: Any):
        if not label:
            return
        if isinstance(node, dict):
            if fund[label] is None:
                fund[label] = _mc_to_float(_mc_find_first(
                    node, "fund", "scheme", "return", "value", "returns", "annualised_return",
                    "annualized_return", "annualised returns", "annualized returns",
                    "fund_return", "fund return", "direct_return", "direct return"
                ))
            if bench[label] is None:
                bench[label] = _mc_to_float(_mc_find_first(
                    node, "benchmark", "benchmark_return", "benchmark_returns",
                    "benchmark return", "category", "category_return", "category_returns",
                    "category return", "benchmarkvalue", "benchmark value", "catavg", "cat_avg"
                ))
        else:
            if fund[label] is None:
                fund[label] = _mc_to_float(node)

    def _walk(node: Any):
        if isinstance(node, dict):
            period = _mc_period_label(_mc_find_first(node, "period", "tenure", "duration", "label", "name", "periodinvested", "period invested"))
            if period:
                _assign(period, node)
            for key, value in node.items():
                label = _mc_period_label(key)
                if label:
                    _assign(label, value)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return fund, bench


def _mc_extract_risk(mc_risk: Any, fallback_benchmark: Optional[str]) -> dict:
    return {
        "sharpe": _mc_extract_metric_value(mc_risk, "sharpe_ratio", "sharpe", "sharpe ratio"),
        "sortino": _mc_extract_metric_value(mc_risk, "sortino_ratio", "sortino", "sortino ratio"),
        "volatility": _mc_extract_metric_value(
            mc_risk, "risk_std_dev", "std_dev", "standard_deviation", "standard deviation",
            "std deviation", "volatility"
        ),
        "beta": _mc_extract_metric_value(mc_risk, "beta"),
        "alpha": _mc_extract_metric_value(mc_risk, "alpha", "jensens_alpha", "jensen alpha", "jension alpha"),
        "max_drawdown_pct": _mc_extract_metric_value(
            mc_risk, "max_drawdown_pct", "max_drawdown", "drawdown", "max drawdown"
        ),
        "benchmark_name": _mc_find_first(mc_risk, "benchmark_name", "benchmark", "benchmarklabel") or fallback_benchmark,
    }


def _mc_extract_fundamentals(mc_fund: Any) -> dict:
    return {
        "aum_cr": _mc_to_float(_mc_find_first(mc_fund, "aum_cr", "aum", "assets_under_management")),
        "expense_ratio": _mc_to_float(_mc_find_first(mc_fund, "expense_ratio", "exp_ratio", "expense")),
        "exit_load": _mc_find_first(mc_fund, "exit_load", "exitload"),
        "portfolio_turnover": _mc_to_float(_mc_find_first(mc_fund, "portfolio_turnover_pct", "portfolio_turnover", "turnover_ratio", "turnover")),
    }


def _mc_extract_holdings(mc_portfolio: Any) -> list[dict]:
    holdings: list[dict] = []

    def _walk(node: Any):
        if isinstance(node, dict):
            name = _mc_find_first(node, "asset", "company", "holding", "security", "stock", "instrument", "name")
            weight = _mc_to_float(_mc_find_first(node, "weight", "holding_pct", "holdingpercent", "percent", "percentage", "value"))
            if isinstance(name, str) and weight is not None:
                holdings.append({"asset": name, "weight": round(weight, 2)})
            else:
                for value in node.values():
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(mc_portfolio)

    deduped: dict[str, float] = {}
    for item in holdings:
        asset = item["asset"].strip()
        if not asset:
            continue
        deduped.setdefault(asset, item["weight"])

    return [
        {"asset": asset, "weight": weight}
        for asset, weight in sorted(deduped.items(), key=lambda pair: pair[1], reverse=True)
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
