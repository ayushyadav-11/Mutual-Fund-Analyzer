"""
app.py — FastAPI backend for Mutual Fund Portfolio Analyzer
Run: uvicorn app:app --reload --port 8000
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional
from datetime import date

import asyncio
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.parser import parse_cas, save_session, load_session, recompute_xirr
from core.risk import compute_risk_metrics
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

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session file location (persists between requests)
SESSION_FILE = os.path.join(os.path.dirname(__file__), "session_data.json")


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
        amc = guess_amc(fn)
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

    await asyncio.gather(*(_fetch(h) for h in equity_holdings))

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
    """Fuses Ground-Truth SQL math with Morningstar Institutional Portfolios for individual analytics."""
    import sqlite3
    from data.database import get_nav_series, DB_PATH
    from core.rolling_returns import compute_rolling_returns
    from core.risk import (compute_sharpe, compute_volatility, compute_sortino,
                      compute_beta, compute_alpha, compute_max_drawdown,
                      _nav_series_to_monthly_returns, _fetch_benchmark_returns,
                      _ticker_to_label)
    from scrapers.morningstar import MorningstarScraper
    from core.parser import get_benchmark_ticker

    # ── 1. Base Scheme from DB ─────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT scheme_name, category, benchmark FROM schemes WHERE isin = ?", (isin,))
    scheme = c.fetchone()
    conn.close()

    if not scheme:
        raise HTTPException(status_code=404, detail="Fund ISIN not indexed in Database.")

    scheme_name = scheme["scheme_name"]
    category    = scheme["category"]
    benchmark_ticker = scheme["benchmark"] or get_benchmark_ticker(scheme_name, category)

    # ── 2. NAV Series + Rolling Returns ───────────────────────────────────────
    navs   = get_nav_series(isin)
    rolling = compute_rolling_returns(navs)

    # ── 3. Risk Metrics from NAV history ──────────────────────────────────────
    risk_data: dict = {
        "sharpe": None, "sortino": None, "volatility": None,
        "beta": None, "alpha": None, "max_drawdown_pct": None,
        "benchmark_name": _ticker_to_label(benchmark_ticker),
    }
    fund_returns = None
    if navs:
        monthly_returns = _nav_series_to_monthly_returns(navs)
        if monthly_returns is not None:
            monthly_returns = monthly_returns.tail(36)  # strict 3-year window
            fund_returns = monthly_returns
            from datetime import date, timedelta
            from scrapers.morningstar import get_rbi_repo_rate
            
            end_date   = date.today().strftime("%Y-%m-%d")
            start_date = (date.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
            
            rfr_monthly = (1 + (get_rbi_repo_rate() / 100.0)) ** (1 / 12) - 1
            
            risk_data["volatility"]       = compute_volatility(monthly_returns)
            risk_data["sharpe"]           = compute_sharpe(monthly_returns, rfr_monthly)
            risk_data["sortino"]          = compute_sortino(monthly_returns, rfr_monthly)
            risk_data["max_drawdown_pct"] = compute_max_drawdown(monthly_returns)

            if benchmark_ticker:
                bench_r = await asyncio.get_event_loop().run_in_executor(
                    None, _fetch_benchmark_returns, benchmark_ticker, start_date, end_date
                )
                risk_data["beta"]  = compute_beta(monthly_returns, bench_r)
                risk_data["alpha"] = compute_alpha(monthly_returns, bench_r, rfr_monthly)

    # ── 4. mfapi_data Initialisation ──────────────────────────────────────────
    mfapi_data = {"aum_cr": None, "expense_ratio": None, "exit_load": None,
                  "current_nav": None, "nav_date": None}

    # ── 5. Morningstar & MoneyControl: API Fetching ───────────────────────────
    ms = MorningstarScraper()
    from scrapers.moneycontrol import MoneyControlScraper
    mc = MoneyControlScraper()
    
    # Run API calls concurrently via threads
    loop = asyncio.get_event_loop()
    ms_fund, mc_risk, mc_perf, mc_fund = await asyncio.gather(
        loop.run_in_executor(None, ms.search_fund, scheme_name),
        loop.run_in_executor(None, mc.get_risk_metrics, isin),
        loop.run_in_executor(None, mc.get_performance, isin),
        loop.run_in_executor(None, mc.get_fundamentals, isin)
    )
    ms_fund = await asyncio.get_event_loop().run_in_executor(None, ms.search_fund, scheme_name)

    ms_portfolio: dict = {}
    sector_allocation: list = []
    portfolio_turnover: float | None = None

    if ms_fund:
        ms_id = ms_fund["id"]
        # Holdings
        raw = await asyncio.get_event_loop().run_in_executor(None, ms.get_portfolio, ms_id)
        ms_portfolio = {k: round(v * 100, 2) for k, v in raw.items()}
        # Sector, Turnover, AUM, Expense
        fund_info = await asyncio.get_event_loop().run_in_executor(None, ms.get_fund_info, ms_id)
        sector_allocation = fund_info.get("sector_allocation", [])
        portfolio_turnover = fund_info.get("portfolio_turnover_pct")
        mfapi_data["aum_cr"] = fund_info.get("aum_cr")
        mfapi_data["expense_ratio"] = fund_info.get("expense_ratio")
        
    # Override with MoneyControl fundamentals if available
    if mc_fund:
        pass # MC fundamentals parsing will go here if MS missing

    sorted_holdings = [
        {"asset": k, "weight": v}
        for k, v in sorted(ms_portfolio.items(), key=lambda i: i[1], reverse=True)
    ]

    # ── 6. mfapi.in: Current NAV using stored scheme_code (fast, no master list download) ───
    try:
        conn2 = sqlite3.connect(DB_PATH)
        conn2.row_factory = sqlite3.Row
        c2 = conn2.cursor()
        c2.execute("SELECT scheme_code FROM schemes WHERE isin = ?", (isin,))
        sc_row = c2.fetchone()
        conn2.close()
        scheme_code_db = sc_row["scheme_code"] if sc_row else None

        if scheme_code_db:
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

            detail = await asyncio.get_event_loop().run_in_executor(None, _fetch_nav, scheme_code_db)
            nav_data = detail.get("data", [])
            if nav_data:
                mfapi_data["current_nav"] = float(nav_data[0]["nav"])
                mfapi_data["nav_date"]    = nav_data[0]["date"]
                # If local DB has no NAVs, rebuild from mfapi history
                if not navs and len(nav_data) > 10:
                    import datetime
                    rebuilt = []
                    for row in nav_data:
                        try:
                            d = datetime.datetime.strptime(row["date"], "%d-%m-%Y").strftime("%Y-%m-%d")
                            rebuilt.append({"date": d, "nav": float(row["nav"])})
                        except (ValueError, KeyError):
                            continue
                    navs = sorted(rebuilt, key=lambda x: x["date"])
                    rolling = compute_rolling_returns(navs)
    except Exception as _nav_err:
        import logging as _log
        _log.getLogger("app").warning("NAV fetch failed for %s: %s", isin, _nav_err)

    # ── 5b. Re-compute risk metrics if navs were rebuilt from mfapi ───────────
    if navs and fund_returns is None:
        monthly_returns = _nav_series_to_monthly_returns(navs)
        if monthly_returns is not None:
            monthly_returns = monthly_returns.tail(36)
            fund_returns = monthly_returns
            from datetime import date as _date, timedelta
            from scrapers.morningstar import get_rbi_repo_rate as _get_rfr
            end_date   = _date.today().strftime("%Y-%m-%d")
            start_date = (_date.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
            _rfr_monthly = (1 + (_get_rfr() / 100.0)) ** (1 / 12) - 1
            risk_data["volatility"]       = compute_volatility(monthly_returns)
            risk_data["sharpe"]           = compute_sharpe(monthly_returns, _rfr_monthly)
            risk_data["sortino"]          = compute_sortino(monthly_returns, _rfr_monthly)
            risk_data["max_drawdown_pct"] = compute_max_drawdown(monthly_returns)
            if benchmark_ticker:
                bench_r = await asyncio.get_event_loop().run_in_executor(
                    None, _fetch_benchmark_returns, benchmark_ticker, start_date, end_date
                )
                risk_data["beta"]  = compute_beta(monthly_returns, bench_r)
                risk_data["alpha"] = compute_alpha(monthly_returns, bench_r, _rfr_monthly)

    # ── 5c. Benchmark CAGR for 1Y / 3Y / 5Y / 10Y ────────────────────────────
    benchmark_returns_cagr: dict = {"1Y": None, "3Y": None, "5Y": None, "10Y": None}
    if benchmark_ticker:
        try:
            from datetime import date as _date, timedelta
            import yfinance as yf, pandas as pd
            end = _date.today()
            start_10y = end - timedelta(days=10 * 365 + 5)
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yf.download(benchmark_ticker, start=start_10y.strftime("%Y-%m-%d"),
                                     end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            )
            if not df.empty:
                price = df["Close"].squeeze()
                latest_p = float(price.iloc[-1])
                for label, years in {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}.items():
                    target = pd.Timestamp(end) - pd.DateOffset(years=years)
                    sub = price[price.index >= target]
                    if not sub.empty:
                        past_p = float(sub.iloc[0])
                        if past_p > 0:
                            if years == 1:
                                benchmark_returns_cagr[label] = round((latest_p / past_p - 1) * 100, 2)
                            else:
                                benchmark_returns_cagr[label] = round(((latest_p / past_p) ** (1 / years) - 1) * 100, 2)
        except Exception:
            pass  # non-critical

    # ── Override LOCAL calculations with MONEYCONTROL if available ───────────
    if mc_risk:
        # MC gives risk_std_dev, sharpe_ratio, sortino_ratio, beta
        if "sharpe_ratio" in mc_risk:
            try: risk_data["sharpe"] = float(mc_risk["sharpe_ratio"])
            except: pass
        if "sortino_ratio" in mc_risk:
            try: risk_data["sortino"] = float(mc_risk["sortino_ratio"])
            except: pass
        if "beta" in mc_risk:
            try: risk_data["beta"] = float(mc_risk["beta"])
            except: pass
        if "risk_std_dev" in mc_risk:
            try: risk_data["volatility"] = float(mc_risk["risk_std_dev"])
            except: pass
            
    fund_trailing_returns = {"1Y": None, "3Y": None, "5Y": None}
    if mc_risk and "returns" in mc_risk:
        rts = mc_risk["returns"]
        for p in ["1y", "3y", "5y"]:
            try: fund_trailing_returns[p.upper()] = float(rts[p])
            except: pass

    # ── 6. SIP vs Benchmark performance ───────────────────────────────────────
    sip_chart: dict = {}
    try:
        data = _load_or_404()
        holding = next((h for h in data["holdings"] if h.get("isin") == isin), None)
        if holding and navs and benchmark_ticker and fund_returns is not None:
            import pandas as pd
            from datetime import date, timedelta

            nav_df = pd.DataFrame(navs)
            nav_df["date"] = pd.to_datetime(nav_df["date"])
            nav_df = nav_df.set_index("date").sort_index()

            sip_txns = [t for t in holding["transactions"]
                        if t["type"] in ("SIP", "BUY") and t.get("nav") and t.get("amount")]
            sip_txns.sort(key=lambda x: x["date"])

            if sip_txns:
                start_str = sip_txns[0]["date"]
                end_str   = date.today().strftime("%Y-%m-%d")
                bench_end = date.today().strftime("%Y-%m-%d")
                bench_s   = await asyncio.get_event_loop().run_in_executor(
                    None, _fetch_benchmark_returns, benchmark_ticker,
                    (date.today() - timedelta(days=10 * 365)).strftime("%Y-%m-%d"), bench_end
                )

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
                        "benchmark_name": _ticker_to_label(benchmark_ticker) or benchmark_ticker
                    }
    except Exception:
        pass  # sip_chart stays empty — non-critical

    return {
        "isin":               isin,
        "name":               scheme_name,
        "category":           category,
        "benchmark":          _ticker_to_label(benchmark_ticker) or benchmark_ticker,
        "aum_cr":             mfapi_data["aum_cr"],
        "expense_ratio":      mfapi_data["expense_ratio"],
        "exit_load":          mfapi_data["exit_load"],
        "current_nav":        mfapi_data["current_nav"],
        "nav_date":           mfapi_data["nav_date"],
        "portfolio_turnover": portfolio_turnover,
        "risk":               risk_data,
        "returns":            rolling,                  # DB-based rolling XIRR math
        "fund_trailing":      fund_trailing_returns,    # MC Trailing Returns (1Y/3Y/5Y)
        "benchmark_cagr":     benchmark_returns_cagr,   # DB yfinance fallback
        "sector_allocation":  sector_allocation,
        "holdings":           sorted_holdings[:20],
        "sip_vs_benchmark":   sip_chart,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
