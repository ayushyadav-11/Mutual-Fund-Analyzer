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
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Request, BackgroundTasks
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.parser import parse_cas, save_session, load_session, recompute_xirr
from core.risk import compute_risk_metrics
from data.database import initialize_database
from data.data_collector import fetch_and_populate_mfapi_data, prefetch_deep_dive_for_user
from core.portfolio_overlap import compute_overlap
from core.advanced_analytics import calculate_taxes_and_loads, calculate_goal_strategy, calculate_sip_step_up, run_monte_carlo_simulation, calculate_stress_test, calculate_rebalance, calculate_dividend_cashflow


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

# ── Serve frontend HTML pages ─────────────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).parent / "frontend"

async def _html(filename: str) -> FileResponse:
    return FileResponse(str(_FRONTEND_DIR / filename))

# Clean URLs (/dashboard) AND legacy .html URLs (/dashboard.html) both work
@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_index():
    return await _html("index.html")

@app.get("/dashboard", include_in_schema=False)
@app.get("/dashboard.html", include_in_schema=False)
async def serve_dashboard():
    return await _html("dashboard.html")

@app.get("/family", include_in_schema=False)
@app.get("/family.html", include_in_schema=False)
async def serve_family():
    return await _html("family.html")

@app.get("/reset-password", include_in_schema=False)
@app.get("/reset-password.html", include_in_schema=False)
async def serve_reset_password():
    return await _html("reset-password.html")

# Serve CSS and any other static frontend assets at the root level
@app.get("/style.css", include_in_schema=False)
async def serve_css():
    return FileResponse(str(_FRONTEND_DIR / "style.css"), media_type="text/css")

# Mount full frontend dir for any other static assets
app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")


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
            "/", "/health", "/favicon.ico",
            "/style.css",
            "/index.html", "/dashboard.html", "/family.html", "/reset-password.html",
            "/dashboard", "/family", "/reset-password",
            "/api/config",
        ] or request.url.path.startswith("/frontend/"):
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
    
    # Track these ISINs in the database for the nightly global prefetch
    from data.database import sync_user_holdings_to_tracker
    holdings_list = data.get("holdings", [])
    sync_user_holdings_to_tracker(user_id, holdings_list)
    
    asyncio.create_task(prefetch_deep_dive_for_user(user_id, holdings_list))

    return {
        "status": "ok",
        "investor": data["investor_info"]["name"],
        "period": data["statement_period"],
        "funds": len(data["holdings"]),
        "transactions": len(data.get("all_transactions", data.get("transactions", []))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/progress
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/progress")
def get_parse_progress(request: Request):
    user_id = request.state.user_id
    try:
        from core.parser import load_session
        data = load_session(user_id)
        if not data:
            return {"total": 0, "cached": 0, "status": "complete"}
    except (FileNotFoundError, Exception):
        return {"total": 0, "cached": 0, "status": "complete"}

    holdings = data.get("holdings", [])
    valid_funds = [h for h in holdings if h.get("units", 0) > 0.001 and h.get("isin")]
    total = len(valid_funds)

    from data.database import count_user_funds_cached
    cached = count_user_funds_cached(user_id)

    return {
        "total": total,
        "cached": cached,
        "status": "complete" if cached >= total else "processing"
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
# Background prefetch: silently warms Redis for all holding ISINs on dashboard load
# ─────────────────────────────────────────────────────────────────────────────
# isin -> asyncio.Event: set when prefetch completes, so foreground requests can wait instead of duplicate-scraping
_prefetch_events: dict[str, asyncio.Event] = {}

async def _prefetch_one_fund(isin: str, scheme_name: str, scheme: dict):
    """Scrape one fund and store full response in Redis + Supabase. Skipped if already cached."""
    from data.cache import get_cached, set_cached
    from data.database import get_cached_fund_deep_dive, cache_fund_deep_dive, insert_or_update_scheme
    from scrapers.moneycontrol import MoneyControlScraper
    from scrapers.morningstar import MorningstarScraper

    redis_key = f"fund_detail:{isin}"

    # Already in Redis? Nothing to do.
    if get_cached(redis_key):
        return

    # Already in Supabase deep-dive cache? Build minimal response and push to Redis.
    cached = get_cached_fund_deep_dive(isin, max_age_hours=24)
    if cached:
        logger.info("Prefetch: Supabase hit for %s, warming Redis", isin)
        _mini = {
            "isin": isin,
            "name": scheme_name,
            "category": scheme.get("category", ""),
            "benchmark": scheme.get("benchmark", ""),
            "fundamentals": cached.get("fundamentals", {}),
            "risk": cached.get("risk", {}),
            "returns": cached.get("returns", {}),
            "fund_trailing": cached.get("returns", {}),
            "benchmark_cagr": cached.get("benchmark_cagr", {}),
            "holdings": cached.get("holdings", []),
            "sector_allocation": cached.get("sectors", []),
            "aum_cr": cached.get("fundamentals", {}).get("aum_cr"),
            "expense_ratio": cached.get("fundamentals", {}).get("expense_ratio"),
            "exit_load": cached.get("fundamentals", {}).get("exit_load"),
            "portfolio_turnover": cached.get("fundamentals", {}).get("portfolio_turnover"),
            "current_nav": None, "nav_date": None,
            "performance_annualised": [], "performance_yearly": [], "performance_sip": [],
            "sip_vs_benchmark": None, "xirr": None,
        }
        set_cached(redis_key, _mini, 3600)
        return

    # Live scrape as last resort
    logger.info("Prefetch: live scraping %s (%s)", isin, scheme_name)
    try:
        # Use ONE shared Morningstar client for all MS calls — avoids double token refresh
        async with MoneyControlScraper() as mc, MorningstarScraper() as ms:
            ms_fund, mc_risk, mc_perf, mc_fund, mc_overview = await asyncio.gather(
                ms.search_fund(scheme_name),
                mc.get_risk_metrics(isin),
                mc.get_performance(isin),
                mc.get_fundamentals(isin),
                mc.get_overview(isin),
            )
        if mc_overview:
            mc_fund = {**mc_overview, **(mc_fund or {})}

        returns_data, benchmark_returns = _mc_extract_period_returns(mc_perf)
        risk_data = _mc_extract_risk(mc_risk, scheme.get("benchmark", ""))
        mc_fundamentals = _mc_extract_fundamentals(mc_fund)

        fundms = {
            "aum_cr": mc_fundamentals.get("aum_cr"),
            "expense_ratio": mc_fundamentals.get("expense_ratio"),
            "exit_load": mc_fundamentals.get("exit_load"),
            "portfolio_turnover": mc_fundamentals.get("portfolio_turnover"),
            "pe": mc_fundamentals.get("pe"), "cat_avg_pe": mc_fundamentals.get("cat_avg_pe"),
            "pb": mc_fundamentals.get("pb"), "cat_avg_pb": mc_fundamentals.get("cat_avg_pb"),
            "price_sale": mc_fundamentals.get("price_sale"),
            "cat_avg_price_sale": mc_fundamentals.get("cat_avg_price_sale"),
            "price_cash_flow": mc_fundamentals.get("price_cash_flow"),
            "cat_avg_price_cash_flow": mc_fundamentals.get("cat_avg_price_cash_flow"),
            "dividend_yield": mc_fundamentals.get("dividend_yield"),
            "cat_avg_dividend_yield": mc_fundamentals.get("cat_avg_dividend_yield"),
            "roe": mc_fundamentals.get("roe"), "cat_avg_roe": mc_fundamentals.get("cat_avg_roe"),
        }

        holdings_list, sector_allocation = [], []
        if ms_fund:
            try:
                # Reuse the SAME ms scraper instance — token already loaded, no second refresh
                async with MorningstarScraper() as ms_portfolio:
                    raw_portfolio, fund_info = await asyncio.gather(
                        ms_portfolio.get_portfolio(ms_fund["id"]),
                        ms_portfolio.get_fund_info(ms_fund["id"]),
                    )
                holdings_list = [
                    {"asset": k, "weight": round(v * 100, 2)}
                    for k, v in sorted(raw_portfolio.items(), key=lambda x: x[1], reverse=True)
                ][:20]
                sector_allocation = fund_info.get("sector_allocation", [])
                if not mc_fundamentals.get("aum_cr"):
                    fundms["aum_cr"] = fund_info.get("aum_cr")
                if not mc_fundamentals.get("expense_ratio"):
                    fundms["expense_ratio"] = fund_info.get("expense_ratio")
            except Exception as _e:
                logger.debug("Prefetch MS portfolio failed for %s: %s", isin, _e)

        await asyncio.get_event_loop().run_in_executor(
            None, insert_or_update_scheme, isin, scheme_name
        )
        cache_fund_deep_dive(
            isin=isin, fundamentals=fundms, risk=risk_data,
            returns=returns_data, bench_returns=benchmark_returns,
            holdings=holdings_list, sectors=sector_allocation,
        )

        payload = {
            "isin": isin, "name": scheme_name,
            "category": scheme.get("category", ""),
            "benchmark": scheme.get("benchmark", ""),
            "fundamentals": fundms, "risk": risk_data,
            "returns": returns_data, "fund_trailing": returns_data,
            "benchmark_cagr": benchmark_returns,
            "holdings": holdings_list, "sector_allocation": sector_allocation,
            "aum_cr": fundms.get("aum_cr"),
            "expense_ratio": fundms.get("expense_ratio"),
            "exit_load": fundms.get("exit_load"),
            "portfolio_turnover": fundms.get("portfolio_turnover"),
            "current_nav": mc_overview.get("latest_nav") if mc_overview else None,
            "nav_date": mc_overview.get("nav_date") if mc_overview else None,
            "performance_annualised": [], "performance_yearly": [], "performance_sip": [],
            "sip_vs_benchmark": None, "xirr": None,
        }
        set_cached(redis_key, payload, 3600)
        logger.info("Prefetch: cached %s in Redis+Supabase", isin)
    except Exception as e:
        logger.warning("Prefetch failed for %s: %s", isin, e)


async def _background_prefetch_holdings(holdings: list):
    """Prefetch all holding ISINs in the background with max 3 concurrent scrapes."""
    from data.database import get_connection
    sem = asyncio.Semaphore(2)  # max 2 concurrent background scrapes — leaves bandwidth for foreground

    # Resolve scheme info for all ISINs in one query
    isins = [h["isin"] for h in holdings if h.get("isin")]
    if not isins:
        return

    try:
        conn = get_connection()
        c = conn.cursor()
        placeholders = ", ".join(["%s"] * len(isins))
        c.execute(
            f"SELECT isin, scheme_name, category, benchmark FROM schemes WHERE isin IN ({placeholders})",
            isins,
        )
        rows = {r["isin"]: r for r in c.fetchall()}
        conn.close()
    except Exception as e:
        logger.warning("Prefetch: scheme lookup failed: %s", e)
        return

    async def _fetch_with_sem(isin, holding):
        if isin in _prefetch_events:  # already being fetched
            return
        scheme = rows.get(isin)
        if not scheme:
            return
        ev = asyncio.Event()
        _prefetch_events[isin] = ev
        try:
            async with sem:
                await _prefetch_one_fund(isin, scheme["scheme_name"], dict(scheme))
        finally:
            ev.set()                       # unblock any foreground waiter
            _prefetch_events.pop(isin, None)

    tasks = [_fetch_with_sem(h["isin"], h) for h in holdings if h.get("isin")]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Background prefetch complete for %d ISINs", len(tasks))


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/holdings
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/holdings")
async def get_holdings(request: Request, background_tasks: BackgroundTasks):
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

    # Fire-and-forget: pre-warm Redis for all holdings in background
    # so fund modals are instant by the time the user clicks one
    background_tasks.add_task(_background_prefetch_holdings, data["holdings"])

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

    # Fetch holdings using async Morningstar scraper
    from scrapers.morningstar import MorningstarScraper as _MS
    raw_holdings_map: dict[str, dict] = {}
    mstar_id_map: dict[str, str] = {}  # fund_name -> mstar_id

    async with _MS() as ms:
        for h in equity_holdings:
            fn = h["name"]
            try:
                mstar_fund = await ms.search_fund(fn)
                if mstar_fund:
                    mid = mstar_fund['id']
                    mstar_id_map[fn] = mid
                    result = await ms.get_portfolio(mid)
                    raw_holdings_map[fn] = result
                else:
                    raw_holdings_map[fn] = {}
            except Exception as exc:
                logger.warning("Failed to get portfolio for %s: %s", fn, exc)
                raw_holdings_map[fn] = {}
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
async def get_fund_details(request: Request, isin: str, refresh: bool = False):
    """Return fund details using Moneycontrol for metrics and Morningstar for portfolio composition."""
    from data.database import get_nav_series, DB_PATH, get_cached_fund_deep_dive, cache_fund_deep_dive, get_connection
    from data.cache import get_cached, set_cached, delete_cached

    # ── 0. Redis full-response cache (TTL: 1 hour) ─────────────────────────────
    _redis_key = f"fund_detail:{isin}"

    if refresh:
        # Hard refresh: delete Redis + Supabase cache so live scrape runs
        logger.info("Force refresh requested for %s — busting caches", isin)
        await asyncio.get_event_loop().run_in_executor(None, delete_cached, _redis_key)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: __import__('data.database', fromlist=['_delete_deep_dive_cache'])._delete_deep_dive_cache(isin)
            if hasattr(__import__('data.database', fromlist=['_delete_deep_dive_cache']), '_delete_deep_dive_cache')
            else None
        )
    elif not DEBUG_BYPASS_DEEP_DIVE_CACHE:
        _redis_hit = await asyncio.get_event_loop().run_in_executor(None, get_cached, _redis_key)
        if _redis_hit:
            logger.info("Redis cache hit for %s", isin)
            return _redis_hit

    # ── 0b. If background prefetch is already running for this ISIN, wait for it ──
    # This prevents a duplicate live scrape when the user clicks a fund that is
    # currently being prefetched in the background.
    if isin in _prefetch_events:
        logger.info("Foreground waiting on background prefetch for %s", isin)
        try:
            await asyncio.wait_for(_prefetch_events[isin].wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Prefetch wait timed out for %s, falling through to live scrape", isin)
        # Check Redis again — prefetch should have populated it
        _redis_hit = await asyncio.get_event_loop().run_in_executor(None, get_cached, _redis_key)
        if _redis_hit:
            logger.info("Redis hit after prefetch wait for %s", isin)
            return _redis_hit

    # ── 1. Fetch scheme + deep-dive cache + session data ALL IN PARALLEL ─────────
    _scheme_key = f"scheme:{isin}"

    def _fetch_scheme_from_db():
        _s = get_cached(_scheme_key)
        if _s:
            return _s
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT scheme_name, category, benchmark, scheme_code FROM schemes WHERE isin = ?", (isin,))
        row = c.fetchone()
        conn.close()
        if row:
            d = dict(row)
            set_cached(_scheme_key, d, 3600)
            return d
        return None

    def _fetch_deep_dive():
        if DEBUG_BYPASS_DEEP_DIVE_CACHE:
            return None
        return get_cached_fund_deep_dive(isin, max_age_hours=24)

    loop = asyncio.get_event_loop()
    scheme, cached_fund = await asyncio.gather(
        loop.run_in_executor(None, _fetch_scheme_from_db),
        loop.run_in_executor(None, _fetch_deep_dive),
    )

    # ── Resolve scheme (fallback to session data if not in DB) ───────────────
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
            from core.parser import get_benchmark_ticker
            name = active_holding.get("name", "Unknown Fund")
            cat = active_holding.get("category", "")
            scheme = {
                "scheme_name": name,
                "category": cat,
                "benchmark": get_benchmark_ticker(name, cat),
                "scheme_code": None,
            }
        else:
            raise HTTPException(status_code=404, detail="Fund ISIN not indexed in Database.")

    scheme_name = scheme["scheme_name"]
    category    = scheme["category"]
    scheme_code = scheme["scheme_code"]
    benchmark_symbol = scheme["benchmark"]

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
        "CRSLDX":        "Nifty 500",
        "NSMIDCP":       "Nifty Midcap 150",
        "^NIFTY_MID_SELECT": "Nifty Midcap Select",
    }
    readable_benchmark = _TICKER_TO_NAME.get(benchmark_symbol, benchmark_symbol)

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

        # Still fetch performance tables from MC — they're lightweight and MC-cached
        # so they are fast even when serving from the DB cache. Without this, the
        # Performance (Lumpsum / Yearly / SIP) tabs show "No data available" after
        # a server restart clears the in-memory Redis cache.
        from scrapers.moneycontrol import MoneyControlScraper
        try:
            async with MoneyControlScraper() as mc:
                results = await asyncio.gather(
                    mc.get_performance(isin),
                    mc.get_performance_yearly(isin),
                    mc.get_performance_sip(isin),
                    mc.get_overview(isin),
                    return_exceptions=True
                )
            mc_perf = results[0] if not isinstance(results[0], Exception) else None
            mc_perf_yearly = results[1] if not isinstance(results[1], Exception) else None
            mc_perf_sip = results[2] if not isinstance(results[2], Exception) else None
            mc_overview = results[3] if not isinstance(results[3], Exception) else None
            
            if mc_overview and mfapi_data["current_nav"] is None:
                mfapi_data["current_nav"] = mc_overview.get("latest_nav")
                mfapi_data["nav_date"] = mc_overview.get("nav_date")
        except Exception as _perf_err:
            logger.warning("Cached-path MC perf fetch failed for %s: %s", isin, _perf_err)
            mc_perf = mc_perf_yearly = mc_perf_sip = mc_overview = None

    if not cached_fund:
        from scrapers.morningstar import MorningstarScraper
        from scrapers.moneycontrol import MoneyControlScraper
        
        async with MoneyControlScraper() as mc, MorningstarScraper() as ms:
            results = await asyncio.gather(
                ms.search_fund(scheme_name),
                mc.get_risk_metrics(isin),
                mc.get_performance(isin),
                mc.get_performance_yearly(isin),
                mc.get_performance_sip(isin),
                mc.get_fundamentals(isin),
                mc.get_overview(isin),
                return_exceptions=True
            )
        ms_fund = results[0] if not isinstance(results[0], Exception) else None
        mc_risk = results[1] if not isinstance(results[1], Exception) else None
        mc_perf = results[2] if not isinstance(results[2], Exception) else None
        mc_perf_yearly = results[3] if not isinstance(results[3], Exception) else None
        mc_perf_sip = results[4] if not isinstance(results[4], Exception) else None
        mc_fund = results[5] if not isinstance(results[5], Exception) else None
        mc_overview = results[6] if not isinstance(results[6], Exception) else None
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
                async with MorningstarScraper() as ms:
                    raw_portfolio, fund_info = await asyncio.gather(
                        ms.get_portfolio(ms_id),
                        ms.get_fund_info(ms_id),
                    )
                sorted_holdings = [
                    {"asset": asset, "weight": round(weight * 100, 2)}
                    for asset, weight in sorted(raw_portfolio.items(), key=lambda item: item[1], reverse=True)
                ][:20]
                sector_allocation = [
                    {"sector": s.get("sector"), "weight": s.get("pct", 0.0)}
                    for s in (fund_info.get("sector_allocation", []) or [])
                ]
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
            from data.database import insert_or_update_scheme as _insert_or_update_scheme
            # Ensure ISIN exists in schemes table to satisfy Postgres foreign key constraint
            await asyncio.get_event_loop().run_in_executor(
                None, _insert_or_update_scheme, isin, scheme_name
            )
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
    navs = await asyncio.get_event_loop().run_in_executor(None, get_nav_series, isin)
    if navs:
        latest_nav = navs[-1]
        try:
            mfapi_data["current_nav"] = float(latest_nav["nav"])
            mfapi_data["nav_date"] = latest_nav["date"]
        except (TypeError, ValueError, KeyError):
            pass

    try:
        import httpx as _httpx
        import asyncio as _asyncio
        import urllib.parse as _urllib_parse
        from data.database import insert_or_update_scheme as _insert_or_update_scheme

        async def _resolve_scheme_code_for_nav(code, fund_name, fund_isin):
            if code:
                return code

            try:
                query = _urllib_parse.quote((fund_name or "")[:60])
                async with _httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.get(f"https://api.mfapi.in/mf/search?q={query}")
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

        async def _fetch_nav(code):
            for attempt in range(3):
                try:
                    async with _httpx.AsyncClient(timeout=10.0) as client:
                        r = await client.get(f"https://api.mfapi.in/mf/{code}")
                        r.raise_for_status()
                        return r.json()
                except Exception:
                    if attempt < 2:
                        await _asyncio.sleep(1)
            raise Exception(f"mfapi failed for scheme {code} after 3 attempts")

        resolved_scheme_code = await _resolve_scheme_code_for_nav(scheme_code, scheme_name, isin)
        
        if resolved_scheme_code and resolved_scheme_code != scheme_code:
            scheme_code = resolved_scheme_code
            await asyncio.get_event_loop().run_in_executor(
                None, _insert_or_update_scheme, isin, scheme_name, str(scheme_code), scheme.get("category"), scheme.get("benchmark")
            )

        if scheme_code:
            detail = await _fetch_nav(scheme_code)
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
        from datetime import datetime as _dt, date, timedelta
        from core.risk import _fetch_benchmark_returns

        if not navs:
            raise ValueError("No NAV data available; skipping chart")

        # 1. Pure Python monthly NAV extraction (O(N), no pandas overhead)
        monthly_nav_dict = {}
        for row in sorted(navs, key=lambda x: x["date"]):
            try:
                ym = str(row["date"])[:7] # YYYY-MM
                monthly_nav_dict[ym] = float(row["nav"])
            except Exception:
                continue

        if not monthly_nav_dict:
            raise ValueError("Monthly NAV series is empty")

        # 2. Try to fetch benchmark (optional)
        bench_s = None
        bench_monthly_dict = None
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
                    # bench_s is a pandas Series from yfinance. Cumprod is fast.
                    bench_cumprod = (1 + bench_s).cumprod()
                    bench_monthly_dict = {}
                    for date_idx, val in bench_cumprod.items():
                        ym = date_idx.strftime("%Y-%m") if hasattr(date_idx, 'strftime') else str(date_idx)[:7]
                        bench_monthly_dict[ym] = float(val)
                else:
                    _chart_logger.warning("Benchmark %s returned None for %s; rendering fund-only chart", benchmark_symbol, isin)
            except Exception as _be:
                _chart_logger.warning("Benchmark fetch failed for %s / %s: %s", isin, benchmark_symbol, _be)

        # 3. Collect SIP transactions (fund must be in portfolio)
        sip_txns = []
        if holding:
            sip_txns = [t for t in holding.get("transactions", [])
                        if t["type"] in ("SIP", "BUY", "SWITCH_IN") and t.get("nav") and t.get("amount")]
            sip_txns.sort(key=lambda x: str(x["date"]))

        labels, fund_vals, bench_vals = [], [], []
        
        # Take the last 60 months (5 years)
        recent_months = sorted(monthly_nav_dict.keys())[-60:]

        if sip_txns and bench_monthly_dict is not None:
            # ── Path A: actual SIP wealth chart ─────────────────────────────
            fund_units = 0.0
            bench_units = 0.0
            for txn in sip_txns:
                txn_ym = str(txn["date"])[:7]
                amount = abs(txn["amount"])
                fund_units += amount / txn["nav"]
                
                # Match benchmark unit purchase
                bm_val = bench_monthly_dict.get(txn_ym)
                if bm_val:
                    bench_units += amount / (100.0 * bm_val)
                    
            for ym in recent_months:
                nav_val = monthly_nav_dict[ym]
                dt_obj = _dt.strptime(ym, "%Y-%m")
                labels.append(dt_obj.strftime("%b %Y"))
                fund_vals.append(round(fund_units * nav_val, 2))
                
                bm_val = bench_monthly_dict.get(ym)
                bench_vals.append(round(bench_units * 100.0 * bm_val if bm_val else 0, 2))
                
            chart_type = "sip"
        else:
            # ── Path B: ₹10,000 lumpsum NAV growth (benchmark optional) ─────
            start_nav = None
            start_bench = None
            
            for ym in recent_months:
                nav_val = monthly_nav_dict[ym]
                if start_nav is None:
                    start_nav = nav_val
                    if bench_monthly_dict is not None:
                        start_bench = bench_monthly_dict.get(ym)
                        
                dt_obj = _dt.strptime(ym, "%Y-%m")
                labels.append(dt_obj.strftime("%b %Y"))
                fund_vals.append(round(10000 * nav_val / start_nav, 2))
                
                if bench_monthly_dict is not None and start_bench:
                    bm_val = bench_monthly_dict.get(ym)
                    bench_vals.append(round((bm_val / start_bench) * 10000 if bm_val else 0, 2))
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



    response_payload = {
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
            (mc_perf[0].get("lumpsum", {}).get("annualised", []) if isinstance(mc_perf, list) and mc_perf else [])
            if mc_perf else []
        ),
        "performance_yearly":     mc_perf_yearly if mc_perf_yearly else [],
        "performance_sip":        mc_perf_sip if mc_perf_sip else [],
        "sector_allocation":  sector_allocation,
        "holdings":           sorted_holdings,
        "sip_vs_benchmark":   sip_chart,
        "xirr":               holding_xirr,
    }

    # Cache the full response in Redis (1h TTL) — skip if user-specific fields present
    if not DEBUG_BYPASS_DEEP_DIVE_CACHE:
        await asyncio.get_event_loop().run_in_executor(None, set_cached, _redis_key, response_payload, 3600)

    return response_payload

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
@app.get("/api/family/search")
async def search_family(request: Request, q: str = ""):
    from data.database import search_users
    return search_users(q)

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
    from data.database import get_family_members, get_user_by_id, get_connection
    from core.parser import load_session
    
    user_ids = [request.state.user_id] + get_family_members(request.state.user_id)
    portfolios = []
    
    for uid in user_ids:
        try:
            data = load_session(uid)
            user_info = get_user_by_id(uid)
            total_invested = 0.0
            current_value = 0.0
            holdings_list = []
            category_counts = {}
            total_weighted_expense = 0.0
            
            all_holdings = data.get("holdings", [])
            
            # Fetch expense ratios from DB for all ISINs in one batch
            isins = [h.get("isin") for h in all_holdings if h.get("isin")]
            expense_map = {}
            try:
                conn = get_connection()
                c = conn.cursor()
                for isin in isins:
                    c.execute("SELECT data FROM fund_deep_dive WHERE isin = ? ORDER BY cached_at DESC LIMIT 1", (isin,))
                    row = c.fetchone()
                    if row:
                        import json
                        dd = json.loads(row[0] if isinstance(row[0], str) else row[0])
                        er = dd.get("fundamentals", {}).get("expense_ratio")
                        if er is not None:
                            expense_map[isin] = float(er)
                conn.close()
            except Exception:
                pass
            
            for h in all_holdings:
                curr_nav = h.get("live_nav") or h.get("nav")
                h_txns = h.get("transactions")
                if not h_txns:
                    h_txns = [t for t in data.get("transactions", []) if t.get("scheme_name") == h.get("name")]
                
                if not curr_nav:
                    navs = [(t["date"], t["nav"]) for t in h_txns if t.get("nav")]
                    if navs:
                        navs.sort(key=lambda x: x[0], reverse=True)
                        curr_nav = navs[0][1]
                
                val = h.get("units", 0) * (curr_nav if curr_nav else 0.0)
                current_value += val
                
                invested = sum(abs(t["amount"]) for t in h_txns if t["type"] in ("BUY", "SIP", "SWITCH_IN", "DIVR")) - sum(abs(t["amount"]) for t in h_txns if t["type"] in ("SELL", "SWITCH_OUT"))
                total_invested += invested
                
                # Track category distribution
                cat = h.get("category", "Unknown")
                category_counts[cat] = category_counts.get(cat, 0) + (val if val > 0 else 0)
                
                if val > 0:
                    holdings_list.append({"name": h.get("name") or "Unknown Fund", "value": val, "isin": h.get("isin", "")})
                
            # Weighted average expense ratio
            weighted_er_sum = 0.0
            weighted_er_weight = 0.0
            for item in holdings_list:
                er = expense_map.get(item["isin"])
                if er is not None and current_value > 0:
                    wt = item["value"] / current_value
                    weighted_er_sum += er * wt
                    weighted_er_weight += wt
            avg_expense_ratio = round(weighted_er_sum, 2) if weighted_er_weight > 0 else None
            
            # Equity vs Debt vs Hybrid split
            equity_keywords = ["equity", "large cap", "mid cap", "small cap", "flexi", "multi cap", "elss", "value", "focused", "contra", "thematic", "sectoral"]
            debt_keywords = ["debt", "liquid", "money market", "gilt", "duration", "credit", "banking and psu", "corporate bond", "overnight", "ultra short"]
            equity_val = sum(v for cat, v in category_counts.items() if any(kw in cat.lower() for kw in equity_keywords))
            debt_val = sum(v for cat, v in category_counts.items() if any(kw in cat.lower() for kw in debt_keywords))
            hybrid_val = current_value - equity_val - debt_val
            equity_pct = round(equity_val / current_value * 100, 1) if current_value > 0 else 0
            debt_pct = round(debt_val / current_value * 100, 1) if current_value > 0 else 0
            hybrid_pct = max(0, round(100 - equity_pct - debt_pct, 1))
            
            investor_name = data.get("investor_info", {}).get("name")
            fallback_name = investor_name if investor_name else "Unknown"
            
            db_username = user_info.get("username") if user_info else None
            if db_username and db_username.startswith("auth_user_"):
                db_username = None
                
            holdings_list.sort(key=lambda x: x["value"], reverse=True)
            top_holdings = []
            for th in holdings_list[:5]:
                alloc_pct = round((th["value"] / current_value * 100), 2) if current_value > 0 else 0
                top_holdings.append({
                    "name": th["name"],
                    "value": round(th["value"], 2),
                    "allocation": alloc_pct
                })
            
            portfolios.append({
                "user_id": uid,
                "username": db_username if db_username else fallback_name,
                "total_invested": round(total_invested, 2),
                "current_value": round(current_value, 2),
                "xirr": data.get("portfolio_xirr"),
                "num_funds": len(all_holdings),
                "avg_expense_ratio": avg_expense_ratio,
                "equity_pct": equity_pct,
                "debt_pct": debt_pct,
                "hybrid_pct": hybrid_pct,
                "top_holdings": top_holdings
            })
        except Exception:
            pass
            
    return portfolios
