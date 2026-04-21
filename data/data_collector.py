import httpx
import asyncio
import logging
from datetime import datetime
from data.database import insert_or_update_scheme, batch_insert_navs, get_nav_series
from core.parser import get_benchmark_ticker

logger = logging.getLogger(__name__)

MFAPI_SEARCH = "https://api.mfapi.in/mf/search?q={}"
MFAPI_HIST   = "https://api.mfapi.in/mf/{}"


async def _resolve_scheme_code(client: httpx.AsyncClient, isin: str, name: str) -> int | None:
    """
    Search MFAPI by fund name and return the schemeCode whose isinGrowth /
    isinDivReinvestment matches our ISIN.  Falls back to the first result if
    no ISIN match is found (works for most schemes).
    """
    query = name[:40]                     # keep query short but specific
    try:
        r = await client.get(MFAPI_SEARCH.format(httpx.URL(query)), timeout=10.0)
        if r.status_code != 200:
            return None
        results = r.json()
        if not results:
            return None
        for entry in results:
            ig   = entry.get("isinGrowth", "")
            idiv = entry.get("isinDivReinvestment", "")
            if isin in (ig, idiv):
                return entry["schemeCode"]
        # Fallback: first result (usually correct for unique fund names)
        return results[0]["schemeCode"]
    except Exception as e:
        logger.warning(f"MFAPI search failed for '{name}': {e}")
        return None


async def _ingest_fund(client: httpx.AsyncClient, holding: dict, semaphore: asyncio.Semaphore):
    """Fetch + store historical NAV for a single holding."""
    if semaphore:
        async with semaphore:
            await _ingest_fund_logic(client, holding)
    else:
        await _ingest_fund_logic(client, holding)

async def _ingest_fund_logic(client: httpx.AsyncClient, holding: dict):
    isin  = holding.get("isin", "")
    name  = holding.get("name", "Unknown")
    units = holding.get("units", 0)

    if not isin or units < 0.001:
        return

    # Register the scheme in DB
    benchmark = get_benchmark_ticker(name, holding.get("category", ""))
    insert_or_update_scheme(isin=isin, scheme_name=name,
                            category=holding.get("category"),
                            benchmark=benchmark)

    # Skip if we already have fresh NAV data (> 100 rows)
    existing = get_nav_series(isin)
    if existing and len(existing) > 100:
        logger.debug(f"NAV data already present for {name} ({isin}), skipping.")
        return

    code = await _resolve_scheme_code(client, isin, name)
    if not code:
        logger.warning(f"Could not resolve scheme code for {name} ({isin})")
        return

    # Use to_thread to offload blocking wait to prevent uvicorn lock
    await asyncio.to_thread(insert_or_update_scheme, isin, name, scheme_code=code)

    try:
        r = await client.get(MFAPI_HIST.format(code), timeout=20.0)
        if r.status_code != 200:
            return
        raw = r.json().get("data", [])
        nav_records = []
        for point in raw:
            try:
                dt = datetime.strptime(point["date"], "%d-%m-%Y")
                nav_records.append({"date": dt.strftime("%Y-%m-%d"),
                                    "nav": float(point["nav"])})
            except Exception:
                pass

        await asyncio.to_thread(batch_insert_navs, isin, nav_records)
        logger.info(f"Ingested {len(nav_records)} NAV records for {name} ({isin})")
    except Exception as e:
        logger.error(f"NAV ingestion failed for {name}: {e}")


async def fetch_and_populate_mfapi_data(holdings: list):
    """
    Called by /api/risk on startup. Parallelizes resolving scheme codes
    and stores historical NAVs with a safe concurrency limit.
    """
    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = [_ingest_fund(client, holding, sem) for holding in holdings]
        await asyncio.gather(*tasks)


async def prefetch_deep_dive_for_user(user_id: str, holdings: list):
    """
    Background task: pre-fetches MoneyControl + MorningStar deep-dive data
    for all of a user's holdings right after PDF upload. Results are stored in
    the DB cache tables so the first Risk tab click is instant.
    """
    from scrapers.morningstar import MorningstarScraper
    from scrapers.moneycontrol import MoneyControlScraper
    from data.database import (
        cache_fund_deep_dive, mark_user_fund_cached, is_user_fund_cached,
        get_cached_fund_deep_dive, insert_or_update_scheme
    )
    from core.parser import get_benchmark_ticker

    # Limit to 1 concurrent scrape session to prevent ThreadPoolExecutor exhaustion.
    # Otherwise, 5 funds * 7 concurrent blocking functions = 35 threads, which 
    # locks up Uvicorn's default threadpool and delays user API clicks.
    sem = asyncio.Semaphore(1)  
    loop = asyncio.get_event_loop()

    logger.info(f"Pre-fetching deep dive for {len(holdings)} holdings (user: {user_id})")

    async def _prefetch_one(holding: dict):
        isin = holding.get("isin", "")
        name = holding.get("name", "Unknown")
        if not isin:
            return

        async with sem:
            # Add a tiny delay between funds so normal API requests jump the async queue
            await asyncio.sleep(0.5)

            # Skip if already cached within 24h for this user
            if is_user_fund_cached(user_id, isin, max_age_hours=24.0):
                logger.debug(f"Pre-fetch skipped (already cached): {name} ({isin})")
                return

            # Also skip if there is already fresh fund-level cache (<24h old)
            if get_cached_fund_deep_dive(isin, max_age_hours=24.0):
                mark_user_fund_cached(user_id, isin)
                logger.debug(f"Pre-fetch reused existing cache: {name} ({isin})")
                return

            try:
                async with MorningstarScraper() as ms, MoneyControlScraper() as mc:
                    ms_fund, mc_risk, mc_perf, mc_perf_yearly, mc_perf_sip, mc_fund, mc_overview = await asyncio.gather(
                        ms.search_fund(name),
                        mc.get_risk_metrics(isin),
                        mc.get_performance(isin),
                        mc.get_performance_yearly(isin),
                        mc.get_performance_sip(isin),
                        mc.get_fundamentals(isin),
                        mc.get_overview(isin),
                    )

                if mc_overview:
                    mc_fund = {**mc_overview, **(mc_fund or {})}

                # Inline extraction (mirrors the /api/fund/<isin> logic)
                from core.mc_helpers import _mc_extract_period_returns, _mc_extract_risk, _mc_extract_fundamentals, _mc_find_first

                returns_data, benchmark_returns = _mc_extract_period_returns(mc_perf)
                risk_returns, fallback_bm = _mc_extract_period_returns(mc_risk)
                for k in returns_data:
                    if returns_data.get(k) in (0.0, None) and risk_returns.get(k) not in (0.0, None):
                        returns_data[k] = risk_returns[k]
                for k in benchmark_returns:
                    if benchmark_returns.get(k) in (0.0, None) and fallback_bm.get(k) not in (0.0, None):
                        benchmark_returns[k] = fallback_bm[k]

                bm_name = (
                    _mc_find_first(mc_perf, "benchmark_name", "benchmark", "benchmarklabel")
                    or _mc_find_first(mc_risk, "benchmark_name", "benchmark", "benchmarklabel")
                    or get_benchmark_ticker(name, holding.get("category", ""))
                )
                risk_data = _mc_extract_risk(mc_risk, bm_name)
                mc_fundamentals = _mc_extract_fundamentals(mc_fund)

                sorted_holdings: list = []
                sector_allocation: list = []
                portfolio_turnover = mc_fundamentals.get("portfolio_turnover")

                if ms_fund:
                    try:
                        async with MorningstarScraper() as ms:
                            raw_portfolio, fund_info = await asyncio.gather(
                                ms.get_portfolio(ms_id),
                                ms.get_fund_info(ms_id),
                            )
                        sorted_holdings = [
                            {"asset": asset, "weight": round(weight * 100, 2)}
                            for asset, weight in sorted(raw_portfolio.items(), key=lambda x: x[1], reverse=True)
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

                fundms = {**mc_fundamentals, "portfolio_turnover": portfolio_turnover}

                # Ensure the ISIN exists in the master schemes table first to satisfy FOREIGN KEY constraint
                insert_or_update_scheme(isin=isin, scheme_name=name)

                cache_fund_deep_dive(
                    isin=isin,
                    fundamentals=fundms,
                    risk=risk_data,
                    returns=returns_data,
                    bench_returns=benchmark_returns,
                    holdings=sorted_holdings,
                    sectors=sector_allocation,
                )
                mark_user_fund_cached(user_id, isin)
                logger.info(f"Pre-fetch cached: {name} ({isin})")

            except Exception as e:
                logger.warning(f"Pre-fetch failed for {name} ({isin}): {e}")

    tasks = [_prefetch_one(h) for h in holdings if h.get("units", 0) > 0.001]
    await asyncio.gather(*tasks)
    logger.info(f"Pre-fetch complete for user {user_id} ({len(tasks)} funds)")
