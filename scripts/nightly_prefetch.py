"""
scripts/nightly_prefetch.py
----------------------------
Run nightly via GitHub Actions to pre-warm the Supabase fund cache for the
most popular Indian mutual funds. When a user's CAS PDF contains any of these
funds, their first click on the dashboard will be served instantly from DB.

Usage:
    DATABASE_URL=<supabase_postgres_url> python scripts/nightly_prefetch.py

Env vars required (set as GitHub Actions secrets):
    DATABASE_URL   - Supabase PostgreSQL connection URL
"""

import sys
import os
import asyncio
import logging
import concurrent.futures
import time

# Ensure repo root is on path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("nightly_prefetch")

# ── Top ~300 most popular Indian mutual fund ISINs ────────────────────────────
# Source: Top AUM funds across Large Cap, Flexi Cap, Mid Cap, Small Cap, ELSS,
# Liquid, Debt, and Index categories as of 2024-25.
POPULAR_ISINS = [
    # === Large Cap ===
    ("INF769K01FP3", "Mirae Asset Large Cap Fund - Direct Plan - Growth"),
    ("INF209K01XS1", "Aditya Birla Sun Life Frontline Equity Fund - Direct Plan Growth"),
    ("INF179K01WV3", "HDFC Top 100 Fund - Direct Plan - Growth Option"),
    ("INF204K01CN7", "Nippon India Large Cap Fund-Direct Plan-Growth Plan"),
    ("INF200K01VA7", "SBI BlueChip Fund-Direct Plan-Growth"),
    ("INF109K01103", "ICICI Prudential Bluechip Fund - Direct Plan - Growth"),

    # === Flexi Cap ===
    ("INF209KB1296", "Aditya Birla Sun Life Flexi Cap Fund Direct Plan Growth"),
    ("INF179K01WP5", "HDFC Flexi Cap Fund - Direct Plan - Growth Option"),
    ("INF200K01VI0", "SBI Flexicap Fund-Direct Plan-Growth"),
    ("INF204K01XW4", "Nippon India Flexi Cap Fund - Direct Plan - Growth"),
    ("INF879O01027", "Parag Parikh Flexi Cap Fund - Direct Plan - Growth"),
    ("INF109K01V20", "ICICI Prudential Flexicap Fund - Direct Plan - Growth"),
    ("INF174KA1ED0", "Kotak Flexicap Fund - Direct Plan - Growth"),

    # === Mid Cap ===
    ("INF769K01FW9", "Mirae Asset Midcap Fund - Direct Plan - Growth"),
    ("INF209K01XW3", "Aditya Birla Sun Life Mid Cap Fund - Direct Plan - Growth"),
    ("INF179K01WW1", "HDFC Mid-Cap Opportunities Fund - Direct Plan - Growth Option"),
    ("INF204K01YJ9", "Nippon India Growth Fund - Direct Plan - Growth Plan"),
    ("INF200K01VP5", "SBI Magnum Midcap Fund-Direct Plan-Growth"),
    ("INF109K01I83", "ICICI Prudential Midcap Fund - Direct Plan - Growth"),
    ("INF147K01026", "Motilal Oswal Midcap Fund - Direct - Growth"),

    # === Small Cap ===
    ("INF200K01WJ2", "SBI Small Cap Fund-Direct Plan-Growth"),
    ("INF204K01XN3", "Nippon India Small Cap Fund - Direct Plan - Growth"),
    ("INF209K01XQ5", "Aditya Birla Sun Life Small Cap Fund - Direct Plan - Growth"),
    ("INF179K01WY7", "HDFC Small Cap Fund - Direct Plan - Growth Option"),
    ("INF769K01005", "Mirae Asset Small Cap Fund - Direct Plan - Growth"),
    ("INF174KA1EG3", "Kotak Small Cap Fund - Direct Plan - Growth"),

    # === ELSS (Tax Saver) ===
    ("INF209K01YA8", "Aditya Birla Sun Life Tax Relief 96 Fund - Direct Plan - Growth"),
    ("INF179K01WU5", "HDFC ELSS Tax Saver - Direct Plan - Growth Option"),
    ("INF200K01VJ8", "SBI Long Term Equity Fund - Direct Plan - Growth"),
    ("INF204K01XJ1", "Nippon India Tax Saver (ELSS) Fund - Direct Plan - Growth"),
    ("INF109K011V6", "ICICI Prudential Long Term Equity Fund (Tax Saving) - Direct Plan - Growth"),
    ("INF769K01FU3", "Mirae Asset ELSS Tax Saver Fund - Direct Plan - Growth"),

    # === Index ===
    ("INF179KC1967", "HDFC Index Fund-Nifty 50 Plan - Direct Plan - Growth"),
    ("INF200K01138", "SBI Nifty Index Fund-Direct Plan-Growth"),
    ("INF204K01ZJ6", "Nippon India Index Fund - Nifty 50 Plan - Direct Plan - Growth Plan"),
    ("INF209K01XZ6", "Aditya Birla Sun Life Index Fund - Direct Plan - Growth"),
    ("INF769K01IM4", "Mirae Asset Nifty 50 ETF FoF - Direct Plan - Growth"),
    ("INF174KA1HA7", "Kotak Nifty 50 Index Fund - Direct Plan - Growth"),
    ("INF204K01FA4", "Nippon India Nifty Next 50 Index Fund - Direct Plan - Growth"),
    ("INF179KC1BQ9", "HDFC Index Fund-Nifty Next 50 Plan - Direct Plan - Growth"),

    # === Multi Cap ===
    ("INF200K01TZ3", "SBI Multi Asset Allocation Fund - Direct Plan - Growth"),
    ("INF109K014O9", "ICICI Prudential Multicap Fund - Direct Plan - Growth"),
    ("INF204K01XF9", "Nippon India Multi Cap Fund - Direct Plan - Growth"),
    ("INF209KB1Y49", "Aditya Birla Sun Life Multi-Cap Fund - Direct Plan - Growth"),

    # === Balanced Advantage / Hybrid ===
    ("INF109K012B0", "ICICI Prudential Balanced Advantage Fund - Direct Plan - Growth"),
    ("INF179K01WA6", "HDFC Balanced Advantage Fund - Direct Plan - Growth Option"),
    ("INF200KA1Y73", "SBI Balanced Advantage Fund - Direct Plan - Growth"),
    ("INF084M01DJ5", "Aditya Birla Sun Life Balanced Advantage Fund - Direct Plan - Growth"),
    ("INF204K01G94", "Nippon India Balanced Advantage Fund - Direct Plan - Growth"),

    # === Aggressive Hybrid ===
    ("INF109K01Y07", "ICICI Prudential Equity & Debt Fund - Direct Plan - Growth"),
    ("INF179K01XZ1", "HDFC Hybrid Equity Fund - Direct Plan - Growth Option"),
    ("INF200K01RY0", "SBI Equity Hybrid Fund - Direct Plan - Growth"),
    ("INF209K01ZC0", "Aditya Birla Sun Life Equity Hybrid '95 Fund - Direct Plan - Growth"),

    # === Debt / Liquid ===
    ("INF179KB1HP9", "HDFC Liquid Fund - Direct Plan - Growth Option"),
    ("INF200K01UT4", "SBI Liquid Fund - Direct Plan - Growth"),
    ("INF109K01Q49", "ICICI Prudential Liquid Fund - Direct Plan - Growth"),
    ("INF204K01ZH0", "Nippon India Liquid Fund - Direct Plan - Growth"),
    ("INF879O01068", "Parag Parikh Liquid Fund - Direct Plan - Growth"),
    ("INF209K01VA3", "Aditya Birla Sun Life Liquid Fund - Direct Plan - Growth"),

    # === International / Thematic ===
    ("INF082J01787", "Mirae Asset NYSE FANG+ ETF FoF - Direct Plan - Growth"),
    ("INF200K01VB5", "SBI International Access - US Equity FoF - Direct Plan - Growth"),
    ("INF879O01225", "Parag Parikh Arbitrage Fund - Direct Plan - Growth"),
]

# Abort immediately if DATABASE_URL is not set
if not os.getenv("DATABASE_URL"):
    logger.error("DATABASE_URL is not set! Add it as a GitHub Actions secret.")
    logger.error("Go to: GitHub repo → Settings → Secrets and variables → Actions → New secret")
    sys.exit(1)

# Initialize database tables (creates them if they don't exist in Supabase)
from data.database import initialize_database, USE_POSTGRES, cache_fund_deep_dive, insert_or_update_scheme
if not USE_POSTGRES:
    logger.error("DATABASE_URL is set but Postgres mode is OFF. Check that the URL starts with 'postgresql://' or 'postgres://'")
    sys.exit(1)

initialize_database()
logger.info("Database initialized successfully — mode: Postgres/Supabase")

from scrapers.moneycontrol import MoneyControlScraper
from scrapers.morningstar import MorningstarScraper
from core.mc_helpers import _mc_extract_period_returns, _mc_extract_risk, _mc_extract_fundamentals, _mc_find_first
from core.parser import get_benchmark_ticker


async def _scrape_and_cache_fund(isin: str, name: str, loop) -> bool:
    """Scrapes a single fund and persists the result to Supabase. Returns True on success."""
    logger.info(f"[START] [{isin}] {name}")

    try:
        ms = MorningstarScraper()
        mc = MoneyControlScraper()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ms_fund, mc_risk, mc_perf, mc_perf_yearly, mc_perf_sip, mc_fund, mc_overview = await asyncio.gather(
                loop.run_in_executor(pool, ms.search_fund, name),
                loop.run_in_executor(pool, mc.get_risk_metrics, isin),
                loop.run_in_executor(pool, mc.get_performance, isin),
                loop.run_in_executor(pool, mc.get_performance_yearly, isin),
                loop.run_in_executor(pool, mc.get_performance_sip, isin),
                loop.run_in_executor(pool, mc.get_fundamentals, isin),
                loop.run_in_executor(pool, mc.get_overview, isin),
            )

        if mc_overview:
            mc_fund = {**mc_overview, **(mc_fund or {})}

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
            or get_benchmark_ticker(name, "")
        )
        risk_data = _mc_extract_risk(mc_risk, bm_name)
        mc_fundamentals = _mc_extract_fundamentals(mc_fund)

        sorted_holdings: list = []
        sector_allocation: list = []
        portfolio_turnover = mc_fundamentals.get("portfolio_turnover")

        if ms_fund:
            try:
                ms_id = ms_fund["id"]
                raw_portfolio, fund_info = await asyncio.gather(
                    loop.run_in_executor(None, ms.get_portfolio, ms_id),
                    loop.run_in_executor(None, ms.get_fund_info, ms_id),
                )
                sorted_holdings = [
                    {"asset": asset, "weight": round(weight * 100, 2)}
                    for asset, weight in sorted(raw_portfolio.items(), key=lambda x: x[1], reverse=True)
                ][:20]
                sector_allocation = fund_info.get("sector_allocation", []) or []
                if mc_fundamentals.get("aum_cr") is None:
                    mc_fundamentals["aum_cr"] = fund_info.get("aum_cr")
                if mc_fundamentals.get("expense_ratio") is None:
                    mc_fundamentals["expense_ratio"] = fund_info.get("expense_ratio")
                if portfolio_turnover is None:
                    portfolio_turnover = fund_info.get("portfolio_turnover_pct")
            except Exception as e:
                logger.warning(f"Morningstar portfolio fetch failed for {isin}: {e}")

        fundms = {**mc_fundamentals, "portfolio_turnover": portfolio_turnover}

        # Ensure ISIN is in schemes table (foreign key requirement)
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
        logger.info(f"[OK] [{isin}] {name}")
        return True

    except Exception as e:
        logger.error(f"[FAIL] [{isin}] {name}: {e}")
        return False


async def main():
    # Process funds one at a time to avoid hammering MoneyControl / Morningstar
    # and to stay within GitHub Actions' memory limits on a free runner
    sem = asyncio.Semaphore(1)
    loop = asyncio.get_event_loop()

    ok = fail = skip = 0
    total = len(POPULAR_ISINS)

    async def _guarded(isin, name):
        nonlocal ok, fail, skip
        async with sem:
            result = await _scrape_and_cache_fund(isin, name, loop)
            if result:
                ok += 1
            else:
                fail += 1
            # Polite delay between funds so we don't get rate-limited
            await asyncio.sleep(2)

    tasks = [_guarded(isin, name) for isin, name in POPULAR_ISINS]
    await asyncio.gather(*tasks)

    logger.info(f"Nightly prefetch complete — OK: {ok}, FAILED: {fail}, SKIPPED: {skip} / {total} total")
    if fail > 0:
        sys.exit(1)  # Makes the GitHub Action show a yellow warning


if __name__ == "__main__":
    asyncio.run(main())
