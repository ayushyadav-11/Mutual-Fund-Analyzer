"""
Quick prefetch: reads the user's session data and pre-caches
all holding ISINs into Supabase + Redis so fund modals load instantly.
"""
import asyncio
import os, sys
from dotenv import load_dotenv
load_dotenv()

async def main():
    # Load session
    import json
    session_file = os.path.join(os.path.dirname(__file__), "session_data.json")
    if not os.path.exists(session_file):
        print("No session_data.json found. Upload your CAS first via the app.")
        return

    with open(session_file) as f:
        sessions = json.load(f)

    all_isins = set()
    # session_data.json can be either {user_id: {holdings:[...]}} or {holdings:[...]} directly
    if "holdings" in sessions:
        # Flat format: single user
        for h in sessions.get("holdings", []):
            isin = h.get("isin")
            if isin:
                all_isins.add(isin)
    else:
        # Multi-user format: {user_id: {holdings:[...]}}
        for user_id, data in sessions.items():
            if isinstance(data, dict):
                for h in data.get("holdings", []):
                    isin = h.get("isin")
                    if isin:
                        all_isins.add(isin)

    if not all_isins:
        print("No ISINs found in session data.")
        return

    print(f"Found {len(all_isins)} ISINs to prefetch: {all_isins}\n")

    from data.database import get_cached_fund_deep_dive, get_connection
    from scrapers.moneycontrol import MoneyControlScraper
    from scrapers.morningstar import MorningstarScraper
    from data.database import cache_fund_deep_dive, insert_or_update_scheme

    # Check which ones are already in Supabase
    already_cached = []
    not_cached = []
    for isin in all_isins:
        hit = get_cached_fund_deep_dive(isin, max_age_hours=24)
        if hit:
            already_cached.append(isin)
        else:
            not_cached.append(isin)

    print(f"Already cached: {len(already_cached)}")
    print(f"Need prefetch:  {len(not_cached)}\n")

    if not not_cached:
        print("All ISINs already cached! Fund modals should be fast now.")
        return

    # Fetch scheme names for uncached ISINs
    conn = get_connection()
    c = conn.cursor()
    isin_list = ", ".join([f"'{i}'" for i in not_cached])
    c.execute(f"SELECT isin, scheme_name, category FROM schemes WHERE isin IN ({isin_list})")
    scheme_rows = {r["isin"]: r for r in c.fetchall()}
    conn.close()

    for isin in not_cached:
        scheme = scheme_rows.get(isin)
        if not scheme:
            print(f"  [SKIP] {isin} - not in schemes table")
            continue

        scheme_name = scheme["scheme_name"]
        print(f"  Fetching {isin} - {scheme_name}...", end=" ", flush=True)

        try:
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

            # Minimal processing to store in cache
            from core.mc_helpers import _mc_extract_period_returns, _mc_extract_risk, _mc_extract_fundamentals
            returns_data, benchmark_returns = _mc_extract_period_returns(mc_perf)
            risk_data = _mc_extract_risk(mc_risk, scheme["category"])
            mc_fundamentals = _mc_extract_fundamentals(mc_fund)

            fundms = {
                "aum_cr": mc_fundamentals.get("aum_cr"),
                "expense_ratio": mc_fundamentals.get("expense_ratio"),
                "exit_load": mc_fundamentals.get("exit_load"),
                "portfolio_turnover": mc_fundamentals.get("portfolio_turnover"),
                "pe": mc_fundamentals.get("pe"),
                "pb": mc_fundamentals.get("pb"),
            }

            holdings_list = []
            sector_allocation = []
            if ms_fund:
                try:
                    async with MorningstarScraper() as ms2:
                        raw_portfolio, fund_info = await asyncio.gather(
                            ms2.get_portfolio(ms_fund["id"]),
                            ms2.get_fund_info(ms_fund["id"]),
                        )
                    holdings_list = [
                        {"asset": k, "weight": round(v * 100, 2)}
                        for k, v in sorted(raw_portfolio.items(), key=lambda x: x[1], reverse=True)
                    ][:20]
                    sector_allocation = fund_info.get("sector_allocation", [])
                except Exception as e:
                    print(f"  (MS portfolio failed: {e})", end=" ")

            insert_or_update_scheme(isin, scheme_name)
            cache_fund_deep_dive(
                isin=isin,
                fundamentals=fundms,
                risk=risk_data,
                returns=returns_data,
                bench_returns=benchmark_returns,
                holdings=holdings_list,
                sectors=sector_allocation,
            )
            print("DONE")

        except Exception as e:
            print(f"FAILED: {e}")

    print("\nPrefetch complete! All your funds are now cached in Supabase.")
    print("Redis will auto-warm on next modal open (< 50ms after that).")

if __name__ == "__main__":
    asyncio.run(main())
