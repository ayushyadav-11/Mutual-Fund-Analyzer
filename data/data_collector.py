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

    async with semaphore:
        code = await _resolve_scheme_code(client, isin, name)
        if not code:
            logger.warning(f"Could not resolve scheme code for {name} ({isin})")
            return

        insert_or_update_scheme(isin, name, scheme_code=code)

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

            batch_insert_navs(isin, nav_records)
            logger.info(f"Ingested {len(nav_records)} NAV records for {name} ({isin})")
        except Exception as e:
            logger.error(f"NAV ingestion failed for {name}: {e}")


async def fetch_and_populate_mfapi_data(holdings: list):
    """
    Called by /api/risk on startup.  Concurrently resolves scheme codes via
    MFAPI search (avoids the broken /mf master-list endpoint) and stores
    historical NAVs in SQLite/Postgres.
    """
    semaphore = asyncio.Semaphore(5)        # max 5 concurrent requests
    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = [_ingest_fund(client, h, semaphore) for h in holdings]
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    import json, sys
    logging.basicConfig(level=logging.INFO)

    try:
        with open("session_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("session_data.json not found – trying database…")
        from core.parser import load_session
        data = load_session()

    print("Ingesting historical NAVs from MFAPI…")
    asyncio.run(fetch_and_populate_mfapi_data(data.get("holdings", [])))
    print("Done.")
