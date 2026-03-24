import httpx
import logging
import asyncio
from datetime import datetime
from data.database import insert_or_update_scheme, batch_insert_navs
from core.parser import get_benchmark_ticker

logger = logging.getLogger(__name__)

async def fetch_and_populate_mfapi_data(holdings: list):
    """
    Operates identically to the legacy memory-cache function,
    but commits explicitly the chronological outputs natively to the SQLite engine.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        # First query: search for the scheme code via MFAPI endpoint
        for h in holdings:
            isin = h.get('isin')
            name = h.get('name')
            units = h.get("units", 0)
            
            if not isin or units < 0.001:
                continue

            # Ensure fund exists in DB
            benchmark = get_benchmark_ticker(name, h.get("category", ""))
            insert_or_update_scheme(
                isin=isin, 
                scheme_name=name,
                category=h.get('category'),
                benchmark=benchmark
            )
            
        # First, grab the master scheme list to map ISIN -> Scheme Code accurately
        try:
            res_master = await client.get("https://api.mfapi.in/mf")
            if res_master.status_code == 200:
                master = res_master.json()
                isin_to_code = {}
                for entry in master:
                    code = entry.get("schemeCode")
                    ig = entry.get("isinGrowth")
                    if ig: isin_to_code[ig] = code
                    idiv = entry.get("isinDivReinvestment")
                    if idiv: isin_to_code[idiv] = code
            else:
                logger.error("Failed to load mfapi master list")
                return
        except Exception as e:
            logger.error(f"Error fetching master list: {e}")
            return

        for h in holdings:
            isin = h.get('isin')
            name = h.get('name')
            units = h.get("units", 0)
            
            if not isin or units < 0.001:
                continue

            # Ensure fund exists in DB
            benchmark = get_benchmark_ticker(name, h.get("category", ""))
            insert_or_update_scheme(
                isin=isin, 
                scheme_name=name,
                category=h.get('category'),
                benchmark=benchmark
            )
            
            scheme_code = isin_to_code.get(isin)
            if scheme_code:
                try:
                    insert_or_update_scheme(isin, name, scheme_code=scheme_code)
                    
                    # Phase 2: Rip exact historical NAV timeline
                    history_url = f"https://api.mfapi.in/mf/{scheme_code}"
                    h_res = await client.get(history_url)
                    if h_res.status_code == 200:
                        h_data = h_res.json()
                        raw_data = h_data.get("data", [])
                        
                        # Convert parsed DD-MM-YYYY to canonical YYYY-MM-DD for mathematically solid SQL indexing
                        nav_records = []
                        for point in raw_data:
                            try:
                                # mfapi format: dd-mm-yyyy
                                dt = datetime.strptime(point["date"], "%d-%m-%Y")
                                canonical_date = dt.strftime("%Y-%m-%d")
                                nav_records.append({
                                    "date": canonical_date, 
                                    "nav": float(point["nav"])
                                })
                            except Exception:
                                pass
                                
                        batch_insert_navs(isin, nav_records)
                        logger.info(f"Ingested {len(nav_records)} historical NAV traces into SQLite for {name} ({isin})")
                except Exception as e:
                    logger.error(f"Ingestion failed for {name}: {e}")
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    
    # Test execution over the user's specific session cache
    with open('session_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Initiating full SQLite data cascade from MFAPI...")
    asyncio.run(fetch_and_populate_mfapi_data(data.get('holdings', [])))
    print("Data Ingestion Concluded Successfully.")
