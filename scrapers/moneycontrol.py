import logging
import httpx
from typing import Optional, Dict, Any
from data.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

DEBUG_BYPASS_MONEYCONTROL_CACHE = False

class MoneyControlScraper:
    """Scrapes quantitative metrics from MoneyControl's internal API with 24h Redis caching."""
    
    BASE_URL = "https://api.moneycontrol.com/swiftapi/v1/mutualfunds"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._client = client
        self._owns_client = False

    async def __aenter__(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
            self._owns_client = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owns_client and self._client:
            await self._client.aclose()

    async def _fetch_mc_data(self, endpoint: str, isin: str) -> Optional[Dict[str, Any]]:
        """Generic wrapper with caching (1 day TTL) and fallback to live fetch."""
        cache_key = f"mc:fund:{isin}:{endpoint}"

        # Debug toggle: bypass MoneyControl endpoint cache to inspect live payloads.
        if not DEBUG_BYPASS_MONEYCONTROL_CACHE:
            # 1. Check Redis/Memory cache
            cached_data = get_cached(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data

        # 2. Live Fetch
        logger.info(f"Live fetch from MoneyControl: {endpoint} for {isin}")
        try:
            # Build URL (handle query params logic)
            separator = "&" if "?" in endpoint else "?"
            url = f"{self.BASE_URL}/{endpoint}{separator}isin={isin}&deviceType=W&responseType=json"
            
            # Use provided client or create an ephemeral one
            if self._client:
                res = await self._client.get(url, headers=self.HEADERS)
            else:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    res = await client.get(url, headers=self.HEADERS)
                    
            res.raise_for_status()
            
            data = res.json()
            if data.get("success") in [1, True]:
                payload = data.get("data", {})
                
                # 3. Save to cache with 24 hour TTL (86400 seconds)
                if payload and not DEBUG_BYPASS_MONEYCONTROL_CACHE:
                    set_cached(cache_key, payload, ttl_seconds=86400)
                return payload
            else:
                logger.warning(f"MC API success=false for {url}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch {endpoint} for {isin}: {e}")
            return None

    async def get_risk_metrics(self, isin: str) -> Dict[str, Any]:
        """Fetches standard deviation, sharpe, sortino, beta, and trailing returns vs category."""
        return await self._fetch_mc_data("risk-metrics", isin) or {}

    async def get_performance(self, isin: str) -> Dict[str, Any]:
        """Fetches annualised returns (1Y, 3Y, 5Y, 10Y) alongside benchmark figures."""
        return await self._fetch_mc_data("performance?section=annualised", isin) or {}

    async def get_performance_yearly(self, isin: str) -> list:
        """Fetches calendar-year returns with category average and rank-in-category."""
        raw = await self._fetch_mc_data("performance?section=yearly", isin)
        try:
            return (raw or [{}])[0].get("lumpsum", {}).get("yearly", [])
        except Exception:
            return []

    async def get_performance_sip(self, isin: str) -> list:
        """Fetches SIP return table (1Y/3Y/5Y/10Y invested & current value)."""
        raw = await self._fetch_mc_data("performance?section=sip", isin)
        try:
            return (raw or [{}])[0].get("sip", [])
        except Exception:
            return []


    async def get_peers(self, isin: str) -> list:
        """Returns a list of direct competitors and their metrics."""
        return await self._fetch_mc_data("peers", isin) or []

    async def get_portfolio(self, isin: str) -> list:
        """MoneyControl's portfolio holdings (can be used as fallback)."""
        return await self._fetch_mc_data("portfolio", isin) or []

    async def get_fundamentals(self, isin: str) -> Dict[str, Any]:
        """Fetches valuation/style-box metrics (PE, PB, Price/Sale, Dividend Yield, ROE)."""
        snap = await self._fetch_mc_data("fundamentals", isin) or {}
        logger.info(f"[MC Debug] Raw fundamentals response keys for {isin}: {list(snap.keys())}")
        return snap

    async def get_overview(self, isin: str) -> Dict[str, Any]:
        """Fetches AUM, expense ratio, and portfolio turnover from the MC overview endpoint."""
        data = await self._fetch_mc_data("overview", isin) or {}

        def _clean_float(val):
            if val is None:
                return None
            try:
                return float(str(val).replace(",", "").replace("%", "").strip())
            except (ValueError, TypeError):
                return None

        return {
            "aum_cr": _clean_float(data.get("aum")),
            "expense_ratio": _clean_float(data.get("expenseRatio")),
            "portfolio_turnover": _clean_float(data.get("turnoverRatio")),
            # NAV is also available directly from overview — use as fallback
            "latest_nav": _clean_float(data.get("latestNAV")),
            "nav_date": data.get("navDate"),
            # Bonus metrics available from overview
            "sharpe_overview": _clean_float(data.get("sharpeRatio")),
            "std_dev_overview": _clean_float(data.get("stadardDeviation")),
            "beta_overview": _clean_float(data.get("beta_3_year")),
        }

