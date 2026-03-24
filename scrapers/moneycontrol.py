import logging
import requests
from typing import Optional, Dict, Any
from data.cache import get_cached, set_cached

logger = logging.getLogger(__name__)

class MoneyControlScraper:
    """Scrapes quantitative metrics from MoneyControl's internal API with 24h Redis caching."""
    
    BASE_URL = "https://api.moneycontrol.com/swiftapi/v1/mutualfunds"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    def _fetch_mc_data(self, endpoint: str, isin: str) -> Optional[Dict[str, Any]]:
        """Generic wrapper with caching (1 day TTL) and fallback to live fetch."""
        cache_key = f"mc:fund:{isin}:{endpoint}"
        
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
            
            res = requests.get(url, headers=self.HEADERS, timeout=10)
            res.raise_for_status()
            
            data = res.json()
            if data.get("success") in [1, True]:
                payload = data.get("data", {})
                
                # 3. Save to cache with 24 hour TTL (86400 seconds)
                if payload:
                    set_cached(cache_key, payload, ttl_seconds=86400)
                return payload
            else:
                logger.warning(f"MC API success=false for {url}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch {endpoint} for {isin}: {e}")
            return None

    def get_risk_metrics(self, isin: str) -> Dict[str, Any]:
        """Fetches standard deviation, sharpe, sortino, beta, and trailing returns vs category."""
        return self._fetch_mc_data("risk-metrics", isin) or {}

    def get_performance(self, isin: str) -> Dict[str, Any]:
        """Fetches annualised returns (1Y, 3Y, 5Y, 10Y) alongside benchmark figures."""
        # Note: 'performance?section=annualised' requires special formatting in our generic fetcher
        return self._fetch_mc_data("performance?section=annualised", isin) or {}

    def get_peers(self, isin: str) -> list:
        """Returns a list of direct competitors and their metrics."""
        return self._fetch_mc_data("peers", isin) or []

    def get_portfolio(self, isin: str) -> list:
        """MoneyControl's portfolio holdings (can be used as fallback)."""
        return self._fetch_mc_data("portfolio", isin) or []

    def get_fundamentals(self, isin: str) -> Dict[str, Any]:
        """Basic fund metrics like AUM, expense ratio from MC."""
        return self._fetch_mc_data("fundamentals", isin) or {}
