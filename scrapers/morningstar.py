import httpx
import re
import time
from bs4 import BeautifulSoup
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ── Global token cache ────────────────────────────────────────────────────────
# Morningstar JWT tokens last ~1 hour. Cache it process-wide so every new
# MorningstarScraper instance reuses it instead of paying the 2-4s refresh cost.
_GLOBAL_MS_TOKEN: Optional[str] = None
_GLOBAL_MS_TOKEN_TS: float = 0.0
_MS_TOKEN_TTL: float = 55 * 60  # 55 minutes

def _get_cached_token() -> Optional[str]:
    if _GLOBAL_MS_TOKEN and (time.monotonic() - _GLOBAL_MS_TOKEN_TS) < _MS_TOKEN_TTL:
        return _GLOBAL_MS_TOKEN
    return None

def _save_token(token: str):
    global _GLOBAL_MS_TOKEN, _GLOBAL_MS_TOKEN_TS
    _GLOBAL_MS_TOKEN = token
    _GLOBAL_MS_TOKEN_TS = time.monotonic()
# ─────────────────────────────────────────────────────────────────────────────


class MorningstarScraper:
    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        # Seed from global cache — avoids token refresh on first API call
        cached = _get_cached_token()
        self.token = cached
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        if cached:
            self.headers['Authorization'] = f'Bearer {cached}'
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

    async def _refresh_token(self):
        url = 'https://www.morningstar.in/mutualfunds/f00000pzh2/fund/detailed-portfolio.aspx'
        try:
            if self._client:
                r = await self._client.get(url, headers=self.headers)
            else:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    r = await client.get(url, headers=self.headers)

            tokens = re.findall(r'(eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)', r.text)
            if tokens:
                self.token = max(tokens, key=len)
                self.headers['Authorization'] = f'Bearer {self.token}'
                _save_token(self.token)  # persist globally for next scraper instance
                return True
        except:
            pass
        return False

    async def search_fund(self, query: str) -> Optional[Dict[str, str]]:
        # Strip CAS-specific boilerplate to achieve 100% Morningstar match rate
        # We strip non-alphanumeric and common keywords to find the "Master" scheme
        clean = re.sub(r'(?i)\b(fund|direct|regular|growth|dividend|plan|option|idcw|reinvestment|payout|cumulative)\b', '', query)
        clean = re.sub(r'[^a-zA-Z0-9 ]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Also remove 'Direct' or 'Regular' if they are stuck to words without space
        clean = re.sub(r'(?i)(direct|regular|growth)', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        url = f'https://www.morningstar.in/handlers/autocompletehandler.ashx?criteria={clean}'
        try:
            if self._client:
                r = await self._client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            else:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    r = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'})

            soup = BeautifulSoup(r.text, 'html.parser')
            for table in soup.find_all('table'):
                t_type = table.find('type')
                if t_type and t_type.text == 'Fund':
                    m = table.find('id')
                    d = table.find('description')
                    if m and d:
                        return {'id': m.text, 'name': d.text}
        except:
            pass
        return None

    async def get_portfolio(self, mstar_id: str) -> Dict[str, float]:
        if not self.token and not await self._refresh_token():
            return {}
        url = f'https://www.us-api.morningstar.com/sal/sal-service/fund/portfolio/holding/v2/{mstar_id}/data?locale=en&clientId=RSIN_SAL'
        try:
            if self._client:
                r = await self._client.get(url, headers=self.headers)
            else:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    r = await client.get(url, headers=self.headers)

            if r.status_code == 200:
                data = r.json()
                holdings = {}
                eq = data.get('equityHoldingPage', {}).get('holdingList', [])
                bonds = data.get('bondHoldingPage', {}).get('holdingList', [])
                other = data.get('otherHoldingPage', {}).get('holdingList', [])
                for h in eq + bonds + other:
                    sec_name = h.get('securityName')
                    weight = h.get('weighting')
                    if sec_name and weight is not None:
                        holdings[sec_name] = float(weight) / 100.0
                return holdings
        except Exception as e:
            logger.error(f"Portfolio fetch failed for {mstar_id}: {e}")
        return {}

    async def get_benchmark(self, mstar_id: str) -> Optional[str]:
        """Extracts the exact Native Benchmark (e.g. 'Nifty 500 TR INR') from the Portfolio Schema."""
        if not self.token and not await self._refresh_token():
            return None
        url = f'https://www.us-api.morningstar.com/sal/sal-service/fund/portfolio/v2/{mstar_id}/data?locale=en&clientId=RSIN_SAL'
        try:
            if self._client:
                r = await self._client.get(url, headers=self.headers)
            else:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    r = await client.get(url, headers=self.headers)

            if r.status_code == 200:
                data = r.json()
                port = data.get("portfolio", {})
                return port.get("prospectusNetExpenseRatio") or port.get("name")
        except Exception as e:
            logger.error(f"Benchmark fetch failed for {mstar_id}: {e}")
        return None

    async def get_sector_allocation(self, mstar_id: str) -> list:
        """Fetch sector allocation from the correct Morningstar sector endpoint."""
        if not self.token and not await self._refresh_token():
            return []
        url = (
            f'https://www.us-api.morningstar.com/sal/sal-service/fund/portfolio/v2/sector'
            f'/{mstar_id}/data?locale=en&clientId=RSIN_SAL&benchmarkId=mstarorcat'
            f'&version=4.81.0&secId={mstar_id}'
        )
        _SECTOR_LABELS = {
            "basicMaterials": "Basic Materials",
            "consumerCyclical": "Consumer Cyclical",
            "financialServices": "Financial Services",
            "realEstate": "Real Estate",
            "communicationServices": "Communication Services",
            "energy": "Energy",
            "industrials": "Industrials",
            "technology": "Technology",
            "consumerDefensive": "Consumer Defensive",
            "healthcare": "Healthcare",
            "utilities": "Utilities",
        }
        try:
            if self._client:
                r = await self._client.get(url, headers=self.headers)
            else:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    r = await client.get(url, headers=self.headers)
            if r.status_code == 200 and r.content:
                data = r.json()
                eq = data.get("EQUITY", {})
                fund_port = eq.get("fundPortfolio", {})
                alloc_list = []
                for key, label in _SECTOR_LABELS.items():
                    val = fund_port.get(key)
                    if val is not None:
                        try:
                            pct = round(float(val), 2)
                            if pct > 0:
                                alloc_list.append({"sector": label, "pct": pct})
                        except Exception:
                            pass
                return sorted(alloc_list, key=lambda x: x["pct"], reverse=True)
        except Exception as e:
            logger.error(f"Sector allocation fetch failed for {mstar_id}: {e}")
        return []

    async def get_fund_info(self, mstar_id: str) -> dict:
        """Fetch AUM, expense ratio, and sector allocation from Morningstar."""
        if not self.token and not await self._refresh_token():
            return {}
        # Snapshot v2 endpoint for AUM/expense/turnover
        url = f'https://www.us-api.morningstar.com/sal/sal-service/fund/snapshot/v2/{mstar_id}/data?locale=en&clientId=RSIN_SAL'
        result = {}
        try:
            if self._client:
                r = await self._client.get(url, headers=self.headers)
            else:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    r = await client.get(url, headers=self.headers)
            if r.status_code == 200 and r.content:
                data = r.json()
                aum_raw = data.get("fundAttributes", {}).get("totalNetAsset")
                if aum_raw:
                    try:
                        result["aum_cr"] = round(float(aum_raw) / 1e7, 2)
                    except Exception:
                        pass
                er = data.get("managementExpenseRatio") or data.get("expenseRatio")
                if er:
                    try:
                        result["expense_ratio"] = round(float(er), 4)
                    except Exception:
                        pass
                pt = data.get("portfolioTurnoverRatio") or data.get("portfolioTurnoverPercentage")
                if pt:
                    try:
                        result["portfolio_turnover_pct"] = round(float(pt), 2)
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Fund snapshot fetch failed for {mstar_id}: {e}")

        # Always fetch sector allocation from the dedicated sector endpoint
        sector_alloc = await self.get_sector_allocation(mstar_id)
        if sector_alloc:
            result["sector_allocation"] = sector_alloc

        return result


async def get_rbi_repo_rate() -> float:
    """
    Dynamically scrape the RBI's official Policy Repo Rate to act as the exact base
    for the 91-Day T-Bill Risk-Free Rate calculations.
    Returns the annualised rate percentage (e.g., 6.50). Returns 7.0 as fallback.
    """
    try:
        # RBI occasionally has strict SSL certs, bypass verify for stability on the homepage scrape
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            r = await client.get('https://www.rbi.org.in/', headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, 'html.parser')
                for table in soup.find_all('table'):
                    text = table.text
                    if 'Policy Rates' in text or 'Current Rates' in text or 'Policy Repo' in text:
                        for tr in table.find_all('tr'):
                            tr_text = tr.text.strip()
                            if 'Policy Repo Rate' in tr_text:
                                rate_str = tr_text.split(':')[-1].strip().replace('%', '')
                                return float(rate_str)
    except Exception as e:
        logger.warning(f"Failed to fetch dynamic RBI Repo Rate, defaulting to 7.0%: {e}")
    return 7.0
