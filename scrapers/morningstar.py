import httpx
import re
from bs4 import BeautifulSoup
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class MorningstarScraper:
    def __init__(self):
        self.token = None
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.client = httpx.Client(timeout=15, follow_redirects=True)
        self._refresh_token()
        
    def _refresh_token(self):
        url = 'https://www.morningstar.in/mutualfunds/f00000pzh2/fund/detailed-portfolio.aspx'
        try:
            r = self.client.get(url, headers=self.headers)
            tokens = re.findall(r'(eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)', r.text)
            if tokens:
                self.token = max(tokens, key=len)
                self.headers['Authorization'] = f'Bearer {self.token}'
                return True
        except: pass
        return False

    def search_fund(self, query: str) -> Optional[Dict[str, str]]:
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
            r = self.client.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(r.text, 'html.parser')
            for table in soup.find_all('table'):
                t_type = table.find('type')
                if t_type and t_type.text == 'Fund':
                    m = table.find('id')
                    d = table.find('description')
                    if m and d: return {'id': m.text, 'name': d.text}
        except: pass
        return None

    def get_portfolio(self, mstar_id: str) -> Dict[str, float]:
        if not self.token and not self._refresh_token(): return {}
        url = f'https://www.us-api.morningstar.com/sal/sal-service/fund/portfolio/holding/v2/{mstar_id}/data?locale=en&clientId=RSIN_SAL'
        try:
            r = self.client.get(url, headers=self.headers)
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

    def get_benchmark(self, mstar_id: str) -> Optional[str]:
        """Extracts the exact Native Benchmark (e.g. 'Nifty 500 TR INR') from the Portfolio Schema."""
        if not self.token and not self._refresh_token(): return None
        url = f'https://www.us-api.morningstar.com/sal/sal-service/fund/portfolio/holding/v2/{mstar_id}/data?locale=en&clientId=RSIN_SAL'
        try:
            r = self.client.get(url, headers=self.headers)
            if r.status_code == 200:
                data = r.json()
                bm = data.get('holdingActiveShare', {}).get('primaryProspectusBenchmark')
                return bm
        except Exception as e:
            logger.error(f"Benchmark fetch failed for {mstar_id}: {e}")
        return None

    def get_fund_info(self, mstar_id: str) -> Dict:
        """
        Fetch sector allocation, AUM, expense ratio, and turnover from Morningstar.
        Returns a dict with: sector_allocation, portfolio_turnover_pct, aum_cr, expense_ratio
        """
        if not self.token and not self._refresh_token():
            return {}
        
        info = {
            'sector_allocation': [],
            'portfolio_turnover_pct': None,
            'aum_cr': None,
            'expense_ratio': None
        }
        
        try:
            # 1. Quote endpoint for AUM, Expense Ratio, and Turnover
            q_url = f'https://www.us-api.morningstar.com/sal/sal-service/fund/quote/v2/{mstar_id}/data?clientId=RSIN_SAL'
            r_q = self.client.get(q_url, headers=self.headers)
            if r_q.status_code == 200:
                q_data = r_q.json()
                
                exp = q_data.get('expense')
                if exp is not None:
                    info['expense_ratio'] = round(float(exp) * 100, 2)
                    
                turnover = q_data.get('lastTurnoverRatio')
                if turnover is not None:
                    info['portfolio_turnover_pct'] = round(float(turnover) * 100, 2)
                    
                aum = q_data.get('tNAInShareClassCurrency')
                if aum is not None:
                    # Convert to Crores (1 Crore = 10,000,000)
                    info['aum_cr'] = round(float(aum) / 10000000.0, 2)

            # 2. Sector endpoint for Equity Sector Breakdown
            s_url = f'https://www.us-api.morningstar.com/sal/sal-service/fund/portfolio/v2/sector/{mstar_id}/data?clientId=RSIN_SAL'
            r_s = self.client.get(s_url, headers=self.headers)
            if r_s.status_code == 200:
                s_data = r_s.json()
                eq_portfolio = s_data.get('EQUITY', {}).get('fundPortfolio', {})
                
                # Format CamelCase sector names to properly spaced titles (e.g. financialServices -> Financial Services)
                def fmt_sector(s):
                    return ''.join(' ' + c if c.isupper() else c for c in s).title().strip()
                
                sectors = []
                for k, v in eq_portfolio.items():
                    if k not in ('portfolioDate', 'masterPortfolioId') and v is not None and float(v) > 0:
                        sectors.append({'sector': fmt_sector(k), 'weight': round(float(v), 2)})
                
                info['sector_allocation'] = sorted(sectors, key=lambda x: x['weight'], reverse=True)

        except Exception as e:
            logger.error(f"Fund info fetch failed for {mstar_id}: {e}")
            
        return info

def get_rbi_repo_rate() -> float:
    """
    Dynamically scrape the RBI's official Policy Repo Rate to act as the exact base 
    for the 91-Day T-Bill Risk-Free Rate calculations.
    Returns the annualised rate percentage (e.g., 6.50). Returns 7.0 as fallback.
    """
    try:
        # RBI occasionally has strict SSL certs, bypass verify for stability on the homepage scrape
        with httpx.Client(timeout=10.0, verify=False) as client:
            r = client.get('https://www.rbi.org.in/', headers={'User-Agent': 'Mozilla/5.0'})
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

