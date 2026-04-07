"""
portfolio_overlap.py — Fetch & parse AMC monthly portfolio Excel disclosures
then compute pairwise portfolio overlap between the user's mutual funds.

Data source: AMC-published monthly portfolio files linked from
  https://www.amfiindia.com/online-center/portfolio-disclosure

Strategy:
  1. Match a fund's AMC name to a known disclosure page URL.
  2. Scrape that page (or use a hardcoded direct-download template) to find
     the latest Excel/XLS file.
  3. Parse the Excel → {instrument_name: weight%} dict.
  4. Compute pairwise weighted overlap for the user's holdings.
"""
import logging
from typing import Optional

from scrapers.morningstar import MorningstarScraper

logger = logging.getLogger(__name__)
_scraper = MorningstarScraper()

# Removing obsolete AMC registry and Excel parsers.


# ── Main Public API ────────────────────────────────────────────────────────────
def fetch_fund_holdings(fund_name: str, amc_name: str) -> dict[str, float]:
    """
    Fetch the latest monthly portfolio for a single fund seamlessly via Morningstar.
    Returns {instrument_name: weight_as_fraction} or {} on failure.
    """
    logger.info("Connecting to Aggregator API for: %s", fund_name)
    mstar_fund = _scraper.search_fund(fund_name)
    if not mstar_fund:
        logger.warning("Fund %s could not be found via aggregator API.", fund_name)
        return {}

    return _scraper.get_portfolio(mstar_fund['id'])


def compute_overlap(holdings_map: dict[str, dict[str, float]]) -> dict:
    """
    Given {fund_name: {instrument: weight_fraction}}, compute pairwise overlap.

    Returns:
        {
          "matrix": [[{"fund_a": str, "fund_b": str, "overlap_pct": float,
                       "shared_stocks": int, "jaccard": float}]],
          "top_pairs": [...],          # sorted by overlap_pct desc
          "per_fund": {fund: {"unique_stocks": [...], "top_holding": str}},
          "all_stocks": {stock: [funds that own it]},
        }
    """
    fund_names = list(holdings_map.keys())
    pairs = []

    for i in range(len(fund_names)):
        for j in range(i + 1, len(fund_names)):
            fa, fb = fund_names[i], fund_names[j]
            ha, hb = holdings_map[fa], holdings_map[fb]
            shared = set(ha.keys()) & set(hb.keys())
            union = set(ha.keys()) | set(hb.keys())

            # Weighted overlap: sum of min weights for shared instruments
            weighted_overlap = sum(min(ha[s], hb[s]) for s in shared)
            jaccard = len(shared) / len(union) if union else 0.0

            top_shared = sorted(shared, key=lambda s: min(ha[s], hb[s]), reverse=True)[:10]
            top_shared_detail = [
                {"name": s, "weight_a": round(ha[s] * 100, 2), "weight_b": round(hb[s] * 100, 2)}
                for s in top_shared
            ]

            pairs.append({
                "fund_a": fa,
                "fund_b": fb,
                "overlap_pct": round(weighted_overlap * 100, 2),
                "shared_stocks": len(shared),
                "jaccard": round(jaccard * 100, 2),
                "top_shared": top_shared_detail,
            })

    pairs.sort(key=lambda x: x["overlap_pct"], reverse=True)

    # Per-fund stats
    per_fund = {}
    for fn, h in holdings_map.items():
        other_instruments = set()
        for other_fn, oh in holdings_map.items():
            if other_fn != fn:
                other_instruments |= set(oh.keys())
        unique = [k for k in h if k not in other_instruments]
        top = max(h, key=h.get) if h else None
        per_fund[fn] = {
            "unique_stocks": unique[:10],
            "top_holding": top,
            "top_holding_weight": round(h[top] * 100, 2) if top else 0,
            "total_stocks": len(h),
        }

    # Cross-fund stock map
    all_stocks: dict[str, list[str]] = {}
    for fn, h in holdings_map.items():
        for stock in h:
            all_stocks.setdefault(stock, []).append(fn)

    # Only keep stocks held by 2+ funds (interesting overlaps)
    multi_fund_stocks = {s: funds for s, funds in all_stocks.items() if len(funds) > 1}

    return {
        "fund_count": len(fund_names),
        "funds_with_data": [f for f, h in holdings_map.items() if h],
        "funds_without_data": [f for f, h in holdings_map.items() if not h],
        "top_pairs": pairs[:10],
        "all_pairs": pairs,
        "per_fund": per_fund,
        "multi_fund_stocks": multi_fund_stocks,
        "data_month": _get_data_month_label(),
    }


def _get_data_month_label() -> str:
    import datetime
    # Data is from previous month (monthly disclosure)
    today = datetime.date.today()
    m = today.month - 1 or 12
    y = today.year if today.month > 1 else today.year - 1
    import calendar
    return f"{calendar.month_name[m]} {y}"
