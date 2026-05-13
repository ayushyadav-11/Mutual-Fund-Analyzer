"""
core/chart_builder.py
---------------------
Builds the ₹10,000 NAV growth chart (fund vs benchmark) from a list of NAV records.
Extracted so the nightly job and app.py both call the same logic.
"""
from __future__ import annotations
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# In-process benchmark cache (shared across imports)
_benchmark_cache: dict = {}


def build_nav_growth_chart(
    navs: List[Dict],
    benchmark_symbol: Optional[str] = None,
    benchmark_label: Optional[str] = None,
    months: int = 60,
) -> dict:
    """
    Build a ₹10,000 lumpsum NAV growth chart from a list of NAV records.

    Parameters
    ----------
    navs : list of {"date": "YYYY-MM-DD", "nav": float}
    benchmark_symbol : Yahoo Finance ticker (e.g. "^NSEI") or None
    benchmark_label  : Human-readable benchmark name
    months           : Number of trailing months to include (default 60 = 5 years)

    Returns
    -------
    dict with keys: labels, fund_value, benchmark_value, benchmark_name, chart_type
    Empty dict if navs is empty.
    """
    if not navs:
        return {}

    # ── 1. Monthly NAV dict (YYYY-MM → float) ────────────────────────────────
    monthly_nav: dict = {}
    for row in sorted(navs, key=lambda x: x["date"]):
        try:
            ym = str(row["date"])[:7]  # "YYYY-MM"
            monthly_nav[ym] = float(row["nav"])
        except Exception:
            continue

    if not monthly_nav:
        return {}

    recent_months = sorted(monthly_nav.keys())[-months:]

    # ── 2. Optional benchmark ────────────────────────────────────────────────
    bench_monthly: Optional[dict] = None
    bench_ok = False
    if benchmark_symbol:
        cache_key = f"bm:{benchmark_symbol}:5y"
        if cache_key in _benchmark_cache:
            bench_cumprod = _benchmark_cache[cache_key]
            bench_ok = True
        else:
            try:
                from core.risk import _fetch_benchmark_returns
                bench_start = (date.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
                bench_end   = date.today().strftime("%Y-%m-%d")
                bench_s = _fetch_benchmark_returns(benchmark_symbol, bench_start, bench_end)
                if bench_s is not None:
                    bench_cumprod = (1 + bench_s).cumprod()
                    _benchmark_cache[cache_key] = bench_cumprod
                    bench_ok = True
            except Exception as e:
                logger.warning("Benchmark fetch failed for %s: %s", benchmark_symbol, e)

        if bench_ok:
            bench_monthly = {}
            for date_idx, val in bench_cumprod.items():
                ym = date_idx.strftime("%Y-%m") if hasattr(date_idx, "strftime") else str(date_idx)[:7]
                bench_monthly[ym] = float(val)

    # ── 3. Build ₹10k growth series ─────────────────────────────────────────
    labels: list = []
    fund_vals: list = []
    bench_vals: list = []
    start_nav = None
    start_bench = None

    for ym in recent_months:
        nav_val = monthly_nav[ym]
        if start_nav is None:
            start_nav = nav_val
            if bench_monthly:
                start_bench = bench_monthly.get(ym)

        try:
            dt_obj = datetime.strptime(ym, "%Y-%m")
            labels.append(dt_obj.strftime("%b %Y"))
        except Exception:
            labels.append(ym)

        fund_vals.append(round(10000 * nav_val / start_nav, 2))

        if bench_monthly and start_bench:
            bm_val = bench_monthly.get(ym)
            bench_vals.append(round((bm_val / start_bench) * 10000 if bm_val else 0, 2))
        else:
            bench_vals.append(None)

    if not labels:
        return {}

    return {
        "labels":          labels,
        "fund_value":      fund_vals,
        "benchmark_value": bench_vals,
        "benchmark_name":  benchmark_label if bench_ok else None,
        "chart_type":      "growth",
    }
