"""
risk.py — Risk metrics engine for Mutual Fund Portfolio Analyzer
Computes: Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, Sector-wise Beta
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

TRADING_MONTHS = 12


def _fetch_benchmark_returns(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """Fetch monthly returns for a benchmark index from yfinance."""
    if not ticker:
        return None
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        monthly = df["Close"].resample("ME").last()
        returns = monthly.pct_change().dropna()
        if len(returns.shape) > 1:
            returns = returns.squeeze()
        return returns
    except Exception:
        return None


def _nav_series_to_monthly_returns(nav_series: list[dict]) -> Optional[pd.Series]:
    """
    Convert a list of {date, nav} dicts into a monthly return Series.
    nav_series: [{"date": "2022-01-10", "nav": 45.23}, ...]
    """
    if not nav_series or len(nav_series) < 3:
        return None
    try:
        df = pd.DataFrame(nav_series)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        monthly = df["nav"].resample("ME").last().dropna()
        returns = monthly.pct_change().dropna()
        if len(returns) < 3:
            return None
        return returns
    except Exception:
        return None


def _transactions_to_nav_series(transactions: list) -> list[dict]:
    """
    Build a pseudo NAV series from transaction NAVs for a holding.
    Groups by month using weighted average NAV where available.
    """
    records = []
    for t in transactions:
        if t.get("nav") and t.get("date"):
            records.append({"date": t["date"], "nav": t["nav"]})
    return records


def compute_volatility(monthly_returns: pd.Series) -> Optional[float]:
    """Annualised volatility (standard deviation of monthly returns × √12)."""
    if monthly_returns is None or len(monthly_returns) < 3:
        return None
    return round(float(monthly_returns.std() * np.sqrt(TRADING_MONTHS) * 100), 2)


def compute_sharpe(monthly_returns: pd.Series, risk_free_monthly: float) -> Optional[float]:
    """Sharpe Ratio = (mean monthly return - risk-free monthly) / std × √12"""
    if monthly_returns is None or len(monthly_returns) < 3:
        return None
    excess = monthly_returns - risk_free_monthly
    if monthly_returns.std() == 0:
        return None
    sharpe = (excess.mean() / monthly_returns.std()) * np.sqrt(TRADING_MONTHS)
    return round(float(sharpe), 2)


def compute_sortino(monthly_returns: pd.Series, risk_free_monthly: float) -> Optional[float]:
    """Sortino Ratio = (mean monthly return - risk-free) / downside std × √12"""
    if monthly_returns is None or len(monthly_returns) < 3:
        return None
    excess = monthly_returns - risk_free_monthly
    downside = monthly_returns[monthly_returns < risk_free_monthly]
    if len(downside) < 2:
        return None
    downside_std = downside.std()
    if downside_std == 0:
        return None
    sortino = (excess.mean() / downside_std) * np.sqrt(TRADING_MONTHS)
    return round(float(sortino), 2)


def compute_max_drawdown(monthly_returns: pd.Series) -> Optional[float]:
    """Max Drawdown = largest peak-to-trough loss from cumulative wealth index."""
    if monthly_returns is None or len(monthly_returns) < 3:
        return None
    wealth = (1 + monthly_returns).cumprod()
    rolling_max = wealth.cummax()
    drawdown = (wealth - rolling_max) / rolling_max
    max_dd = drawdown.min()
    return round(float(max_dd * 100), 2)


def compute_beta(fund_returns: pd.Series, benchmark_returns: pd.Series) -> Optional[float]:
    """
    Beta = Cov(fund, benchmark) / Var(benchmark)
    Aligns series by date before computing.
    """
    if fund_returns is None or benchmark_returns is None:
        return None
    # Align on common dates
    combined = pd.DataFrame({"fund": fund_returns, "bench": benchmark_returns}).dropna()
    if len(combined) < 6:
        return None
    cov_matrix = np.cov(combined["fund"].values, combined["bench"].values)
    bench_var = cov_matrix[1][1]
    if bench_var == 0:
        return None
    beta = cov_matrix[0][1] / bench_var
    return round(float(beta), 3)


def compute_alpha(fund_returns: pd.Series, benchmark_returns: pd.Series, risk_free_monthly: float) -> Optional[float]:
    """
    Jensen's Alpha (annualised) = (fund_mean - rf) - beta * (bench_mean - rf), × 12
    """
    if fund_returns is None or benchmark_returns is None:
        return None
    combined = pd.DataFrame({"fund": fund_returns, "bench": benchmark_returns}).dropna()
    if len(combined) < 6:
        return None
    cov_matrix = np.cov(combined["fund"].values, combined["bench"].values)
    bench_var = cov_matrix[1][1]
    if bench_var == 0:
        return None
    beta = cov_matrix[0][1] / bench_var
    fund_mean  = combined["fund"].mean()
    bench_mean = combined["bench"].mean()
    alpha_monthly = (fund_mean - risk_free_monthly) - beta * (bench_mean - risk_free_monthly)
    return round(float(alpha_monthly * TRADING_MONTHS * 100), 2)


def compute_std_dev(monthly_returns: pd.Series) -> Optional[float]:
    """Annualised standard deviation (identical to volatility — alias for clarity in UI)."""
    return compute_volatility(monthly_returns)


# ── Category-aware metric thresholds ─────────────────────────────────────────
_THRESHOLDS = {
    "large cap": {"sharpe_good": 0.80, "sharpe_excel": 1.3, "vol_low": 15, "vol_high": 22, "dd_mild": 15, "dd_severe": 30},
    "mid cap":   {"sharpe_good": 0.70, "sharpe_excel": 1.2, "vol_low": 20, "vol_high": 30, "dd_mild": 20, "dd_severe": 38},
    "small cap": {"sharpe_good": 0.60, "sharpe_excel": 1.1, "vol_low": 25, "vol_high": 38, "dd_mild": 25, "dd_severe": 45},
    "flexi cap": {"sharpe_good": 0.75, "sharpe_excel": 1.2, "vol_low": 17, "vol_high": 26, "dd_mild": 18, "dd_severe": 32},
    "multi cap": {"sharpe_good": 0.75, "sharpe_excel": 1.2, "vol_low": 17, "vol_high": 26, "dd_mild": 18, "dd_severe": 32},
    "sectoral":  {"sharpe_good": 0.50, "sharpe_excel": 1.0, "vol_low": 25, "vol_high": 38, "dd_mild": 25, "dd_severe": 45},
    "thematic":  {"sharpe_good": 0.50, "sharpe_excel": 1.0, "vol_low": 25, "vol_high": 38, "dd_mild": 25, "dd_severe": 45},
    "debt":      {"sharpe_good": 0.30, "sharpe_excel": 0.7, "vol_low": 3,  "vol_high": 8,  "dd_mild": 3,  "dd_severe": 8},
    "liquid":    {"sharpe_good": 0.20, "sharpe_excel": 0.5, "vol_low": 1,  "vol_high": 3,  "dd_mild": 1,  "dd_severe": 3},
    "default":   {"sharpe_good": 0.80, "sharpe_excel": 1.3, "vol_low": 18, "vol_high": 28, "dd_mild": 20, "dd_severe": 35},
}


def _get_thresholds(category: str) -> dict:
    cat = (category or "").lower()
    for key in _THRESHOLDS:
        if key in cat:
            return _THRESHOLDS[key]
    return _THRESHOLDS["default"]


def _sharpe_rating(sharpe: Optional[float], thr: dict) -> str:
    if sharpe is None:
        return "N/A"
    if sharpe >= thr["sharpe_excel"]:
        return "Excellent"
    if sharpe >= thr["sharpe_good"]:
        return "Good"
    if sharpe >= 0:
        return "Average"
    return "Poor"


def _vol_rating(vol: Optional[float], thr: dict) -> str:
    if vol is None:
        return "N/A"
    if vol <= thr["vol_low"]:
        return "Low"
    if vol <= thr["vol_high"]:
        return "Moderate"
    return "High"


def _dd_rating(dd: Optional[float], thr: dict) -> str:
    if dd is None:
        return "N/A"
    abs_dd = abs(dd)
    if abs_dd <= thr["dd_mild"]:
        return "Mild"
    if abs_dd <= thr["dd_severe"]:
        return "Moderate"
    return "Severe"


def _beta_rating(beta: Optional[float]) -> str:
    if beta is None:
        return "N/A"
    if beta < 0.8:
        return "Defensive"
    if beta <= 1.2:
        return "Neutral"
    return "Aggressive"


def _sortino_rating(sortino: Optional[float], thr: dict) -> str:
    if sortino is None:
        return "N/A"
    if sortino >= thr["sharpe_excel"] * 1.2:
        return "Excellent"
    if sortino >= thr["sharpe_good"] * 1.1:
        return "Good"
    if sortino >= 0:
        return "Average"
    return "Poor"


def _build_interpretation(sharpe_r: str, vol_r: str, dd_r: str, beta_r: str, category: str) -> str:
    """Build a single plain-English sentence summarising the fund's risk profile."""
    sentences = []
    is_debt = "debt" in (category or "").lower() or "liquid" in (category or "").lower()

    # Volatility
    if vol_r == "Low":
        sentences.append("very stable price movement")
    elif vol_r == "High":
        sentences.append("high price swings")
    else:
        sentences.append("moderate price swings")

    # Sharpe
    if sharpe_r == "Excellent":
        sentences.append("exceptional risk-adjusted returns")
    elif sharpe_r == "Good":
        sentences.append("good risk-adjusted returns")
    elif sharpe_r == "Poor":
        sentences.append("returns below the risk-free rate — consider reviewing this holding")
    else:
        sentences.append("average risk-adjusted returns")

    # Drawdown
    if dd_r == "Mild":
        sentences.append("limited downside in past crashes")
    elif dd_r == "Severe":
        sentences.append("steep losses during market downturns")
    else:
        sentences.append("moderate drawdown history")

    # Beta (only for equity)
    if not is_debt and beta_r != "N/A":
        if beta_r == "Defensive":
            sentences.append("moves less aggressively than its benchmark")
        elif beta_r == "Aggressive":
            sentences.append("amplifies market moves — suitable for high-risk appetite")

    return ", ".join(sentences[:3]).capitalize() + "."


def _interpret_metrics(metrics: dict, category: str) -> dict:
    """Return rating fields and an interpretation note for a fund's risk metrics."""
    thr = _get_thresholds(category)
    sharpe_r = _sharpe_rating(metrics.get("sharpe_ratio"), thr)
    sortino_r = _sortino_rating(metrics.get("sortino_ratio"), thr)
    vol_r = _vol_rating(metrics.get("volatility_pct"), thr)
    dd_r = _dd_rating(metrics.get("max_drawdown_pct"), thr)
    beta_r = _beta_rating(metrics.get("beta"))
    note = _build_interpretation(sharpe_r, vol_r, dd_r, beta_r, category)
    return {
        "sharpe_rating": sharpe_r,
        "sortino_rating": sortino_r,
        "volatility_rating": vol_r,
        "drawdown_rating": dd_r,
        "beta_rating": beta_r,
        "interpretation_note": note,
    }


def _compute_peer_rankings(results: dict) -> dict:
    """
    For each fund, compute peer rank within its category for Sharpe and Sortino.
    Mutates results in-place and returns it.
    """
    from collections import defaultdict

    # Group ISINs by category
    cat_groups: dict[str, list] = defaultdict(list)
    for isin, m in results.items():
        cat_groups[m.get("category", "Other")].append(isin)

    for cat, isins in cat_groups.items():
        n = len(isins)
        if n < 2:
            for isin in isins:
                results[isin]["peer_rank"] = f"Only {cat} fund"
            continue

        # Sort by Sharpe descending (None last)
        sharpe_sorted = sorted(
            isins,
            key=lambda i: results[i].get("sharpe_ratio") if results[i].get("sharpe_ratio") is not None else float("-inf"),
            reverse=True,
        )
        for rank, isin in enumerate(sharpe_sorted, 1):
            results[isin]["peer_rank"] = f"#{rank} of {n} {cat} funds"

    return results


def compute_risk_metrics(holdings: list) -> dict:
    """
    Main entry point. For each holding, compute all risk metrics.
    Returns per-fund metrics + portfolio aggregate.

    holdings: list of holding dicts from parser.py
    """
    from scrapers.morningstar import get_rbi_repo_rate
    results = {}
    all_fund_returns = []

    # Dynamically fetch RBI India Policy Repo Rate for exact math
    rfr_annual = get_rbi_repo_rate() / 100.0
    rfr_monthly = (1 + rfr_annual) ** (1 / 12) - 1

    # Date range: 3 years back
    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    benchmark_cache = {}  # ticker → returns, avoid duplicate yfinance calls

    for holding in holdings:
        isin = holding.get("isin", "")
        name = holding.get("name", "Unknown")
        benchmark_ticker = holding.get("benchmark")  # None for debt funds
        category = holding.get("category", "")

        # Read NAV arrays linearly from the normalized SQLite Database
        from data.database import get_nav_series
        nav_series = get_nav_series(holding.get("isin", ""))
        
        # Fallback to sparse transactions if entirely unavailable
        if not nav_series or len(nav_series) < 3:
            nav_series = _transactions_to_nav_series(holding.get("transactions", []))
        
        fund_returns = _nav_series_to_monthly_returns(nav_series)
        
        # Enforce strict 3-Year (36 month) trailing window for all risk metric math
        if fund_returns is not None:
            fund_returns = fund_returns.tail(36)
            if len(fund_returns) < 3:
                fund_returns = None

        volatility = compute_volatility(fund_returns)
        sharpe = compute_sharpe(fund_returns, rfr_monthly)
        sortino = compute_sortino(fund_returns, rfr_monthly)
        max_dd = compute_max_drawdown(fund_returns)

        # Sector-wise Beta
        beta = None
        beta_benchmark_name = None
        benchmark_label = _ticker_to_label(benchmark_ticker)

        if fund_returns is not None and benchmark_ticker:
            # Cache benchmark data
            if benchmark_ticker not in benchmark_cache:
                benchmark_cache[benchmark_ticker] = _fetch_benchmark_returns(
                    benchmark_ticker, start_date, end_date
                )
            bench_returns = benchmark_cache[benchmark_ticker]
            beta = compute_beta(fund_returns, bench_returns)
            beta_benchmark_name = benchmark_label

        if fund_returns is not None:
            all_fund_returns.append(fund_returns)

        fund_metrics = {
            "name": name,
            "category": category,
            "beta": beta,
            "beta_benchmark": beta_benchmark_name,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd,
            "data_points": len(fund_returns) if fund_returns is not None else 0,
            "note": "Beta not applicable for debt/liquid funds" if not benchmark_ticker else None,
        }

        # Attach interpretation ratings
        interp = _interpret_metrics(fund_metrics, category)
        fund_metrics.update(interp)
        results[isin] = fund_metrics

    # ── Peer rankings across portfolio funds in same category ─────────────────
    results = _compute_peer_rankings(results)

    # ── Portfolio-level aggregate metrics ─────────────────────────────────────
    portfolio_metrics = {}
    if all_fund_returns:
        try:
            combined_port = pd.concat(all_fund_returns, axis=1).mean(axis=1).dropna()
            portfolio_metrics = {
                "volatility_pct": compute_volatility(combined_port),
                "sharpe_ratio": compute_sharpe(combined_port, rfr_monthly),
                "sortino_ratio": compute_sortino(combined_port, rfr_monthly),
                "max_drawdown_pct": compute_max_drawdown(combined_port),
            }
            # Portfolio beta vs Nifty 50
            if "^NSEI" not in benchmark_cache:
                benchmark_cache["^NSEI"] = _fetch_benchmark_returns("^NSEI", start_date, end_date)
            nifty_returns = benchmark_cache.get("^NSEI")
            portfolio_metrics["beta_vs_nifty50"] = compute_beta(combined_port, nifty_returns)
            # Portfolio-level interpretation
            port_interp = _interpret_metrics(portfolio_metrics, "default")
            portfolio_metrics.update(port_interp)
        except Exception:
            portfolio_metrics = {}

    return {
        "per_fund": results,
        "portfolio": portfolio_metrics,
    }


def _ticker_to_label(ticker: Optional[str]) -> Optional[str]:
    """Convert yfinance ticker to a human-readable benchmark name."""
    mapping = {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "Nifty Bank",
        "^NSEMDCP50": "Nifty Midcap 50",
        "MID150BEES.NS": "Nifty Midcap 150 Total Return Index (TRI)",
        "HDFCSML250.NS": "Nifty Smallcap 250 TRI",
        "^CNXIT": "Nifty IT",
        "^CNXPHARMA": "Nifty Pharma",
        "^CRSLDX": "NIFTY 500 TRI",
        "^CNXCMDT": "Nifty Commodities",
        "LIQUIDCASE.NS": "NIFTY Liquid Index",
        "^CNX100": "NIFTY 100 TRI",
        "BSE-100.BO": "S&P BSE 100 - TRI",
        "GOLDBEES.NS": "Domestic Price of Physical Gold",
    }
    return mapping.get(ticker, ticker) if ticker else None
