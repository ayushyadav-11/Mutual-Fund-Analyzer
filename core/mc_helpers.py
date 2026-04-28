import re
from typing import Any, Optional

def _mc_to_float(value: Any) -> Optional[float]:
    """Best-effort numeric parsing for Moneycontrol payload values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned in {"--", "-", "N/A", "NA", "null", "None"}:
            return None
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("%", "")
        cleaned = cleaned.replace("₹", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None

def _mc_find_first(payload: Any, *keys: str) -> Any:
    """Recursively find the first non-empty value for any matching key."""
    def _norm(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    normalized = {_norm(k) for k in keys}

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in node.items():
                if _norm(key) in normalized and value not in (None, "", [], {}):
                    return value
            for value in node.values():
                found = _walk(value)
                if found not in (None, "", [], {}):
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _walk(item)
                if found not in (None, "", [], {}):
                    return found
        return None

    return _walk(payload)

def _mc_period_label(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    token = re.sub(r"[^a-z0-9]", "", str(raw).strip().lower())
    mapping = {
        "1y": "1Y", "1yr": "1Y", "1year": "1Y", "oneyear": "1Y",
        "1years": "1Y", "1yrs": "1Y",
        "3y": "3Y", "3yr": "3Y", "3year": "3Y", "threeyear": "3Y",
        "3years": "3Y", "3yrs": "3Y",
        "5y": "5Y", "5yr": "5Y", "5year": "5Y", "fiveyear": "5Y",
        "5years": "5Y", "5yrs": "5Y",
        "10y": "10Y", "10yr": "10Y", "10year": "10Y", "tenyear": "10Y",
        "10years": "10Y", "10yrs": "10Y",
    }
    return mapping.get(token)

def _mc_pick_period_value(periods: Any, values: Any, preferred: tuple[str, ...] = ("3Y", "5Y", "1Y", "10Y")) -> Optional[float]:
    if isinstance(periods, dict) and isinstance(values, dict):
        normalized_periods = {str(key).lower(): _mc_period_label(val) for key, val in periods.items()}
        for target in preferred:
            for raw_key, label in normalized_periods.items():
                if label == target and raw_key in values:
                    numeric = _mc_to_float(values.get(raw_key))
                    if numeric is not None:
                        return numeric
        for raw_key in periods.keys():
            numeric = _mc_to_float(values.get(raw_key))
            if numeric is not None:
                return numeric
        return None
    if not isinstance(periods, list) or not isinstance(values, list):
        return None
    labels = [_mc_period_label(p) for p in periods]
    for target in preferred:
        for idx, label in enumerate(labels):
            if label == target and idx < len(values):
                numeric = _mc_to_float(values[idx])
                if numeric is not None:
                    return numeric
    for value in values:
        numeric = _mc_to_float(value)
        if numeric is not None:
            return numeric
    return None

def _mc_extract_metric_value(payload: Any, *keys: str) -> Optional[float]:
    value = _mc_find_first(payload, *keys)
    numeric = _mc_to_float(value)
    if numeric is not None:
        return numeric
    if isinstance(payload, dict) and isinstance(value, (list, dict)):
        period_list = _mc_find_first(payload, "period", "tenure", "duration")
        numeric = _mc_pick_period_value(period_list, value)
        if numeric is not None:
            return numeric
    return None

def _mc_extract_cat_avg(payload: Any, *keys: str) -> Optional[float]:
    node = _mc_find_first(payload, *keys)
    if isinstance(node, dict):
        for p in ["3y", "5y", "1y", "10y", "3yr", "5yr", "1yr", "10yr"]:
            for prefix in ["cat_avg_", "category_avg_", "catavg"]:
                val = _mc_to_float(node.get(f"{prefix}{p}"))
                if val is not None:
                    return val
    return None

def _mc_extract_period_returns(payload: Any) -> tuple[dict, dict]:
    """Extract fund and benchmark/category returns from varied Moneycontrol shapes."""
    fund = {"1Y": None, "3Y": None, "5Y": None, "10Y": None}
    bench = {"1Y": None, "3Y": None, "5Y": None, "10Y": None}

    if isinstance(payload, dict):
        periods = _mc_find_first(payload, "period", "tenure", "duration")
        direct_returns = _mc_find_first(payload, "returns", "return", "fund_return", "fund returns")
        benchmark_direct = _mc_find_first(payload, "benchmark", "benchmark_return", "benchmark returns", "category_return", "category returns")
        if isinstance(periods, dict) and isinstance(direct_returns, dict):
            normalized_periods = {str(key).lower(): _mc_period_label(val) for key, val in periods.items()}
            for raw_key, label in normalized_periods.items():
                if label and fund[label] is None:
                    fund[label] = _mc_to_float(direct_returns.get(raw_key))
        if isinstance(periods, dict) and isinstance(benchmark_direct, dict):
            normalized_periods = {str(key).lower(): _mc_period_label(val) for key, val in periods.items()}
            for raw_key, label in normalized_periods.items():
                if label and bench[label] is None:
                    bench[label] = _mc_to_float(benchmark_direct.get(raw_key))
        if isinstance(periods, list) and isinstance(direct_returns, list):
            for period, value in zip(periods, direct_returns):
                label = _mc_period_label(period)
                if label and fund[label] is None:
                    fund[label] = _mc_to_float(value)
        if isinstance(periods, list) and isinstance(benchmark_direct, list):
            for period, value in zip(periods, benchmark_direct):
                label = _mc_period_label(period)
                if label and bench[label] is None:
                    bench[label] = _mc_to_float(value)

    def _assign(label: Optional[str], node: Any):
        if not label:
            return
        if isinstance(node, dict):
            if fund[label] is None:
                fund[label] = _mc_to_float(_mc_find_first(
                    node, "fund", "scheme", "return", "value", "returns", "annualised_return",
                    "annualized_return", "annualised returns", "annualized returns",
                    "fund_return", "fund return", "direct_return", "direct return"
                ))
            if bench[label] is None:
                bench[label] = _mc_to_float(_mc_find_first(
                    node, "benchmark", "benchmark_return", "benchmark_returns",
                    "benchmark return", "category", "category_return", "category_returns",
                    "category return", "benchmarkvalue", "benchmark value", "catavg", "cat_avg"
                ))
        else:
            if fund[label] is None:
                fund[label] = _mc_to_float(node)

    def _walk(node: Any):
        if isinstance(node, dict):
            period = _mc_period_label(_mc_find_first(node, "period", "tenure", "duration", "label", "name", "periodinvested", "period invested"))
            if period:
                _assign(period, node)
            for key, value in node.items():
                label = _mc_period_label(key)
                if label:
                    _assign(label, value)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return fund, bench

def _mc_extract_risk(mc_risk: Any, fallback_benchmark: Optional[str]) -> dict:
    return {
        "sharpe":          _mc_extract_metric_value(mc_risk, "sharpe_ratio", "sharpe", "sharpe ratio"),
        "sortino":         _mc_extract_metric_value(mc_risk, "sortino_ratio", "sortino", "sortino ratio"),
        "volatility":      _mc_extract_metric_value(
            mc_risk, "risk_std_dev", "std_dev", "standard_deviation", "standard deviation",
            "std deviation", "volatility"
        ),
        "beta":            _mc_extract_metric_value(mc_risk, "beta"),
        "alpha":           _mc_extract_metric_value(mc_risk, "alpha", "jensens_alpha", "jensen alpha", "jension alpha"),
        "max_drawdown_pct": _mc_extract_metric_value(
            mc_risk, "max_drawdown_pct", "max_drawdown", "drawdown", "max drawdown"
        ),
        "benchmark_name":  _mc_find_first(mc_risk, "benchmark_name", "benchmark", "benchmarklabel") or fallback_benchmark,
        # ── Category averages (from MoneyControl risk-metrics + overview) ──────────
        "cat_avg_sharpe":    _mc_extract_cat_avg(mc_risk, "sharpe_ratio", "sharpe", "sharpe ratio") or _mc_extract_metric_value(mc_risk, "cat_avg_sharpe", "category_avg_sharpe", "catAvgSharpe",
                                                       "category_sharpe", "cat_sharpe", "catSharpe"),
        "cat_avg_sortino":   _mc_extract_cat_avg(mc_risk, "sortino_ratio", "sortino", "sortino ratio") or _mc_extract_metric_value(mc_risk, "cat_avg_sortino", "category_avg_sortino", "catAvgSortino",
                                                       "category_sortino", "cat_sortino", "catSortino"),
        "cat_avg_beta":      _mc_extract_cat_avg(mc_risk, "beta") or _mc_extract_metric_value(mc_risk, "cat_avg_beta", "category_avg_beta", "catAvgBeta",
                                                       "category_beta", "cat_beta", "catBeta"),
        "cat_avg_std_dev":   _mc_extract_cat_avg(mc_risk, "risk_std_dev", "std_dev", "standard_deviation", "standard deviation", "std deviation", "volatility") or _mc_extract_metric_value(mc_risk, "cat_avg_std_dev", "category_avg_std_dev",
                                                       "catAvgStdDev", "category_std_dev", "cat_std_dev",
                                                       "catAvgVolatility", "category_volatility"),
        "cat_avg_alpha":     _mc_extract_cat_avg(mc_risk, "alpha", "jensens_alpha", "jensen alpha", "jension alpha") or _mc_extract_metric_value(mc_risk, "cat_avg_alpha", "category_avg_alpha", "catAvgAlpha",
                                                       "category_alpha", "cat_alpha"),
    }


def _mc_extract_fundamentals(mc_fund: Any) -> dict:
    return {
        "aum_cr": _mc_to_float(_mc_find_first(mc_fund, "aum_cr", "aum", "assets_under_management")),
        "expense_ratio": _mc_to_float(_mc_find_first(mc_fund, "expense_ratio", "exp_ratio", "expense")),
        "exit_load": _mc_find_first(mc_fund, "exit_load", "exitload"),
        "portfolio_turnover": _mc_to_float(_mc_find_first(mc_fund, "portfolio_turnover_pct", "portfolio_turnover", "turnover_ratio", "turnover")),
        "pe": _mc_extract_metric_value(
            mc_fund, "pe", "p_e", "p/e", "price_earnings", "price_to_earnings", "price_earning"
        ),
        "cat_avg_pe": _mc_extract_metric_value(
            mc_fund, "cat_avg_pe", "category_avg_pe", "category_average_pe", "catpe", "category_pe"
        ),
        "pb": _mc_extract_metric_value(
            mc_fund, "pb", "p_b", "p/b", "price_book", "price_to_book", "price_book_value"
        ),
        "cat_avg_pb": _mc_extract_metric_value(
            mc_fund, "cat_avg_pb", "category_avg_pb", "category_average_pb", "catpb", "category_pb"
        ),
        "price_sale": _mc_extract_metric_value(
            mc_fund, "price_sale", "price_sales", "price_to_sale", "price_to_sales", "priceSale", "ps"
        ),
        "cat_avg_price_sale": _mc_extract_metric_value(
            mc_fund,
            "cat_avg_price_sale",
            "category_avg_price_sale",
            "category_average_price_sale",
            "catAvgPriceSale",
            "category_price_sale",
        ),
        "price_cash_flow": _mc_extract_metric_value(
            mc_fund,
            "price_cash_flow",
            "price_cashflow",
            "price_to_cash_flow",
            "price_to_cashflow",
            "priceCashFlow",
            "pcf",
        ),
        "cat_avg_price_cash_flow": _mc_extract_metric_value(
            mc_fund,
            "cat_avg_price_cash_flow",
            "category_avg_price_cash_flow",
            "category_average_price_cash_flow",
            "catAvgPriceCashFlow",
            "category_price_cash_flow",
        ),
        "dividend_yield": _mc_extract_metric_value(
            mc_fund, "dividend_yield", "dividendYield", "div_yield", "dy"
        ),
        "cat_avg_dividend_yield": _mc_extract_metric_value(
            mc_fund,
            "cat_avg_dividend_yield",
            "category_avg_dividend_yield",
            "category_average_dividend_yield",
            "catAvgDividendYield",
            "category_dividend_yield",
        ),
        "roe": _mc_extract_metric_value(mc_fund, "roe", "ROE", "return_on_equity"),
        "cat_avg_roe": _mc_extract_metric_value(
            mc_fund, "cat_avg_roe", "category_avg_roe", "category_average_roe", "catAvgRoe", "category_roe"
        ),
    }

def _mc_extract_holdings(mc_portfolio: Any) -> list[dict]:
    holdings: list[dict] = []

    def _walk(node: Any):
        if isinstance(node, dict):
            name = _mc_find_first(node, "asset", "company", "holding", "security", "stock", "instrument", "name")
            weight = _mc_to_float(_mc_find_first(node, "weight", "holding_pct", "holdingpercent", "percent", "percentage", "value"))
            if isinstance(name, str) and weight is not None:
                holdings.append({"asset": name, "weight": round(weight, 2)})
            else:
                for value in node.values():
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(mc_portfolio)

    deduped: dict[str, float] = {}
    for item in holdings:
        asset = item["asset"].strip()
        if not asset:
            continue
        deduped.setdefault(asset, item["weight"])

    return [
        {"asset": asset, "weight": weight}
        for asset, weight in sorted(deduped.items(), key=lambda pair: pair[1], reverse=True)
    ]
