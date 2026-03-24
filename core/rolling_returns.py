import pandas as pd
from typing import Dict, Optional
import numpy as np

def compute_rolling_returns(nav_series: list[dict]) -> Dict[str, Optional[float]]:
    """
    Given a raw SQL NAV series [{'date': 'YYYY-MM-DD', 'nav': float}],
    computes exact trailing 1-Year, 3-Year, 5-Year and 10-Year CAGR returns.
    """
    if not nav_series or len(nav_series) < 2:
        return {"1Y": None, "3Y": None, "5Y": None, "10Y": None}

    df = pd.DataFrame(nav_series)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df = df.dropna(subset=["nav"])
    if len(df) < 5:
        return {"1Y": None, "3Y": None, "5Y": None, "10Y": None}

    latest_date = df.index[-1]
    latest_nav = df.iloc[-1]["nav"]

    results: Dict[str, Optional[float]] = {"1Y": None, "3Y": None, "5Y": None, "10Y": None}

    horizons = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}

    for label, years in horizons.items():
        target_date = latest_date - pd.DateOffset(years=years)
        if df.index[0] > target_date:
            continue
        closest_idx = df.index.get_indexer([target_date], method="nearest")[0]
        past_nav = df.iloc[closest_idx]["nav"]
        if past_nav <= 0:
            continue
        if years == 1:
            cagr = (latest_nav / past_nav) - 1
        else:
            cagr = ((latest_nav / past_nav) ** (1.0 / years)) - 1
        results[label] = round(float(cagr * 100), 2)

    return results
