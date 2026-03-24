"""
advanced_analytics.py — Advanced metrics including Taxes, Exit Loads, and Goal-based strategy.

Provides:
  1. Exit Load detection (based on a curated category lookup table).
  2. FIFO Capital Gains calculation (Short Term vs Long Term).
  3. Goal-based strategy engine (target wealth, timeline, SIP recommendation).
"""

import math
import datetime
from typing import Optional

# ── General Curated Constants ──────────────────────────────────────────────────
# Standard taxation rules as of latest Indian budget (approx):
STCG_TAX_RATE = 0.20   # 20% Short Term Capital Gains on Equity
LTCG_TAX_RATE = 0.125  # 12.5% Long Term Capital Gains on Equity (> 1.25L exempt)
LTCG_EXEMPTION = 125_000

DEBT_TAX_RATE = 0.30   # Add to income slab (assuming 30% for conservative estimate)

# Exit Load lookup logic (since no free API provides accurate scheme-level loads)
# Format: { 'keywords': [list of lower case keywords in category/name], 'load_pct': % penalty, 'days': threshold_days }
EXIT_LOAD_RULES = [
    {"keywords": ["liquid", "overnight", "money market"], "load_pct": 0.0, "days": 0},
    {"keywords": ["elss", "tax saver"], "load_pct": 0.0, "days": 1095}, # Lock-in, but technically no exit load after lock-in
    {"keywords": ["debt", "gilt", "corporate bond", "banking and psu"], "load_pct": 0.0, "days": 0}, # Simplified, many are zero or very minor (e.g., 0.25% for 30 days)
    # Default equity rule (mostly 1% for 1 year)
    {"keywords": ["equity", "flexi cap", "large cap", "mid cap", "small cap", "value", "contra", "focused", "sectoral", "thematic", "index", "arbitrage", "hybrid", "balanced"], "load_pct": 1.0, "days": 365},
]

def get_exit_load_rule(category: str, fund_name: str) -> dict:
    """Returns the expected exit load rule based on fund category or name."""
    cat_lower = (category or "").lower()
    name_lower = (fund_name or "").lower()
    
    for rule in EXIT_LOAD_RULES:
        for kw in rule["keywords"]:
            if kw in cat_lower or kw in name_lower:
                return rule
                
    # Fallback default: assume 1% for 1 year to be safe
    return {"load_pct": 1.0, "days": 365}


# ── Tax & Gains Engine ─────────────────────────────────────────────────────────

def calculate_taxes_and_loads(transactions: list[dict], current_navs: dict[str, float]) -> dict:
    """
    Simulates selling the entire portfolio today.
    Calculates estimated exit loads and STCG/LTCG taxes using FIFO (First-In, First-Out).
    """
    today = datetime.date.today()
    results = {
        "total_exit_load": 0.0,
        "total_stcg_equity": 0.0,
        "total_ltcg_equity": 0.0,
        "total_gains_debt": 0.0,
        "estimated_tax": 0.0,
        "per_fund": {}
    }

    # Group valid buy/sip transactions by scheme
    # Note: To be perfectly accurate, we need to subtract SELL transactions from the earliest BUYs (FIFO).
    # Since CAS parser returns current active holdings, we will approximate by taking remaining units
    # and assigning them to the most recent BUYs (LIFO for matching remaining units, but logically FIFO for what was sold).
    #
    # Actually, the simplest approach for "Current Portfolio Tax Liability" is:
    # 1. Take all BUY/SIPs.
    # 2. Replay history: BUY adds to queue, SELL removes from queue (oldest first).
    # 3. Whatever is left in the queue is the current holding.
    # 4. Compute tax on what is left if sold today.
    
    queues = {} # scheme -> list of {date, units, nav, type}
    
    # Sort transactions chronologically
    def parse_date(date_str):
        try:
            return datetime.datetime.strptime(date_str, "%d-%b-%Y").date()
        except:
            return today
            
    txns = [t for t in transactions if t.get("date")]
    txns.sort(key=lambda x: parse_date(x["date"]))
    
    for txn in txns:
        scheme = txn["scheme_name"]
        if scheme not in queues:
            queues[scheme] = []
            
        t_type = txn["type"]
        units = txn.get("units", 0)
        
        if units > 0: # Buy, SIP, Switch In, DivReinvest
            queues[scheme].append({
                "date": parse_date(txn["date"]),
                "units": units,
                "cost_nav": txn["nav"],
                "category": txn.get("category", "") # We might not have category in txn, need it mapped below
            })
        elif units < 0: # Sell, Switch Out
            units_to_sell = abs(units)
            # FIFO removal
            i = 0
            while units_to_sell > 0.001 and i < len(queues[scheme]):
                if queues[scheme][i]["units"] <= units_to_sell:
                    units_to_sell -= queues[scheme][i]["units"]
                    queues[scheme][i]["units"] = 0
                    i += 1
                else:
                    queues[scheme][i]["units"] -= units_to_sell
                    units_to_sell = 0
            
            # Clean up empty queue items
            queues[scheme] = [q for q in queues[scheme] if q["units"] > 0.001]

    # Now we have the exact units held today and their purchase dates/costs.
    for scheme, queue in queues.items():
        if not queue:
            continue
            
        current_nav = current_navs.get(scheme, queue[-1]["cost_nav"])
        # Assume category from the first item if available, otherwise guess from name
        category = queue[0].get("category", "")
        rule = get_exit_load_rule(category, scheme)
        
        is_debt = "debt" in category.lower() or "liquid" in category.lower() or "money market" in category.lower()
        
        fund_exit_load = 0.0
        fund_stcg = 0.0
        fund_ltcg = 0.0
        fund_debt_gains = 0.0
        
        for p in queue:
            days_held = (today - p["date"]).days
            current_value = p["units"] * current_nav
            cost_value = p["units"] * p["cost_nav"]
            gain = current_value - cost_value
            
            # Exit Load
            if rule["days"] > 0 and days_held < rule["days"]:
                fund_exit_load += current_value * (rule["load_pct"] / 100.0)
                
            # Taxes
            if is_debt:
                # Debt mutual funds bought after Mar 2023 are taxed at slab rates regardless of holding period
                fund_debt_gains += gain
            else:
                # Equity taxation
                if days_held < 365:
                    if gain > 0: fund_stcg += gain
                else:
                    if gain > 0: fund_ltcg += gain
                    
        results["per_fund"][scheme] = {
            "exit_load": round(fund_exit_load, 2),
            "stcg": round(fund_stcg, 2),
            "ltcg": round(fund_ltcg, 2),
            "debt_gains": round(fund_debt_gains, 2),
            "load_rule": rule
        }
        
        results["total_exit_load"] += fund_exit_load
        results["total_stcg_equity"] += fund_stcg
        results["total_ltcg_equity"] += fund_ltcg
        results["total_gains_debt"] += fund_debt_gains

    # Final Tax Calculation
    tax_stcg = results["total_stcg_equity"] * STCG_TAX_RATE
    taxable_ltcg = max(0, results["total_ltcg_equity"] - LTCG_EXEMPTION)
    tax_ltcg = taxable_ltcg * LTCG_TAX_RATE
    tax_debt = results["total_gains_debt"] * DEBT_TAX_RATE
    
    results["estimated_tax"] = tax_stcg + tax_ltcg + tax_debt
    
    # Round numerical outputs
    for k in ["total_exit_load", "total_stcg_equity", "total_ltcg_equity", "total_gains_debt", "estimated_tax"]:
        results[k] = round(results[k], 2)
        
    return results

# ── Goal-Based Engine ──────────────────────────────────────────────────────────

def calculate_goal_strategy(target_amount: float, horizon_years: int, current_value: float = 0.0) -> dict:
    """
    Recommends a strategy to reach a target goal amount within horizon_years.
    """
    if horizon_years <= 0:
        return {"error": "Horizon must be > 0"}
        
    # Baseline expected returns based on horizon
    if horizon_years <= 3:
        asset_mix = {"equity": 20, "hybrid": 30, "debt": 50}
        exp_return_pct = 7.5
        risk_profile = "Conservative"
        suggestion = "Focus on capital preservation. Use Ultra-Short Duration, Liquid, and Arbitrage funds."
    elif horizon_years <= 7:
        asset_mix = {"equity": 60, "hybrid": 20, "debt": 20}
        exp_return_pct = 10.0
        risk_profile = "Moderate"
        suggestion = "Balanced approach. Use Flexi Cap, Balanced Advantage, and Corporate Bond funds."
    else:
        asset_mix = {"equity": 90, "hybrid": 0, "debt": 10}
        exp_return_pct = 12.0
        risk_profile = "Aggressive"
        suggestion = "Growth-oriented. Use Flexi Cap, Mid Cap, and Small Cap funds. Benefit from long-term compounding."
        
    r = exp_return_pct / 100.0
    n_months = horizon_years * 12
    r_monthly = r / 12.0
    
    # Value of current portfolio at the end of horizon
    future_value_current = current_value * math.pow(1 + r, horizon_years)
    
    # Shortfall to cover with SIPs
    shortfall = max(0, target_amount - future_value_current)
    
    # SIP required formula: FV = P * [ ( (1+r)^n - 1 ) / r ] * (1+r)
    # So P = FV / [ ( (1+r)^n - 1 ) / r * (1+r) ]
    if shortfall > 0:
        numerator = shortfall * r_monthly
        denominator = (math.pow(1 + r_monthly, n_months) - 1) * (1 + r_monthly)
        required_sip = numerator / denominator
    else:
        required_sip = 0
        
    # ELSS Recommendation (Tax saving under 80C)
    # Max limit is 1.5L / year = 12500 / month
    elss_suggestion = 0.0
    if asset_mix["equity"] > 0 and required_sip > 0:
        elss_suggestion = min(12500, required_sip * (asset_mix["equity"] / 100.0))

    return {
        "target_amount": target_amount,
        "horizon_years": horizon_years,
        "current_value": current_value,
        "future_value_of_current": round(future_value_current, 0),
        "shortfall": round(shortfall, 0),
        "required_sip_monthly": round(required_sip, 0),
        "assumed_return_pct": exp_return_pct,
        "risk_profile": risk_profile,
        "asset_allocation_recommendation": asset_mix,
        "fund_strategy": suggestion,
        "tax_saving_elss_sip": round(elss_suggestion, 0)
    }

# ── Projections & Simulations ──────────────────────────────────────────────────

def calculate_sip_step_up(initial_sip: float, step_up_pct: float, horizon_years: int, expected_return_pct: float) -> dict:
    """
    Calculates future value of a SIP that increases by a fixed percentage every year.
    Returns the final amount and year-by-year cashflow.
    """
    if horizon_years <= 0 or initial_sip <= 0:
        return {"error": "Invalid SIP or Horizon"}
        
    r_monthly = (expected_return_pct / 100.0) / 12.0
    step_up_rate = step_up_pct / 100.0
    
    current_sip = initial_sip
    total_invested = 0.0
    fv = 0.0
    year_data = []

    for year in range(1, horizon_years + 1):
        year_invested = 0.0
        for month in range(12):
            total_invested += current_sip
            year_invested += current_sip
            # Compounding
            fv = (fv + current_sip) * (1 + r_monthly)
            
        year_data.append({
            "year": year,
            "monthly_sip": round(current_sip, 0),
            "total_invested": round(total_invested, 0),
            "future_value": round(fv, 0)
        })
        
        # Apply annual step up
        current_sip = current_sip * (1 + step_up_rate)
        
    return {
        "initial_sip": initial_sip,
        "step_up_pct": step_up_pct,
        "expected_return_pct": expected_return_pct,
        "horizon_years": horizon_years,
        "total_invested": round(total_invested, 0),
        "future_value": round(fv, 0),
        "wealth_gain": round(fv - total_invested, 0),
        "yearly_breakdown": year_data
    }

def run_monte_carlo_simulation(current_value: float, monthly_sip: float, horizon_years: int, 
                               mean_return_pct: float, volatility_pct: float, num_paths: int = 500) -> dict:
    """
    Runs a Monte Carlo simulation (geometric Brownian motion) for portfolio growth.
    Returns the median, 10th percentile (bear market), and 90th percentile (bull market) outcomes.
    """
    import random
    
    if horizon_years <= 0:
        return {"error": "Horizon must be > 0"}

    months = horizon_years * 12
    # Convert annual mean and vol to monthly
    mu = (mean_return_pct / 100.0) / 12.0
    sigma = (volatility_pct / 100.0) / math.sqrt(12)
    
    final_values = []
    
    # Generate 3 representative paths to return for plotting (best, median, worst scenarios roughly)
    paths = []
    
    for i in range(num_paths):
        value = current_value
        path = [value]
        
        for m in range(months):
            # GBM step: S_t = S_{t-1} * exp((mu - sigma^2/2) + sigma * Z)
            # Simplified for continuous additions:
            z = random.gauss(0, 1)
            monthly_return = mu + sigma * z
            value = value * (1 + monthly_return) + monthly_sip
            path.append(value)
            
        final_values.append(value)
        if i < 3: # Keep a few random paths just to have variance lines, but later we pick percentiles
            paths.append(path)
            
    final_values.sort()
    
    p10 = final_values[int(num_paths * 0.10)]
    p50 = final_values[int(num_paths * 0.50)]
    p90 = final_values[int(num_paths * 0.90)]
    
    # Let's generate synthetic "smooth" curves for the percentiles to send back to the UI for plotting
    median_path = [current_value]
    bear_path = [current_value]
    bull_path = [current_value]
    
    # Approximate monthly rates for the synthetic curves
    r_med = math.pow(p50 / current_value, 1/months) - 1 if current_value > 0 and monthly_sip == 0 else mu
    # If there are SIPs involved, calculating the exact implied synthetic rate is iterative, 
    # so we'll just use the raw mean/vol for a deterministic curve
    
    val_med = current_value
    val_bear = current_value
    val_bull = current_value
    
    for m in range(months):
        val_med = val_med * (1 + mu) + monthly_sip
        val_bear = val_bear * (1 + (mu - sigma * 1.28)) + monthly_sip # roughly 10th percentile Z
        val_bull = val_bull * (1 + (mu + sigma * 1.28)) + monthly_sip # roughly 90th percentile Z
        median_path.append(round(val_med, 0))
        bear_path.append(round(val_bear, 0))
        bull_path.append(round(val_bull, 0))

    return {
        "current_value": current_value,
        "monthly_sip": monthly_sip,
        "horizon_years": horizon_years,
        "mean_return_pct": mean_return_pct,
        "volatility_pct": volatility_pct,
        "paths_simulated": num_paths,
        "outcomes": {
            "median": round(p50, 0),
            "bear_market_10th": round(p10, 0),
            "bull_market_90th": round(p90, 0)
        },
        "plot_data": {
            "months": list(range(months + 1)),
            "median_path": median_path,
            "bear_path": bear_path,
            "bull_path": bull_path
        }
    }


def calculate_stress_test(current_value: float, equity_allocation_pct: float) -> dict:
    """
    Estimates portfolio drawdown during major historical market crashes based on equity allocation.
    """
    if current_value <= 0:
        return {"error": "Portfolio value is 0. Please upload a portfolio to see stress tests."}
        
    equity_fraction = equity_allocation_pct / 100.0
    debt_fraction = 1.0 - equity_fraction
    
    # Historical Crash Data (Equity drop %, Debt drop/gain %)
    # Approximations for Indian Market (Nifty 50 vs High Quality Debt)
    scenarios = [
        {
            "name": "COVID-19 Crash (Mar 2020)",
            "equity_drop": -38.0,
            "debt_drop": 2.0, # Debt actually rallied as rates were cut
            "description": "Global pandemic shock. Rapid but short-lived V-shaped recovery."
        },
        {
            "name": "Global Financial Crisis (2008)",
            "equity_drop": -60.0,
            "debt_drop": 5.0,
            "description": "Systemic banking crisis. Severe drawn-out bear market."
        },
        {
            "name": "Dot Com Bubble (2000-2001)",
            "equity_drop": -50.0,
            "debt_drop": 8.0,
            "description": "Tech stock collapse. Prolonged period of negative returns."
        },
        {
            "name": "Interest Rate Hike Shock",
            "equity_drop": -15.0,
            "debt_drop": -5.0,
            "description": "Sudden central bank tightening. Both bonds and equities decline simultaneously."
        }
    ]
    
    results = []
    
    for sc in scenarios:
        eq_val = current_value * equity_fraction
        dt_val = current_value * debt_fraction
        
        eq_new = eq_val * (1 + (sc["equity_drop"] / 100.0))
        dt_new = dt_val * (1 + (sc["debt_drop"] / 100.0))
        
        new_val = eq_new + dt_new
        loss = current_value - new_val
        blended_drop = (loss / current_value) * -100.0
        
        results.append({
            "scenario": sc["name"],
            "description": sc["description"],
            "portfolio_value_before": current_value,
            "portfolio_value_after": round(new_val, 0),
            "estimated_loss": round(loss, 0),
            "blended_drop_pct": round(blended_drop, 1)
        })
        
    return {
        "current_value": current_value,
        "equity_allocation_pct": equity_allocation_pct,
        "stress_tests": results
    }

# ── Portfolio Management & Cashflow ─────────────────────────────────────────────

def calculate_rebalance(holdings: list[dict], target_equity_pct: float) -> dict:
    """
    Given current holdings and a target equity allocation, suggests buy/sell 
    amounts to reach the target allocation with minimum transactions.
    """
    if not holdings:
        return {"error": "No holdings to rebalance"}
        
    target_eq_frac = target_equity_pct / 100.0
    
    # Categorize current value
    total_val = 0.0
    equity_val = 0.0
    debt_val = 0.0
    
    for h in holdings:
        val = h.get("current_value", 0)
        cat = (h.get("category") or "").lower()
        
        if val <= 0: continue
        total_val += val
        
        if "equity" in cat or "flexi" in cat or "cap" in cat or "sectoral" in cat:
            equity_val += val
        elif "hybrid" in cat or "balanced" in cat:
            # Approximate hybrid as 65% eq / 35% dt
            equity_val += val * 0.65
            debt_val += val * 0.35
        else:
            # Assume everything else is debt/liquid
            debt_val += val

    if total_val <= 0:
        return {"error": "Portfolio value is zero"}
        
    current_eq_frac = equity_val / total_val
    
    target_equity_val = total_val * target_eq_frac
    target_debt_val = total_val * (1.0 - target_eq_frac)
    
    diff_equity = target_equity_val - equity_val
    diff_debt = target_debt_val - debt_val
    
    # Action required? (Tolerance 2%)
    if abs(current_eq_frac - target_eq_frac) <= 0.02:
        return {
            "action_required": False,
            "message": "Portfolio is within 2% of target allocation. No rebalancing needed.",
            "current_equity_pct": round(current_eq_frac * 100, 1),
            "target_equity_pct": target_equity_pct,
            "actions": []
        }
        
    actions = []
    if diff_equity > 0:
        actions.append({"type": "BUY", "asset_class": "Equity", "amount": round(diff_equity, 0)})
        actions.append({"type": "SELL", "asset_class": "Debt/Liquid", "amount": round(abs(diff_debt), 0)})
        msg = f"Portfolio is UNDERWEIGHT in Equity. Move ₹{abs(round(diff_debt, 0)):,} from Debt to Equity."
    else:
        actions.append({"type": "SELL", "asset_class": "Equity", "amount": round(abs(diff_equity), 0)})
        actions.append({"type": "BUY", "asset_class": "Debt/Liquid", "amount": round(diff_debt, 0)})
        msg = f"Portfolio is OVERWEIGHT in Equity. Move ₹{abs(round(diff_equity, 0)):,} from Equity to Debt."
        
    return {
        "action_required": True,
        "message": msg,
        "current_equity_pct": round(current_eq_frac * 100, 1),
        "target_equity_pct": target_equity_pct,
        "deviation_pct": round(abs(current_eq_frac - target_eq_frac) * 100, 1),
        "actions": actions
    }


def calculate_dividend_cashflow(transactions: list[dict]) -> dict:
    """
    Extracts total dividends paid out or reinvested per year.
    """
    if not transactions:
        return {"error": "No transactions provided"}
        
    import collections
    
    div_by_year = collections.defaultdict(float)
    total_div = 0.0
    
    for t in transactions:
        t_type = t.get("type", "").upper()
        if "DIV" in t_type: # DIVR (Reinvest) or DIVP (Payout)
            # Typically for reinvestment, amount is present or can be inferred closely by units * nav
            amt = t.get("amount")
            if not amt and t.get("units") and t.get("nav"):
                amt = t["units"] * t["nav"]
                
            if amt and amt > 0:
                try:
                    date_obj = datetime.datetime.strptime(t["date"], "%d-%b-%Y")
                    year = date_obj.year
                    div_by_year[year] += amt
                    total_div += amt
                except Exception:
                    pass
                    
    # Format for charting
    years = sorted(list(div_by_year.keys()))
    amounts = [round(div_by_year[y], 0) for y in years]
    
    # Check if there are any dividends at all
    has_dividends = total_div > 0
    
    return {
        "has_dividends": has_dividends,
        "total_dividends_received": round(total_div, 0),
        "chart_data": {
            "years": years,
            "amounts": amounts
        }
    }
