import os
import json
import tempfile
from datetime import date, datetime
from typing import Optional
import casparser
from pyxirr import xirr

# ── Sector / Category → Benchmark mapping (for risk.py) ──────────────────────
CATEGORY_BENCHMARK = {
    "large cap": "^NSEI",          # Nifty 50
    "large-cap": "^NSEI",
    "index": "^NSEI",
    "flexi cap": "^CRSLDX",        # Nifty 500
    "flexicap": "^CRSLDX",
    "multi cap": "^CRSLDX",        # Nifty 500
    "mid cap": "^NSEMDCP50",       # Nifty Midcap 50
    "mid-cap": "^NSEMDCP50",
    "small cap": "HDFCSML250.NS",  # Nifty Smallcap 250
    "small-cap": "HDFCSML250.NS",
    "banking": "^NSEBANK",
    "bank": "^NSEBANK",
    "psu": "^NSEBANK",
    "it": "^CNXIT",
    "technology": "^CNXIT",
    "pharma": "^CNXPHARMA",
    "healthcare": "^CNXPHARMA",
    "hybrid": "^CRSLDX",
    "balanced": "^CRSLDX",
    "aggressive hybrid": "^CRSLDX",
    "balanced advantage": "^CRSLDX",
    "elss": "^CRSLDX",
    "tax saver": "^CRSLDX",
    "gold": "^CNXCMDT",
    "commodities": "^CNXCMDT",
    "liquid": "LIQUIDCASE.NS",
    # Debt — Beta not computed
    "debt": None,
    "overnight": None,
    "money market": None,
    "ultra short": None,
    "short duration": None,
    "corporate bond": None,
    "gilt": None,
}


def infer_category(category: str, scheme_name: str) -> str:
    """Infer the category based on category or scheme name."""
    if category:
        return category
        
    name = scheme_name.lower()
    if "midcap" in name or "mid cap" in name:
        return "Mid Cap Fund"
    if "smallcap" in name or "small cap" in name:
        return "Small Cap Fund"
    if "flexicap" in name or "flexi cap" in name:
        return "Flexi Cap Fund"
    if "multicap" in name or "multi cap" in name:
        return "Multi Cap Fund"
    if "largecap" in name or "large cap" in name or "bluechip" in name:
        return "Large Cap Fund"
    if "liquid" in name:
        return "Liquid Fund"
    if "debt" in name or "bond" in name or "gilt" in name:
        return "Debt Fund"
    if "elss" in name or "tax saver" in name:
        return "ELSS"
    if "index" in name or "nifty 50" in name:
        return "Index Fund"
    if "gold" in name:
        return "Commodities"
    if "hybrid" in name or "balanced" in name:
        return "Hybrid Fund"
    if "bank" in name:
        return "Banking Sector Fund"
    if "pharma" in name or "healthcare" in name:
        return "Pharma Sector Fund"
    if "tech" in name or "it " in name:
        return "Technology Sector Fund"
        
    return "Other"

# ── Fund Name → Exact Benchmark Ticker mapping ─────────────────────────────
FUND_BENCHMARK = {
    # Liquid / Debt
    "hdfc liquid fund": "LIQUIDCASE.NS",
    "parag parikh liquid fund": "LIQUIDCASE.NS",
    "axis liquid fund": "LIQUIDCASE.NS",
    
    # Large & Flexi Cap
    "aditya birla sun life large cap fund": "^CNX100",
    "mirae asset flexi cap fund": "^CNX100",
    "axis large cap fund": "BSE-100.BO",
    
    "parag parikh flexi cap fund": "^CRSLDX",
    "aditya birla sun life flexi cap fund": "^CRSLDX",
    "kotak flexicap fund": "^CRSLDX",
    
    # Mid & Small Cap
    "motilal oswal midcap fund": "MID150BEES.NS",
    "axis small cap fund": "HDFCSML250.NS",
    
    # Gold
    "aditya birla sun life gold fund": "GOLDBEES.NS"
}

def get_benchmark_ticker(scheme_name: str, category: str) -> Optional[str]:
    """Return the exact yfinance ticker for the benchmark matching the specific fund name or category."""
    name_lower = (scheme_name or "").lower()
    for key, ticker in FUND_BENCHMARK.items():
        if key in name_lower:
            return ticker
            
    # Fallback to category
    if not category:
        return "^NSEI"
    cat_lower = category.lower()
    for key, ticker in CATEGORY_BENCHMARK.items():
        if key in cat_lower:
            return ticker
    return "^NSEI"  # default fallback


def parse_cas(pdf_path: str, password: str = "") -> dict:
    """
    Parse a CAS PDF and return a clean, validated data dict.

    Steps:
    1. Run casparser on the PDF
    2. Merge multiple folios of the same ISIN
    3. Tag each transaction by type
    4. Validate unit balances (never negative)
    5. Handle missing NAVs gracefully
    6. Calculate XIRR
    """
    try:
        data_raw = casparser.read_cas_pdf(pdf_path, password=password, output="json")
        if isinstance(data_raw, str):
            data = json.loads(data_raw)
        elif hasattr(data_raw, "model_dump"):
            data = data_raw.model_dump()
        elif hasattr(data_raw, "dict"):
            data = data_raw.dict()
        else:
            data = dict(data_raw) if not isinstance(data_raw, dict) else data_raw
    except Exception as e:
        raise ValueError(f"Failed to parse CAS PDF: {str(e)}")

    investor_info = {
        "name": data.get("investor_info", {}).get("name", "Investor"),
        "email": "",
        "mobile": "",
        "address": "",
    }

    statement_period = {
        "from": str(data.get("statement_period", {}).get("from", "")),
        "to": str(data.get("statement_period", {}).get("to", "")),
    }

    # ── Build holdings dict keyed by ISIN ─────────────────────────────────────
    holdings = {}   # isin → holding dict
    all_transactions = []

    for folio in data.get("folios", []):
        for scheme in folio.get("schemes", []):
            isin = scheme.get("isin") or scheme.get("amfi") or scheme.get("scheme")
            if not isin:
                continue

            scheme_name = scheme.get("scheme", "Unknown Fund")
            category = scheme.get("category", "")
            rta_code = scheme.get("rta_code", "")
            advisor = scheme.get("advisor", "")
            open_units = float(scheme.get("open", 0) or 0)
            close_units = float(scheme.get("close", 0) or 0)
            close_calculated = float(scheme.get("close_calculated", close_units) or close_units)

            inferred_category = infer_category(category, scheme_name)
            if isin not in holdings:
                holdings[isin] = {
                    "isin": isin,
                    "name": scheme_name,
                    "category": inferred_category,
                    "rta_code": rta_code,
                    "advisor": advisor,
                    "units": 0.0,
                    "transactions": [],
                    "benchmark": get_benchmark_ticker(scheme_name, inferred_category),
                }

            # ── Process transactions ───────────────────────────────────────────
            running_units = holdings[isin]["units"] + open_units
            for txn in scheme.get("transactions", []):
                txn_date = txn.get("date")
                if isinstance(txn_date, (date, datetime)):
                    txn_date_str = str(txn_date)
                else:
                    txn_date_str = str(txn_date) if txn_date else None

                units = float(txn.get("units") or 0)
                nav = txn.get("nav")
                nav_val = float(nav) if nav is not None else None
                amount = float(txn.get("amount") or 0)
                raw_type = str(txn.get("type", "")).strip().upper()

                # Tag transaction type
                txn_type = _classify_txn(raw_type, units, amount)

                running_units += units
                # Validate: units should not go negative (catches parsing errors)
                if running_units < -0.001:
                    running_units = 0.0  # reset gracefully

                txn_record = {
                    "date": txn_date_str,
                    "type": txn_type,
                    "units": units,
                    "nav": nav_val,
                    "amount": amount,
                    "balance_units": round(running_units, 4),
                    "scheme_name": scheme_name,
                    "isin": isin,
                }
                holdings[isin]["transactions"].append(txn_record)
                all_transactions.append(txn_record)

            holdings[isin]["units"] = round(close_calculated, 4)

    # ── Calculate XIRR for each holding and portfolio ─────────────────────────
    portfolio_cashflows = []
    for isin, h in holdings.items():
        cf_dates, cf_amounts = _build_cashflows(h["transactions"], h["units"])
        if cf_dates and len(cf_dates) >= 2:
            try:
                h["xirr"] = round(xirr(cf_dates, cf_amounts) * 100, 2)
            except Exception:
                h["xirr"] = None
        else:
            h["xirr"] = None
            
        # Only include active holdings in the aggregate portfolio XIRR
        if h.get("units", 0) > 0.001:
            portfolio_cashflows.extend(zip(cf_dates, cf_amounts))

    # Portfolio-level XIRR
    portfolio_xirr = None
    if portfolio_cashflows and len(portfolio_cashflows) >= 2:
        try:
            p_dates, p_amounts = zip(*portfolio_cashflows)
            portfolio_xirr = round(xirr(list(p_dates), list(p_amounts)) * 100, 2)
        except Exception:
            portfolio_xirr = None

    # Sort all transactions by date
    all_transactions.sort(key=lambda x: x["date"] or "")

    result = {
        "investor_info": investor_info,
        "statement_period": statement_period,
        "holdings": list(holdings.values()),
        "all_transactions": all_transactions,
        "portfolio_xirr": portfolio_xirr,
    }
    return result


def recompute_xirr(data: dict) -> dict:
    """
    Recompute XIRR for all holdings and the portfolio, accounting for updated `live_nav`.
    Mutates `data` in place and returns it.
    """
    if "holdings" not in data:
        return data

    portfolio_cashflows = []
    for h in data["holdings"]:
        # If live_nav exists, use it for the final cashflow
        cf_dates, cf_amounts = _build_cashflows(h.get("transactions", []), h.get("units", 0), h.get("live_nav"))
        
        if cf_dates and len(cf_dates) >= 2:
            try:
                h["xirr"] = round(xirr(cf_dates, cf_amounts) * 100, 2)
            except Exception:
                h["xirr"] = None
        else:
            h["xirr"] = None
            
        # Only include active holdings in the aggregate portfolio XIRR
        if h.get("units", 0) > 0.001:
            portfolio_cashflows.extend(zip(cf_dates, cf_amounts))

    if portfolio_cashflows and len(portfolio_cashflows) >= 2:
        try:
            p_dates, p_amounts = zip(*portfolio_cashflows)
            data["portfolio_xirr"] = round(xirr(list(p_dates), list(p_amounts)) * 100, 2)
        except Exception:
            data["portfolio_xirr"] = None

    return data


def _classify_txn(raw_type: str, units: float, amount: float) -> str:
    """Classify transaction type from raw casparser type string."""
    rt = raw_type.upper()
    if "SIP" in rt:
        return "SIP"
    if "SWITCH" in rt and "IN" in rt:
        return "SWITCH_IN"
    if "SWITCH" in rt and "OUT" in rt:
        return "SWITCH_OUT"
    if "DIVR" in rt or ("DIV" in rt and "REINV" in rt):
        return "DIVR"
    if "DIV" in rt:
        return "DIVIDEND"
    if "REDEEM" in rt or "REDEMPTION" in rt or units < 0:
        return "SELL"
    if "PURCHASE" in rt or "BUY" in rt or "LUMPSUM" in rt:
        return "BUY"
    # fallback by sign
    return "BUY" if units > 0 else "SELL"


def _build_cashflows(transactions: list, current_units: float, live_nav: float = None):
    """Build (dates, amounts) cashflow series for XIRR calculation."""
    cf_dates = []
    cf_amounts = []
    today = date.today()

    for txn in transactions:
        if not txn.get("date") or txn.get("amount") is None:
            continue
        # Always ensure date is a datetime.date object (session JSON loads dates as strings)
        raw_date = txn["date"]
        if isinstance(raw_date, str):
            try:
                d = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
            except Exception:
                continue
        elif isinstance(raw_date, date):
            d = raw_date
        else:
            continue

        amt = txn["amount"]
        txn_type = txn.get("type", "")
        # Outflows (investments) = negative, Inflows (redemptions) = positive
        if txn_type in ("BUY", "SIP", "SWITCH_IN", "DIVR"):
            cf_dates.append(d)
            cf_amounts.append(-abs(float(amt)))
        elif txn_type in ("SELL", "SWITCH_OUT", "DIVIDEND"):
            cf_dates.append(d)
            cf_amounts.append(abs(float(amt)))

    # Add today's value as final inflow if units remain
    if current_units > 0.001:
        # Prefer live NAV, fall back to last known transaction NAV
        nav_for_val = live_nav
        if nav_for_val is None:
            for t in reversed(transactions):
                if t.get("nav") is not None:
                    nav_for_val = float(t["nav"])
                    break

        if nav_for_val is not None:
            cf_dates.append(today)
            cf_amounts.append(float(current_units * nav_for_val))

    return cf_dates, cf_amounts


def save_session(data: dict, filepath: str = 'session_data.json', merge: bool = False):
    """Save parsed data to DB/JSON file. If merge=True, combine with existing data."""
    existing_data = None
    if merge:
        try:
            existing_data = load_session(filepath)
        except Exception:
            pass

    if existing_data:
        try:
            # Merge Investor info (keep a list of names if they differ)
            names = set()
            if "investor_info" in existing_data and "name" in existing_data["investor_info"]:
                names.add(existing_data["investor_info"]["name"])
            if "investor_info" in data and "name" in data["investor_info"]:
                names.add(data["investor_info"]["name"])
            
            if names:
                data["investor_info"] = data.get("investor_info", {})
                data["investor_info"]["name"] = " & ".join(sorted(list(names)))
            
            # Merge Holdings
            existing_holdings = {h.get("name"): h for h in existing_data.get("holdings", []) if h.get("name")}
            new_holdings = {h.get("name"): h for h in data.get("holdings", []) if h.get("name")}
            
            merged_holdings = {}
            for name in set(existing_holdings.keys()).union(new_holdings.keys()):
                h1 = existing_holdings.get(name)
                h2 = new_holdings.get(name)
                
                if h1 and h2:
                    units = h1["units"] + h2["units"]
                    val1 = h1["units"] * h1.get("nav", 0)
                    val2 = h2["units"] * h2.get("nav", 0)
                    nav = (val1 + val2) / units if units > 0 else h1.get("nav", 0)
                    
                    merged_holdings[name] = {
                        "name": name,
                        "units": round(units, 4),
                        "nav": round(nav, 4),
                        "value": round(val1 + val2, 2),
                        "category": h1.get("category", h2.get("category", "")),
                        "subcategory": h1.get("subcategory", h2.get("subcategory", "")),
                        "amc": h1.get("amc", h2.get("amc", ""))
                    }
                elif h1:
                    merged_holdings[name] = h1
                else:
                    merged_holdings[name] = h2
                    
            data["holdings"] = list(merged_holdings.values())
            
            # Merge Transactions
            existing_txns = existing_data.get("transactions", [])
            new_txns = data.get("transactions", [])
            all_txns = existing_txns + new_txns
            unique_txns = []
            seen = set()
            
            for t in all_txns:
                sig = f"{t.get('date')}_{t.get('scheme_name')}_{t.get('type')}_{t.get('amount')}_{t.get('units')}"
                if sig not in seen:
                    seen.add(sig)
                    unique_txns.append(t)
                    
            data["transactions"] = unique_txns
            
        except Exception as e:
            print(f"Error merging session data: {e}. Overwriting instead.")

    # Always write to disk (keeps local dev in sync)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Failed to write session to disk: {e}")

    # Always write to DB (keeps Render/cloud in sync)
    try:
        from data.database import save_portfolio_session
        save_portfolio_session(json.dumps(data))
    except Exception as e:
        print(f"Failed to save session to database: {e}")


def load_session(filepath: str = 'session_data.json') -> Optional[dict]:
    """Load previously parsed CAS session.
    Priority:
      1. Local disk file (present in dev; freshest copy).
      2. Database (Supabase/SQLite — used on Render where disk is wiped).
    Whenever a disk file is successfully read, it is also written to the DB
    so that the cloud copy stays in sync.
    """
    # ── 1. Disk (local dev / file still present) ──────────────────────────────
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Sync this to DB so Render always has the latest copy
            try:
                from data.database import save_portfolio_session
                save_portfolio_session(json.dumps(data))
            except Exception:
                pass
            return data
        except Exception:
            pass  # fall through to DB

    # ── 2. Database (Render / disk wiped) ─────────────────────────────────────
    try:
        from data.database import get_portfolio_session
        db_data = get_portfolio_session()
        if db_data:
            return json.loads(db_data)
    except Exception:
        pass

    if not os.path.exists(filepath):
        raise FileNotFoundError("No parsed CAS data found. Please upload your CAS PDF first.")
        
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
