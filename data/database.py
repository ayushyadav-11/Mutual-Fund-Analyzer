import os
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = Path("mutual_funds.db")
DATABASE_URL = os.getenv("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith("postgres"))

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor

class AgnosticCursor:
    def __init__(self, cursor):
        self.cursor = cursor
        
    def _translate(self, query: str) -> str:
        if not USE_POSTGRES:
            return query
        # Translate SQLite ? to Postgres %s
        query = query.replace('?', '%s')
        # Translate INSERT OR IGNORE syntax
        if "INSERT OR IGNORE INTO nav_history" in query:
            query = query.replace("INSERT OR IGNORE INTO nav_history", "INSERT INTO nav_history")
            if "ON CONFLICT" not in query:
                # Append conflict rule for nav_history PK
                query = query + " ON CONFLICT (isin, nav_date) DO NOTHING"
        return query
        
    def execute(self, query, params=None):
        query = self._translate(query)
        if params is not None:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
            
    def executemany(self, query, params):
        query = self._translate(query)
        self.cursor.executemany(query, params)
        
    def fetchone(self): return self.cursor.fetchone()
    def fetchall(self): return self.cursor.fetchall()

class AgnosticConnection:
    def __init__(self, conn):
        self.conn = conn
        
    def cursor(self):
        c = self.conn.cursor(cursor_factory=RealDictCursor) if USE_POSTGRES else self.conn.cursor()
        return AgnosticCursor(c)
        
    def commit(self): self.conn.commit()
    def rollback(self): self.conn.rollback()
    def close(self): self.conn.close()

def get_connection():
    """Returns an agnostic connection instance wrapping SQLite or Postgres perfectly."""
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    return AgnosticConnection(conn)

def initialize_database():
    """Creates normalized database structures if they don't natively exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Core Fund Information Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schemes (
            isin TEXT PRIMARY KEY,
            scheme_code TEXT UNIQUE,
            scheme_name TEXT NOT NULL,
            category TEXT,
            benchmark TEXT,
            last_updated TIMESTAMP
        )
    ''')
    
    # Heavy Time-Series NAV Data Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nav_history (
            isin TEXT NOT NULL,
            nav_date DATE NOT NULL,
            nav REAL NOT NULL,
            PRIMARY KEY (isin, nav_date),
            FOREIGN KEY (isin) REFERENCES schemes (isin) ON DELETE CASCADE
        )
    ''')
    
    # Optimize sequential chronological reads for risk calculations
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_nav_history_date ON nav_history(nav_date)')
    
    # --- Deep Dive Caching Tables ---
    
    # Fund Fundamentals
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fund_fundamentals (
            isin TEXT PRIMARY KEY,
            aum_cr REAL,
            expense_ratio REAL,
            exit_load TEXT,
            portfolio_turnover REAL,
            last_updated_at TIMESTAMP,
            FOREIGN KEY (isin) REFERENCES schemes (isin) ON DELETE CASCADE
        )
    ''')
    
    # Fund Risk
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fund_risk (
            isin TEXT PRIMARY KEY,
            volatility REAL,
            benchmark_name TEXT,
            sharpe REAL,
            sortino REAL,
            beta REAL,
            alpha REAL,
            max_drawdown_pct REAL,
            last_updated_at TIMESTAMP,
            FOREIGN KEY (isin) REFERENCES schemes (isin) ON DELETE CASCADE
        )
    ''')
    
    # Fund Performance (Returns over time)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fund_performance (
            isin TEXT,
            period TEXT,
            fund_return REAL,
            benchmark_return REAL,
            last_updated_at TIMESTAMP,
            PRIMARY KEY (isin, period),
            FOREIGN KEY (isin) REFERENCES schemes (isin) ON DELETE CASCADE
        )
    ''')
    
    # Session Persistence Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_sessions (
            id TEXT PRIMARY KEY,
            session_data TEXT NOT NULL,
            last_updated TIMESTAMP
        )
    ''')
    
    # Fund Holdings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fund_holdings (
            isin TEXT,
            asset_name TEXT,
            weight_pct REAL,
            last_updated_at TIMESTAMP,
            PRIMARY KEY (isin, asset_name),
            FOREIGN KEY (isin) REFERENCES schemes (isin) ON DELETE CASCADE
        )
    ''')
    
    # Fund Sectors
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fund_sectors (
            isin TEXT,
            sector_name TEXT,
            weight_pct REAL,
            last_updated_at TIMESTAMP,
            PRIMARY KEY (isin, sector_name),
            FOREIGN KEY (isin) REFERENCES schemes (isin) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Successfully initialized the core SQLite database engine and normalized schema.")

def insert_or_update_scheme(isin: str, scheme_name: str, scheme_code: Optional[str] = None, category: Optional[str] = None, benchmark: Optional[str] = None):
    """Upsert fund configuration matrices into the `schemes` table."""
    conn = get_connection()
    c = conn.cursor()
    now = datetime.now().isoformat()
    
    c.execute('''
        INSERT INTO schemes (isin, scheme_code, scheme_name, category, benchmark, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(isin) DO UPDATE SET
            scheme_name=excluded.scheme_name,
            scheme_code=COALESCE(excluded.scheme_code, schemes.scheme_code),
            category=COALESCE(excluded.category, schemes.category),
            benchmark=COALESCE(excluded.benchmark, schemes.benchmark),
            last_updated=excluded.last_updated
    ''', (isin, scheme_code, scheme_name, category, benchmark, now))
    
    conn.commit()
    conn.close()

def batch_insert_navs(isin: str, nav_records: List[Dict[str, float]]):
    """
    Ingest heavy chunks of chronological NAV data.
    nav_records format: [{'date': 'YYYY-MM-DD', 'nav': 14.5}, ...]
    """
    if not nav_records:
        return
        
    conn = get_connection()
    c = conn.cursor()
    
    # We use INSERT OR IGNORE because historical NAVs never change
    data_tuples = [(isin, r['date'], float(r['nav'])) for r in nav_records]
    
    c.executemany('''
        INSERT OR IGNORE INTO nav_history (isin, nav_date, nav)
        VALUES (?, ?, ?)
    ''', data_tuples)
    
    # Update the last_updated timestamp
    c.execute("UPDATE schemes SET last_updated = ? WHERE isin = ?", (datetime.now().isoformat(), isin))
    
    conn.commit()
    conn.close()

def get_nav_series(isin: str) -> List[Dict]:
    """Retrieves identical JSON serializable NAV series from the SQL layer."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT nav_date as date, nav FROM nav_history WHERE isin = ? ORDER BY nav_date ASC', (isin,))
    rows = c.fetchall()
    conn.close()
    
    # Return identically formatted dict arrays for drop-in replacement
    return [{"date": row['date'], "nav": row['nav']} for row in rows]

# ── Deep Dive Caching Helpers ───────────────────────────────────────────────

def cache_fund_deep_dive(isin: str, fundamentals: dict, risk: dict, returns: dict, bench_returns: dict, holdings: list, sectors: list):
    """Caches the full deep-dive payload into SQLite with the current timestamp."""
    conn = get_connection()
    c = conn.cursor()
    now = datetime.now().isoformat()
    
    try:
        # 1. Fundamentals
        c.execute('''
            INSERT INTO fund_fundamentals (isin, aum_cr, expense_ratio, exit_load, portfolio_turnover, last_updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(isin) DO UPDATE SET
                aum_cr=excluded.aum_cr,
                expense_ratio=excluded.expense_ratio,
                exit_load=excluded.exit_load,
                portfolio_turnover=excluded.portfolio_turnover,
                last_updated_at=excluded.last_updated_at
        ''', (isin, fundamentals.get("aum_cr"), fundamentals.get("expense_ratio"), fundamentals.get("exit_load"), fundamentals.get("portfolio_turnover"), now))
        
        # 2. Risk
        c.execute('''
            INSERT INTO fund_risk (isin, benchmark_name, volatility, sharpe, sortino, beta, alpha, max_drawdown_pct, last_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(isin) DO UPDATE SET
                benchmark_name=excluded.benchmark_name,
                volatility=excluded.volatility,
                sharpe=excluded.sharpe,
                sortino=excluded.sortino,
                beta=excluded.beta,
                alpha=excluded.alpha,
                max_drawdown_pct=excluded.max_drawdown_pct,
                last_updated_at=excluded.last_updated_at
        ''', (isin, risk.get("benchmark_name"), risk.get("volatility"), risk.get("sharpe"), risk.get("sortino"), risk.get("beta"), risk.get("alpha"), risk.get("max_drawdown_pct"), now))
        
        # 3. Performance
        for period, val in returns.items():
            bench_val = bench_returns.get(period)
            c.execute('''
                INSERT INTO fund_performance (isin, period, fund_return, benchmark_return, last_updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(isin, period) DO UPDATE SET
                    fund_return=excluded.fund_return,
                    benchmark_return=excluded.benchmark_return,
                    last_updated_at=excluded.last_updated_at
            ''', (isin, period, val, bench_val, now))
            
        # 4. Holdings (Clear old and insert new)
        c.execute('DELETE FROM fund_holdings WHERE isin = ?', (isin,))
        for h in holdings:
            c.execute('''
                INSERT INTO fund_holdings (isin, asset_name, weight_pct, last_updated_at)
                VALUES (?, ?, ?, ?)
            ''', (isin, h.get("asset"), h.get("weight"), now))
            
        # 5. Sectors (Clear old and insert new)
        c.execute('DELETE FROM fund_sectors WHERE isin = ?', (isin,))
        for s in sectors:
            # handle formats depending on how scrapers provide it
            sec_name = s.get("name") or s.get("sector")
            sec_weight = s.get("value") or s.get("weight") or s.get("weight_pct")
            c.execute('''
                INSERT INTO fund_sectors (isin, sector_name, weight_pct, last_updated_at)
                VALUES (?, ?, ?, ?)
            ''', (isin, sec_name, sec_weight, now))
            
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to cache deep dive for {isin}: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_cached_fund_deep_dive(isin: str, max_age_hours=1) -> Optional[dict]:
    """Retrieves cached deep dive data if it's newer than max_age_hours. Returns None if stale/missing."""
    conn = get_connection()
    c = conn.cursor()
    
    # Check age via fund_fundamentals table
    c.execute('SELECT last_updated_at FROM fund_fundamentals WHERE isin = ?', (isin,))
    row = c.fetchone()
    if not row or not row['last_updated_at']:
        conn.close()
        return None
        
    try:
        last_updated = datetime.fromisoformat(row['last_updated_at'])
        age = (datetime.now() - last_updated).total_seconds() / 3600
        if age > max_age_hours:
            conn.close()
            return None # Stale
    except ValueError:
        conn.close()
        return None

    # Fetch all pieces
    result = {}
    
    # Fundamentals
    c.execute('SELECT * FROM fund_fundamentals WHERE isin = ?', (isin,))
    f_row = c.fetchone()
    result['fundamentals'] = {k: f_row[k] for k in f_row.keys() if k not in ('isin', 'last_updated_at')}
    
    # Risk
    c.execute('SELECT * FROM fund_risk WHERE isin = ?', (isin,))
    r_row = c.fetchone()
    if r_row:
        result['risk'] = {k: r_row[k] for k in r_row.keys() if k not in ('isin', 'last_updated_at')}
    else:
        result['risk'] = {}
        
    # Performance
    c.execute('SELECT period, fund_return, benchmark_return FROM fund_performance WHERE isin = ?', (isin,))
    returns = {}
    bench_returns = {}
    for p_row in c.fetchall():
        returns[p_row['period']] = p_row['fund_return']
        bench_returns[p_row['period']] = p_row['benchmark_return']
    result['returns'] = returns
    result['benchmark_cagr'] = bench_returns
    
    # Holdings
    c.execute('SELECT asset_name, weight_pct FROM fund_holdings WHERE isin = ? ORDER BY weight_pct DESC', (isin,))
    result['holdings'] = [{"asset": h_row['asset_name'], "weight": h_row['weight_pct']} for h_row in c.fetchall()]
    
    # Sectors
    c.execute('SELECT sector_name, weight_pct FROM fund_sectors WHERE isin = ? ORDER BY weight_pct DESC', (isin,))
    result['sectors'] = [{"sector": s_row['sector_name'], "weight": s_row['weight_pct']} for s_row in c.fetchall()]
    
    conn.close()
    return result

def get_portfolio_session(session_id: str = "master") -> Optional[str]:
    """Retrieve raw JSON session data from the database."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT session_data FROM portfolio_sessions WHERE id = ?', (session_id,))
    row = c.fetchone()
    conn.close()
    return row['session_data'] if row else None

def save_portfolio_session(data_json: str, session_id: str = "master"):
    """Persists raw JSON session payload to the database."""
    now = datetime.now().isoformat()
    conn = get_connection()
    c = conn.cursor()
    try:
        # standard upsert logic independent of Postgres JSONB / SQLite limitations
        c.execute('SELECT id FROM portfolio_sessions WHERE id = ?', (session_id,))
        if c.fetchone():
            c.execute('UPDATE portfolio_sessions SET session_data = ?, last_updated = ? WHERE id = ?', 
                      (data_json, now, session_id))
        else:
            c.execute('INSERT INTO portfolio_sessions (id, session_data, last_updated) VALUES (?, ?, ?)', 
                      (session_id, data_json, now))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to save portfolio session: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    initialize_database()
