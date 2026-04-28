import os
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = Path("mutual_funds.db")
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.strip().strip("'").strip('"')

USE_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith("postgres"))

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import ThreadedConnectionPool
    
    _db_url = DATABASE_URL
    if "?" not in _db_url:
        _db_url += "?sslmode=require&keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
    else:
        if "sslmode" not in _db_url:
            _db_url += "&sslmode=require"
        if "keepalives" not in _db_url:
            _db_url += "&keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"

    _pg_pool = None  # Lazy: initialized on first use

    def _get_pool() -> ThreadedConnectionPool:
        """Return the global pool, creating it on first call (lazy init)."""
        global _pg_pool
        if _pg_pool is None:
            try:
                _pg_pool = ThreadedConnectionPool(1, 10, _db_url)
                logger.info("Postgres connection pool created.")
            except Exception as e:
                logger.error("FATAL: Could not create ThreadedConnectionPool: %s", e)
                raise
        return _pg_pool

    def _get_live_pg_conn():
        """Get a healthy connection from the pool, discarding stale ones."""
        pool = _get_pool()
        for attempt in range(3):
            conn = pool.getconn()
            if conn.closed:
                logger.warning("Discarding closed connection from pool (attempt %d)", attempt + 1)
                pool.putconn(conn, close=True)
                continue
            try:
                conn.rollback()  # reset transaction state + verify alive
                return conn
            except psycopg2.OperationalError:
                logger.warning("Discarding stale connection (attempt %d)", attempt + 1)
                try:
                    pool.putconn(conn, close=True)
                except Exception:
                    pass
        logger.warning("Pool exhausted healthy connections — opening direct connection")
        return psycopg2.connect(_db_url)

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
    def __init__(self, conn, is_pooled=False):
        self.conn = conn
        self.is_pooled = is_pooled
        self._had_error = False  # flag if this connection saw an OperationalError
        
    def cursor(self):
        c = self.conn.cursor(cursor_factory=RealDictCursor) if USE_POSTGRES else self.conn.cursor()
        return AgnosticCursor(c)
        
    def commit(self):
        try:
            self.conn.commit()
        except Exception:
            self._had_error = True
            raise

    def rollback(self):
        try:
            self.conn.rollback()
        except Exception:
            self._had_error = True

    def close(self):
        if self.is_pooled and USE_POSTGRES:
            pool = _get_pool()
            if self._had_error or self.conn.closed:
                try:
                    pool.putconn(self.conn, close=True)
                except Exception:
                    pass
            else:
                pool.putconn(self.conn)
        else:
            self.conn.close()

def get_connection():
    """Returns an agnostic connection wrapping SQLite or a healthy Postgres connection."""
    if USE_POSTGRES:
        conn = _get_live_pg_conn()
        return AgnosticConnection(conn, is_pooled=True)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return AgnosticConnection(conn, is_pooled=False)

def initialize_database():
    """Creates normalized database structures if they don't natively exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # ── User & Authentication ──────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP
        )
    ''')
    
    # ── Family Links ───────────────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS family_links (
            user_id_1 TEXT,
            user_id_2 TEXT,
            status TEXT,
            last_updated_at TIMESTAMP,
            PRIMARY KEY (user_id_1, user_id_2),
            FOREIGN KEY (user_id_1) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (user_id_2) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
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
    
    # Optimize sequential chronological reads for risk calculations.
    # Use a separate try/except + zero timeout so a slow CREATE INDEX on
    # a large nav_history table never aborts the rest of initialization.
    try:
        if USE_POSTGRES:
            # Disable statement timeout just for this DDL (restored after)
            cursor.execute("SET statement_timeout = 0")
            conn.commit()
            # Check if the index already exists to avoid re-building it
            cursor.execute(
                "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_nav_history_date'"
            )
            if not cursor.fetchone():
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_nav_history_date ON nav_history(nav_date)'
                )
                conn.commit()
            # Restore a sensible default timeout
            cursor.execute("SET statement_timeout = '30s'")
            conn.commit()
        else:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nav_history_date ON nav_history(nav_date)')
    except Exception as _idx_err:
        logger.warning("nav_history index creation skipped (non-fatal): %s", _idx_err)
        try:
            conn.rollback()
        except Exception:
            pass

    
    # --- Deep Dive Caching Tables ---
    
    # Fund Fundamentals
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fund_fundamentals (
            isin TEXT PRIMARY KEY,
            aum_cr REAL,
            expense_ratio REAL,
            exit_load TEXT,
            portfolio_turnover REAL,
            pe TEXT,
            cat_avg_pe TEXT,
            pb TEXT,
            cat_avg_pb TEXT,
            price_sale TEXT,
            cat_avg_price_sale TEXT,
            price_cash_flow TEXT,
            cat_avg_price_cash_flow TEXT,
            dividend_yield TEXT,
            cat_avg_dividend_yield TEXT,
            roe TEXT,
            cat_avg_roe TEXT,
            last_updated_at TIMESTAMP,
            FOREIGN KEY (isin) REFERENCES schemes (isin) ON DELETE CASCADE
        )
    ''')
    
    # Graceful Migration: Use IF NOT EXISTS (Postgres-native) so it never aborts the transaction
    for col in ["pe", "cat_avg_pe", "pb", "cat_avg_pb", "price_sale", "cat_avg_price_sale", "price_cash_flow", "cat_avg_price_cash_flow", "dividend_yield", "cat_avg_dividend_yield", "roe", "cat_avg_roe"]:
        try:
            cursor.execute(f"ALTER TABLE fund_fundamentals ADD COLUMN IF NOT EXISTS {col} TEXT")
            conn.commit()
        except Exception:
            pass
    
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
            cat_avg_sharpe REAL,
            cat_avg_sortino REAL,
            cat_avg_beta REAL,
            cat_avg_std_dev REAL,
            cat_avg_alpha REAL,
            last_updated_at TIMESTAMP,
            FOREIGN KEY (isin) REFERENCES schemes (isin) ON DELETE CASCADE
        )
    ''')

    # Graceful migration for cat_avg columns added after initial schema
    for col in ["cat_avg_sharpe", "cat_avg_sortino", "cat_avg_beta", "cat_avg_std_dev", "cat_avg_alpha"]:
        try:
            cursor.execute(f"ALTER TABLE fund_risk ADD COLUMN {col} REAL")
            conn.commit()
        except Exception:
            pass


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
    
    # Postgres/SQLite KV Store for generic JSON caching (Replacing Redis)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kv_cache (
            key TEXT PRIMARY KEY,
            value_json TEXT,
            expires_at REAL
        )
    ''')

    # ── Per-User Fund Pre-fetch Tracking ───────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_fund_cache (
            user_id   TEXT NOT NULL,
            isin      TEXT NOT NULL,
            cached_at TIMESTAMP NOT NULL,
            PRIMARY KEY (user_id, isin)
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
            INSERT INTO fund_fundamentals (isin, aum_cr, expense_ratio, exit_load, portfolio_turnover, price_sale, cat_avg_price_sale, price_cash_flow, cat_avg_price_cash_flow, dividend_yield, cat_avg_dividend_yield, roe, cat_avg_roe, last_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(isin) DO UPDATE SET
                aum_cr=excluded.aum_cr,
                expense_ratio=excluded.expense_ratio,
                exit_load=excluded.exit_load,
                portfolio_turnover=excluded.portfolio_turnover,
                price_sale=excluded.price_sale,
                cat_avg_price_sale=excluded.cat_avg_price_sale,
                price_cash_flow=excluded.price_cash_flow,
                cat_avg_price_cash_flow=excluded.cat_avg_price_cash_flow,
                dividend_yield=excluded.dividend_yield,
                cat_avg_dividend_yield=excluded.cat_avg_dividend_yield,
                roe=excluded.roe,
                cat_avg_roe=excluded.cat_avg_roe,
                last_updated_at=excluded.last_updated_at
        ''', (
            isin, fundamentals.get("aum_cr"), fundamentals.get("expense_ratio"), fundamentals.get("exit_load"), fundamentals.get("portfolio_turnover"),
            fundamentals.get("price_sale"), fundamentals.get("cat_avg_price_sale"), fundamentals.get("price_cash_flow"), fundamentals.get("cat_avg_price_cash_flow"),
            fundamentals.get("dividend_yield"), fundamentals.get("cat_avg_dividend_yield"), fundamentals.get("roe"), fundamentals.get("cat_avg_roe"),
            now
        ))
        
        # 2. Risk
        c.execute('''
            INSERT INTO fund_risk (isin, benchmark_name, volatility, sharpe, sortino, beta, alpha, max_drawdown_pct,
                                   cat_avg_sharpe, cat_avg_sortino, cat_avg_beta, cat_avg_std_dev, cat_avg_alpha,
                                   last_updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(isin) DO UPDATE SET
                benchmark_name=excluded.benchmark_name,
                volatility=excluded.volatility,
                sharpe=excluded.sharpe,
                sortino=excluded.sortino,
                beta=excluded.beta,
                alpha=excluded.alpha,
                max_drawdown_pct=excluded.max_drawdown_pct,
                cat_avg_sharpe=excluded.cat_avg_sharpe,
                cat_avg_sortino=excluded.cat_avg_sortino,
                cat_avg_beta=excluded.cat_avg_beta,
                cat_avg_std_dev=excluded.cat_avg_std_dev,
                cat_avg_alpha=excluded.cat_avg_alpha,
                last_updated_at=excluded.last_updated_at
        ''', (
            isin, risk.get("benchmark_name"), risk.get("volatility"),
            risk.get("sharpe"), risk.get("sortino"), risk.get("beta"),
            risk.get("alpha"), risk.get("max_drawdown_pct"),
            risk.get("cat_avg_sharpe"), risk.get("cat_avg_sortino"),
            risk.get("cat_avg_beta"), risk.get("cat_avg_std_dev"), risk.get("cat_avg_alpha"),
            now
        ))
        
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
            sec_weight = s.get("value") or s.get("weight") or s.get("weight_pct") or s.get("pct")
            c.execute('''
                INSERT INTO fund_sectors (isin, sector_name, weight_pct, last_updated_at)
                VALUES (?, ?, ?, ?)
            ''', (isin, sec_name, sec_weight, now))

            
        conn.commit()
    except Exception as e:
        conn.rollback()
        err_str = str(e).lower()
        # Self-healing migration: if Postgres reports a missing column, add it and retry
        if "column" in err_str and "does not exist" in err_str:
            logger.warning(f"Schema migration required for {isin}, running ALTER TABLE and retrying: {e}")
            conn2 = get_connection()
            c2 = conn2.cursor()
            for col in ["pe", "cat_avg_pe", "pb", "cat_avg_pb", "price_sale", "cat_avg_price_sale", "price_cash_flow", "cat_avg_price_cash_flow", "dividend_yield", "cat_avg_dividend_yield", "roe", "cat_avg_roe"]:
                try:
                    c2.execute(f"ALTER TABLE fund_fundamentals ADD COLUMN IF NOT EXISTS {col} TEXT")
                    conn2.commit()
                except Exception:
                    conn2.rollback()
            conn2.close()
            # Retry the full insert after migration
            conn3 = get_connection()
            c3 = conn3.cursor()
            try:
                c3.execute('''
                    INSERT INTO fund_fundamentals (isin, aum_cr, expense_ratio, exit_load, portfolio_turnover, pe, cat_avg_pe, pb, cat_avg_pb, price_sale, cat_avg_price_sale, price_cash_flow, cat_avg_price_cash_flow, dividend_yield, cat_avg_dividend_yield, roe, cat_avg_roe, last_updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(isin) DO UPDATE SET
                        aum_cr=excluded.aum_cr,
                        expense_ratio=excluded.expense_ratio,
                        exit_load=excluded.exit_load,
                        portfolio_turnover=excluded.portfolio_turnover,
                        pe=excluded.pe,
                        cat_avg_pe=excluded.cat_avg_pe,
                        pb=excluded.pb,
                        cat_avg_pb=excluded.cat_avg_pb,
                        price_sale=excluded.price_sale,
                        cat_avg_price_sale=excluded.cat_avg_price_sale,
                        price_cash_flow=excluded.price_cash_flow,
                        cat_avg_price_cash_flow=excluded.cat_avg_price_cash_flow,
                        dividend_yield=excluded.dividend_yield,
                        cat_avg_dividend_yield=excluded.cat_avg_dividend_yield,
                        roe=excluded.roe,
                        cat_avg_roe=excluded.cat_avg_roe,
                        last_updated_at=excluded.last_updated_at
                ''', (
                    isin, fundamentals.get("aum_cr"), fundamentals.get("expense_ratio"), fundamentals.get("exit_load"), fundamentals.get("portfolio_turnover"),
                    fundamentals.get("pe"), fundamentals.get("cat_avg_pe"), fundamentals.get("pb"), fundamentals.get("cat_avg_pb"),
                    fundamentals.get("price_sale"), fundamentals.get("cat_avg_price_sale"), fundamentals.get("price_cash_flow"), fundamentals.get("cat_avg_price_cash_flow"),
                    fundamentals.get("dividend_yield"), fundamentals.get("cat_avg_dividend_yield"), fundamentals.get("roe"), fundamentals.get("cat_avg_roe"),
                    now
                ))
                conn3.commit()
                logger.info(f"Successfully cached fundamentals for {isin} after schema migration.")
            except Exception as e2:
                logger.error(f"Retry cache also failed for {isin}: {e2}")
                conn3.rollback()
            finally:
                conn3.close()
        else:
            logger.error(f"Failed to cache deep dive for {isin}: {e}")
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
        val = row['last_updated_at']
        if isinstance(val, str):
            last_updated = datetime.fromisoformat(val)
        else:
            last_updated = val
            
        age = (datetime.now() - last_updated).total_seconds() / 3600
        if age > max_age_hours:
            conn.close()
            return None # Stale
    except (ValueError, TypeError):
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


def _delete_deep_dive_cache(isin: str):
    """Delete all Supabase/SQLite cached deep-dive rows for an ISIN (used by force-refresh)."""
    conn = get_connection()
    c = conn.cursor()
    try:
        for table in ('fund_fundamentals', 'fund_risk', 'fund_performance', 'fund_holdings', 'fund_sectors'):
            c.execute(f'DELETE FROM {table} WHERE isin = ?', (isin,))
        conn.commit()
        logger.info("Deleted deep-dive cache for %s from all tables", isin)
    except Exception as e:
        logger.warning("_delete_deep_dive_cache failed for %s: %s", isin, e)
    finally:
        conn.close()

def mark_user_fund_cached(user_id: str, isin: str):
    """Records that a fund's deep-dive data has been pre-fetched for this user."""
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO user_fund_cache (user_id, isin, cached_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, isin) DO UPDATE SET cached_at=excluded.cached_at
        ''', (user_id, isin, datetime.now().isoformat()))
        conn.commit()
    except Exception as e:
        logger.warning(f"mark_user_fund_cached failed for {user_id}/{isin}: {e}")
    finally:
        conn.close()

def is_user_fund_cached(user_id: str, isin: str, max_age_hours: float = 24.0) -> bool:
    """Returns True if this user's fund data was pre-fetched within max_age_hours."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT cached_at FROM user_fund_cache WHERE user_id = ? AND isin = ?', (user_id, isin))
    row = c.fetchone()
    conn.close()
    if not row or not row['cached_at']:
        return False
    try:
        val = row['cached_at']
        cached_at = datetime.fromisoformat(val) if isinstance(val, str) else val
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600
        return age_hours <= max_age_hours
    except (ValueError, TypeError):
        return False

def count_user_funds_cached(user_id: str, max_age_hours: float = 24.0) -> int:
    """Returns the total number of funds uniquely cached for the user within max_age."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT cached_at FROM user_fund_cache WHERE user_id = ?', (user_id,))
    rows = c.fetchall()
    conn.close()
    
    count = 0
    now = datetime.now()
    for row in rows:
        val = row['cached_at']
        if val:
            try:
                cached_at = datetime.fromisoformat(val) if isinstance(val, str) else val
                if (now - cached_at).total_seconds() / 3600 <= max_age_hours:
                    count += 1
            except (ValueError, TypeError):
                pass
    return count

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

def create_user(username: str, password_hash: str) -> str:
    import uuid
    user_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)', (user_id, username, password_hash, now))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise ValueError(f"Username {username} may already exist") from e
    finally:
        conn.close()
    return user_id

def get_user_by_username(username: str) -> Optional[dict]:
    conn = get_connection()
    c = conn.cursor()
    # Check users table first (legacy)
    c.execute('SELECT id, username, password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    if row:
        conn.close()
        return dict(row)
    
    # If not found, try to find a portfolio_session where investor_info.name matches partially
    c.execute('SELECT id, session_data FROM portfolio_sessions')
    for p_row in c.fetchall():
        try:
            import json
            data = json.loads(p_row['session_data'])
            name = data.get("investor_info", {}).get("name", "")
            if name and username.lower() in name.lower():
                conn.close()
                return {"id": p_row['id'], "username": name, "password_hash": ""}
        except:
            pass
            
    conn.close()
    return None

def get_user_by_id(user_id: str) -> Optional[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT id, username, password_hash FROM users WHERE id = ?', (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def accept_family_invite(user_id_1: str, user_id_2: str):
    now = datetime.now().isoformat()
    conn = get_connection()
    c = conn.cursor()
    try:
        # Ensure users exist in public.users to satisfy foreign keys
        c.execute('INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING', (user_id_1, f'auth_user_{user_id_1}', '', now))
        c.execute('INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING', (user_id_2, f'auth_user_{user_id_2}', '', now))
        
        # Check if pending inverse relationship exists
        c.execute('SELECT status FROM family_links WHERE user_id_1 = ? AND user_id_2 = ?', (user_id_2, user_id_1))
        row = c.fetchone()
        if row and row['status'] == 'pending':
            c.execute('UPDATE family_links SET status = ?, last_updated_at = ? WHERE user_id_1 = ? AND user_id_2 = ?', 
                      ('accepted', now, user_id_2, user_id_1))
        else:
            c.execute('''
                INSERT INTO family_links (user_id_1, user_id_2, status, last_updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id_1, user_id_2) DO UPDATE SET
                    status=excluded.status,
                    last_updated_at=excluded.last_updated_at
            ''', (user_id_2, user_id_1, 'accepted', now))
            
        c.execute('''
            INSERT INTO family_links (user_id_1, user_id_2, status, last_updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id_1, user_id_2) DO UPDATE SET
                status=excluded.status,
                last_updated_at=excluded.last_updated_at
        ''', (user_id_1, user_id_2, 'accepted', now))
        conn.commit()
    finally:
        conn.close()

def send_family_invite(from_user_id: str, to_user_id: str):
    now = datetime.now().isoformat()
    conn = get_connection()
    c = conn.cursor()
    try:
        # Ensure users exist in public.users to satisfy foreign keys
        c.execute('INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING', (from_user_id, f'auth_user_{from_user_id}', '', now))
        c.execute('INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING', (to_user_id, f'auth_user_{to_user_id}', '', now))
        
        c.execute('''
            INSERT INTO family_links (user_id_1, user_id_2, status, last_updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id_1, user_id_2) DO NOTHING
        ''', (from_user_id, to_user_id, 'pending', now))
        conn.commit()
    finally:
        conn.close()

def get_family_members(user_id: str) -> List[str]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT user_id_2 FROM family_links WHERE user_id_1 = ? AND status = ?', (user_id, 'accepted'))
    rows = c.fetchall()
    conn.close()
    return [row['user_id_2'] for row in rows]
    
def get_pending_invites(user_id: str) -> List[dict]:
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT user_id_1
        FROM family_links
        WHERE user_id_2 = ? AND status = 'pending'
    ''', (user_id,))
    rows = c.fetchall()
    
    results = []
    for row in rows:
        uid = row['user_id_1']
        # Try users table first
        c.execute('SELECT username FROM users WHERE id = ?', (uid,))
        u_row = c.fetchone()
        name = u_row['username'] if u_row else None
        
        # Fallback to portfolio session
        if not name:
            c.execute('SELECT session_data FROM portfolio_sessions WHERE id = ?', (uid,))
            p_row = c.fetchone()
            if p_row:
                try:
                    import json
                    data = json.loads(p_row['session_data'])
                    name = data.get("investor_info", {}).get("name")
                except:
                    pass
        
        results.append({"id": uid, "username": name or "Unknown"})
        
    conn.close()
    return results

def search_users(query: str) -> List[dict]:
    if not query or len(query) < 2:
        return []
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT id, session_data FROM portfolio_sessions')
    results = []
    seen = set()
    for p_row in c.fetchall():
        try:
            import json
            data = json.loads(p_row['session_data'])
            name = data.get("investor_info", {}).get("name", "")
            if name and query.lower() in name.lower():
                if name not in seen:
                    seen.add(name)
                    results.append({"id": p_row['id'], "username": name})
        except:
            pass
    conn.close()
    return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    initialize_database()
