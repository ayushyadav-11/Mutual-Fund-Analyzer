import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = Path("mutual_funds.db")

def get_connection():
    """Returns a connected SQLite instance with row factories enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    initialize_database()
