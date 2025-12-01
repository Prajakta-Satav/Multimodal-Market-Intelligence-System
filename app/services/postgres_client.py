# app/services/postgres_client.py

import psycopg2
import psycopg2.extras
from app.core.config import settings
from app.core.logging import logger

class PostgresService:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )
        self.conn.autocommit = True
        self._init_schema()
    
    def _init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10),
                date DATE,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT
            );""")
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id SERIAL PRIMARY KEY,
                company VARCHAR(50),
                quarter VARCHAR(20),
                year VARCHAR(10),
                text TEXT,
                file_path VARCHAR,
                uploaded_at TIMESTAMP DEFAULT NOW()
            );""")
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS balance_sheets (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10),
                year VARCHAR(10),
                total_assets DOUBLE PRECISION,
                total_liabilities DOUBLE PRECISION,
                stockholder_equity DOUBLE PRECISION,
                uploaded_at TIMESTAMP DEFAULT NOW()
            );""")
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_fundamentals (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10),
                year VARCHAR(10),
                revenue DOUBLE PRECISION,
                net_income DOUBLE PRECISION,
                eps DOUBLE PRECISION,
                pe_ratio DOUBLE PRECISION,
                uploaded_at TIMESTAMP DEFAULT NOW()
            );""")
            
            cur.execute("""
            CREATE TABLE IF NOT EXISTS presentations (
                id SERIAL PRIMARY KEY,
                company VARCHAR(50),
                quarter VARCHAR(20),
                year VARCHAR(10),
                slide_number INTEGER,
                slide_text TEXT,
                chart_description TEXT,
                file_path VARCHAR,
                uploaded_at TIMESTAMP DEFAULT NOW()
            );""")
            
            logger.info("[Postgres] Schema initialized")
    
    # --- Insert methods ---
    
    def insert_stock_price(self, ticker, date, open_price, high, low, close, volume):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_prices (ticker, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (ticker, date, open_price, high, low, close, volume))
    
    def insert_transcript(self, company, quarter, year, text, file_path):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO transcripts (company, quarter, year, text, file_path)
                VALUES (%s, %s, %s, %s, %s)
            """, (company, quarter, year, text, file_path))
    
    def insert_balance_sheet(self, ticker, year, total_assets, total_liabilities, stockholder_equity):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO balance_sheets (ticker, year, total_assets, total_liabilities, stockholder_equity)
                VALUES (%s, %s, %s, %s, %s)
            """, (ticker, year, total_assets, total_liabilities, stockholder_equity))
    
    def insert_fundamental(self, ticker, year, revenue, net_income, eps, pe_ratio):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_fundamentals (ticker, year, revenue, net_income, eps, pe_ratio)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (ticker, year, revenue, net_income, eps, pe_ratio))
    
    def insert_presentation(self, company, quarter, year, slide_number, slide_text, chart_description, file_path):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO presentations (company, quarter, year, slide_number, slide_text, chart_description, file_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (company, quarter, year, slide_number, slide_text, chart_description, file_path))
    
    # --- Fetch methods (FIXED - No SQL injection) ---
    
    def fetch_all_fundamentals(self, limit=None):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if limit:
                cur.execute("SELECT * FROM stock_fundamentals ORDER BY id ASC LIMIT %s", (limit,))
            else:
                cur.execute("SELECT * FROM stock_fundamentals ORDER BY id ASC")
            return cur.fetchall()
    
    def fetch_all_prices(self, limit=None):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if limit:
                cur.execute("SELECT * FROM stock_prices ORDER BY id ASC LIMIT %s", (limit,))
            else:
                cur.execute("SELECT * FROM stock_prices ORDER BY id ASC")
            return cur.fetchall()
    
    def fetch_all_transcripts(self, limit=None):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if limit:
                cur.execute("SELECT * FROM transcripts ORDER BY id ASC LIMIT %s", (limit,))
            else:
                cur.execute("SELECT * FROM transcripts ORDER BY id ASC")
            return cur.fetchall()
    
    def fetch_all_presentations(self, limit=None):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if limit:
                cur.execute("SELECT * FROM presentations ORDER BY id ASC LIMIT %s", (limit,))
            else:
                cur.execute("SELECT * FROM presentations ORDER BY id ASC")
            return cur.fetchall()
    
    def fetch_all_balance_sheets(self, limit=None):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if limit:
                cur.execute("SELECT * FROM balance_sheets ORDER BY id ASC LIMIT %s", (limit,))
            else:
                cur.execute("SELECT * FROM balance_sheets ORDER BY id ASC")
            return cur.fetchall()
