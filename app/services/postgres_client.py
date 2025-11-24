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
                quarter VARCHAR(10),
                year VARCHAR(10),
                text TEXT,
                file_path VARCHAR,
                uploaded_at TIMESTAMP DEFAULT NOW()
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS balance_sheets (
                id SERIAL PRIMARY KEY,
                company VARCHAR(50),
                quarter VARCHAR(20),
                json_data JSONB,
                file_path VARCHAR
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS presentations (
                id SERIAL PRIMARY KEY,
                company VARCHAR(50),
                quarter VARCHAR(20),
                slides JSONB,
                file_path VARCHAR
            );""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_fundamentals (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            quarter VARCHAR(10),
            market_cap BIGINT,
            pb_ratio DOUBLE PRECISION,
            pe_ratio DOUBLE PRECISION,
            peg_ratio DOUBLE PRECISION,
            price_to_sales DOUBLE PRECISION,
            eps DOUBLE PRECISION,
            roe DOUBLE PRECISION,
            roa DOUBLE PRECISION,
            beta DOUBLE PRECISION,
            dividend_yield DOUBLE PRECISION,
            debt_to_equity DOUBLE PRECISION,
            book_value DOUBLE PRECISION,
            ebitda DOUBLE PRECISION,
            revenue DOUBLE PRECISION,
            profit DOUBLE PRECISION,
            networth DOUBLE PRECISION,
            face_value DOUBLE PRECISION
        );""")
        logger.info("[Postgres] Schema ensured")

    def insert_balance_sheet(self, payload: dict):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO balance_sheets (company, quarter, json_data, file_path)
                VALUES (%s, %s, %s::jsonb, %s)
                RETURNING id
            """, (payload["company"], payload["quarter"], payload["json_data"], payload["file_path"]))
            return cur.fetchone()["id"]

    def insert_presentation(self, payload: dict):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO presentations (company, quarter, slides, file_path)
                VALUES (%s, %s, %s::jsonb, %s)
                RETURNING id
            """, (payload["company"], payload["quarter"], payload["slides"], payload["file_path"]))
            return cur.fetchone()["id"]

    def insert_transcript(self, payload: dict):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO transcripts (company, quarter, year, text, file_path)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (payload["company"], payload["quarter"], payload["year"], payload["text"], payload["file_path"]))
            return cur.fetchone()["id"]

    def insert_stock(self, payload: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_prices (ticker, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                payload["ticker"], payload["date"], payload["open"], payload["high"],
                payload["low"], payload["close"], payload["volume"]
            ))
    def insert_stock_fundamentals(self, payload: dict):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_fundamentals (
                    ticker, date, quarter, market_cap, pb_ratio, pe_ratio, peg_ratio,
                    price_to_sales, eps, roe, roa, beta, dividend_yield,
                    debt_to_equity, book_value, ebitda, revenue, profit, networth, face_value
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                payload.get("ticker"),
                payload.get("date"),
                payload.get("quarter"),
                payload.get("MarketCap"),
                payload.get("PB_Ratio"),
                payload.get("PE_Ratio"),
                payload.get("PEG_Ratio"),
                payload.get("PriceToSales"),
                payload.get("EPS"),
                payload.get("ROE"),
                payload.get("ROA"),
                payload.get("Beta"),
                payload.get("DividendYield"),
                payload.get("DebtToEquity"),
                payload.get("BookValue"),
                payload.get("EBITDA"),
                payload.get("Revenue"),
                payload.get("Profit"),
                payload.get("Networth"),
                payload.get("FaceValue"),
            ))

