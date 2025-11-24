# app/services/ingestion_pipeline.py
import os, json
import pandas as pd
from app.core.config import settings
from app.core.logging import logger
from app.services.minio_client import MinioService
from app.services.postgres_client import PostgresService

minio = MinioService()
postgres = PostgresService()

def ingest_stocks():
    folder = os.path.join(settings.DATA_FOLDER, "Stocks")
    processed, citations = 0, []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            ticker = file.split(".")[0].upper()

            for _, row in df.iterrows():
                # --- Insert OHLCV into stock_prices ---
                price_payload = {
                    "ticker": row.get("Ticker") or ticker,
                    "date": row.get("Date"),
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                }
                postgres.insert_stock(price_payload)

                # --- Insert fundamentals into stock_fundamentals ---
                fundamentals_payload = {
                    "ticker": row.get("Ticker") or ticker,
                    "date": row.get("Date"),
                    "quarter": row.get("Quarter"),
                    "market_cap": row.get("MarketCap"),
                    "pb_ratio": row.get("PB_Ratio"),
                    "pe_ratio": row.get("PE_Ratio"),
                    "peg_ratio": row.get("PEG_Ratio"),
                    "price_to_sales": row.get("PriceToSales"),
                    "eps": row.get("EPS"),
                    "roe": row.get("ROE"),
                    "roa": row.get("ROA"),
                    "beta": row.get("Beta"),
                    "dividend_yield": row.get("DividendYield"),
                    "debt_to_equity": row.get("DebtToEquity"),
                    "book_value": row.get("BookValue"),
                    "ebitda": row.get("EBITDA"),
                    "revenue": row.get("Revenue"),
                    "profit": row.get("Profit"),
                    "networth": row.get("Networth"),
                    "face_value": row.get("FaceValue"),
                }
                postgres.insert_stock_fundamentals(fundamentals_payload)

            # --- Upload raw CSV to MinIO ---
            minio_path = minio.upload_file(
                os.path.join(folder, file),
                bucket=settings.MINIO_BUCKET_DOCS,
                object_name=f"stocks/{file}"
            )

            # --- Add citations for both tables ---
            citations.append({
                "tag": "postgres:stock_prices",
                "label": f"Stock prices {ticker} ({file})",
                "minio": f"minio:{minio_path}"
            })
            citations.append({
                "tag": "postgres:stock_fundamentals",
                "label": f"Stock fundamentals {ticker} ({file})",
                "minio": f"minio:{minio_path}"
            })
            processed += 1

    return processed, citations


def ingest_transcripts():
    folder = os.path.join(settings.DATA_FOLDER, "Transcript")
    processed, citations = 0, []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        filename = os.path.splitext(file)[0] 
        file_parts = filename.split("-")
       
        company = file_parts[0] if len(file_parts) > 0 else "Unknown"
        quarter = file_parts[1] if len(file_parts) > 1 else "Unknown"
        year = file_parts[2] #if len(file_parts) > 2 else "Unknown"
        
        print(f"DEBUG: company={company}, quarter={quarter}, year={year}")

        minio_path = minio.upload_file(
            path,
            bucket="docs",
            object_name=f"transcripts/{filename}"
        )
       
        postgres.insert_transcript({
            "company": company,
            "quarter": quarter,
            "year": year,
            "text": open(path, encoding="utf-8").read(),
            "file_path": minio_path,
        })

        citations.append({
            "tag": "postgres:transcripts",
            "label": f"Transcript {filename}",
            "minio": f"minio:{minio_path}",
        })
        processed += 1

    return processed, citations


def ingest_balance_sheets():
    folder = os.path.join(settings.DATA_FOLDER, "Earnings_Release")
    processed, citations = 0, []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            path = os.path.join(folder, file)
            data = json.load(open(path))
            minio_path = minio.upload_file(path, bucket="docs", object_name=f"earnings/{file}")
            postgres.insert_balance_sheet({
                "company": data.get("company", "Unknown"),
                "quarter": data.get("quarter", "Unknown"),
                "json_data": json.dumps(data),
                "file_path": minio_path
            })
            citations.append({
                "tag": "postgres:balance_sheets",
                "label": f"Earnings release {file}",
                "minio": f"minio:{minio_path}"
            })
            processed += 1
    return processed, citations

def ingest_ppt_json():
    base_folder = os.path.join(settings.DATA_FOLDER, "PPT_Json")
    processed, citations = 0, []
    for subdir in os.listdir(base_folder):
        subpath = os.path.join(base_folder, subdir)
        if os.path.isdir(subpath):
            for file in os.listdir(subpath):
                if file.endswith(".json"):
                    path = os.path.join(subpath, file)
                    data = json.load(open(path))
                    minio_path = minio.upload_file(path, bucket="docs", object_name=f"ppt_json/{subdir}/{file}")
                    
                    
                    
                    
                    postgres.insert_presentation({
                        "company": subdir.split("_")[0],
                        "quarter": subdir.split("_")[1] if "_" in subdir else "Unknown",
                        "slides": json.dumps(data),
                        "file_path": minio_path
                    })
                    citations.append({
                        "tag": "postgres:presentations",
                        "label": f"PPT JSON {subdir}",
                        "minio": f"minio:{minio_path}"
                    })
                    processed += 1
    return processed, citations

def ingest_audio():
    folder = os.path.join(settings.DATA_FOLDER, "Audio")
    processed, citations = 0, []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        minio_path = minio.upload_file(path, bucket="docs", object_name=f"audio/{file}")
        citations.append({
            "tag": "minio:audio",
            "label": f"Audio file {file}",
            "minio": f"minio:{minio_path}"
        })
        processed += 1
    return processed, citations

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


def run_pipeline():
    logger.info("[Pipeline] Starting unified ingestion...")
    total_processed, all_citations = 0, []

    for func in [ingest_stocks, ingest_transcripts, ingest_balance_sheets, ingest_ppt_json, ingest_audio]:
        processed, citations = func()
        total_processed += processed
        all_citations.extend(citations)

    logger.info(f"[Pipeline] Completed. Files processed: {total_processed}")
    return {"processed": total_processed, "citations": all_citations}
