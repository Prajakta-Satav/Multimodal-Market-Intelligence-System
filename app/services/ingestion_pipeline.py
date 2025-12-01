import os
import json
import pandas as pd
from datetime import datetime
from app.core.config import settings
from app.core.logging import logger
from app.services.minio_client import MinioService
from app.services.postgres_client import PostgresService

minio = MinioService()
postgres = PostgresService()

# ============================================================
# HELPERS
# ============================================================

def upload_to_minio(local_path: str, object_path: str, citations: list, label: str, tag="minio:file"):
    """Upload file to MinIO + record citation."""
    try:
        minio_path = minio.upload_file(
            local_path,
            bucket=settings.MINIO_BUCKET_DOCS,
            object_name=object_path
        )
        citations.append({
            "tag": tag,
            "label": label,
            "minio": f"minio:{minio_path}"
        })
        logger.info(f"[Upload] Uploaded {label} to {minio_path}")
        return minio_path
    except Exception as e:
        logger.error(f"[Upload] Failed to upload {local_path}: {e}")
        raise

def json_to_markdown(data, level=1):
    """Convert nested JSON into Markdown recursively."""
    md = []
    prefix = "#" * min(level + 1, 6)  # Limit heading depth to h6
    
    if isinstance(data, dict):
        for k, v in data.items():
            # Skip empty values
            if v is None or v == "":
                continue
            md.append(f"{prefix} {k.replace('_', ' ').title()}\n")
            md.append(json_to_markdown(v, level + 1))
    
    elif isinstance(data, list):
        if not data:  # Skip empty lists
            return ""
        for i, item in enumerate(data, start=1):
            md.append(f"{prefix} Item {i}\n")
            md.append(json_to_markdown(item, level + 1))
    
    else:
        # Convert to string and escape if needed
        md.append(f"{str(data)}\n\n")
    
    return "\n".join(md)

def convert_and_upload_json_folder(base_folder: str, minio_prefix: str, tag: str):
    """Generic handler for folders containing JSON → MD → MinIO."""
    processed, citations = 0, []
    
    if not os.path.exists(base_folder):
        logger.warning(f"[Ingest] Folder not found: {base_folder}")
        return processed, citations
    
    for root, _, files in os.walk(base_folder):
        for file in files:
            if not file.endswith(".json"):
                continue
            
            try:
                file_path = os.path.join(root, file)
                
                # Read JSON
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Convert to Markdown
                md_content = f"# {file.replace('.json', '').replace('_', ' ').title()}\n\n"
                md_content += json_to_markdown(data)
                md_name = file.replace(".json", ".md")
                md_path = os.path.join(root, md_name)
                
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
                
                # Upload both files
                upload_to_minio(
                    file_path,
                    f"{minio_prefix}/{file}",
                    citations,
                    f"{tag} JSON {file}",
                    tag=tag
                )
                upload_to_minio(
                    md_path,
                    f"{minio_prefix}/{md_name}",
                    citations,
                    f"{tag} MD {md_name}",
                    tag=tag
                )
                
                processed += 1
                logger.info(f"[Ingest] Processed {file} → {md_name}")
            
            except Exception as e:
                logger.error(f"[ERROR] Failed processing {file}: {e}")
    
    return processed, citations

# ============================================================
# INGESTORS
# ============================================================

def ingest_stocks():
    """
    Ingest stock price and fundamental data from CSV files.
    
    Expected CSV columns:
    - Date, Ticker, Open, High, Low, Close, Volume
    - Quarter, MarketCap, PE_Ratio, EPS, Revenue, Profit, etc.
    """
    folder = os.path.join(settings.DATA_FOLDER, "Stocks")
    processed, citations = 0, []
    
    if not os.path.exists(folder):
        logger.warning(f"[Ingest] Stocks folder not found: {folder}")
        return processed, citations
    
    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue
        
        try:
            ticker = file.split(".")[0].upper()
            df = pd.read_csv(os.path.join(folder, file))
            
            logger.info(f"[Ingest] Processing stock data for {ticker} ({len(df)} rows)")
            
            # Process each row
            for idx, row in df.iterrows():
                # Insert price data
                try:
                    postgres.insert_stock_price(
                        ticker=row.get("Ticker", ticker),
                        date=row["Date"],
                        open_price=row.get("Open"),
                        high=row.get("High"),
                        low=row.get("Low"),
                        close=row.get("Close"),
                        volume=row.get("Volume")
                    )
                except Exception as e:
                    logger.error(f"[Ingest] Failed to insert price for {ticker} on {row.get('Date')}: {e}")
                
                # Insert fundamental data if available
                if "Revenue" in row or "EPS" in row:
                    try:
                        # Extract year from date (assuming YYYY-MM-DD format)
                        year = str(row["Date"]).split("-")[0] if "Date" in row else None
                        
                        postgres.insert_fundamental(
                            ticker=row.get("Ticker", ticker),
                            year=year,
                            revenue=row.get("Revenue"),
                            net_income=row.get("Profit"),
                            eps=row.get("EPS"),
                            pe_ratio=row.get("PE_Ratio")
                        )
                    except Exception as e:
                        logger.error(f"[Ingest] Failed to insert fundamentals for {ticker}: {e}")
            
            # Upload raw CSV to MinIO
            upload_to_minio(
                os.path.join(folder, file),
                f"stocks/{file}",
                citations,
                f"Stock Data {ticker}",
                tag="postgres:stocks"
            )
            
            processed += 1
        
        except Exception as e:
            logger.error(f"[ERROR] Stock ingest failed ({file}): {e}")
    
    return processed, citations

def ingest_transcripts():
    """
    Ingest earnings call transcripts.
    
    Expected filename format: CompanyName-Q1-2024.txt
    """
    folder = os.path.join(settings.DATA_FOLDER, "Transcript")
    processed, citations = 0, []
    
    if not os.path.exists(folder):
        logger.warning(f"[Ingest] Transcripts folder not found: {folder}")
        return processed, citations
    
    for file in os.listdir(folder):
        try:
            filename = os.path.splitext(file)[0]
            parts = filename.split("-")
            
            company = parts[0] if len(parts) > 0 else "Unknown"
            quarter = parts[1] if len(parts) > 1 else None
            year = parts[2] if len(parts) > 2 else None
            
            path = os.path.join(folder, file)
            
            # Read transcript text
            with open(path, encoding="utf-8") as f:
                text = f.read()
            
            # Upload to MinIO first
            minio_path = upload_to_minio(
                path,
                f"transcripts/{file}",
                citations,
                f"Transcript {filename}",
                tag="postgres:transcript"
            )
            
            # Insert into Postgres with year field
            postgres.insert_transcript(
                company=company,
                quarter=quarter,
                year=year,
                text=text,
                file_path=minio_path
            )
            
            processed += 1
            logger.info(f"[Ingest] Processed transcript: {filename}")
        
        except Exception as e:
            logger.error(f"[ERROR] Transcript ingest failed ({file}): {e}")
    
    return processed, citations

def ingest_balance_sheets():
    """
    Ingest balance sheet data from JSON files.
    
    Expected JSON structure:
    {
        "ticker": "AAPL",
        "year": "2024",
        "total_assets": 1234567,
        "total_liabilities": 987654,
        "stockholder_equity": 246913
    }
    """
    folder = os.path.join(settings.DATA_FOLDER, "Balance_Sheets")
    processed, citations = 0, []
    
    if not os.path.exists(folder):
        logger.warning(f"[Ingest] Balance sheets folder not found: {folder}")
        return processed, citations
    
    for file in os.listdir(folder):
        if not file.endswith(".json"):
            continue
        
        try:
            path = os.path.join(folder, file)
            
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            
            # Upload to MinIO
            upload_to_minio(
                path,
                f"balance_sheets/{file}",
                citations,
                f"Balance Sheet {file}",
                tag="postgres:balance_sheets"
            )
            
            # Insert into Postgres
            postgres.insert_balance_sheet(
                ticker=data.get("ticker"),
                year=str(data.get("year")),
                total_assets=data.get("total_assets"),
                total_liabilities=data.get("total_liabilities"),
                stockholder_equity=data.get("stockholder_equity")
            )
            
            processed += 1
            logger.info(f"[Ingest] Processed balance sheet: {file}")
        
        except Exception as e:
            logger.error(f"[ERROR] Balance sheet ingest failed ({file}): {e}")
    
    return processed, citations

def ingest_presentations():
    """
    Ingest presentation data from JSON files.
    
    Expected JSON structure:
    {
        "company": "Apple",
        "quarter": "Q1",
        "year": "2024",
        "slides": [
            {
                "slide_number": 1,
                "text": "...",
                "chart_description": "..."
            }
        ]
    }
    """
    folder = os.path.join(settings.DATA_FOLDER, "PPT_Json")
    processed, citations = 0, []
    
    if not os.path.exists(folder):
        logger.warning(f"[Ingest] Presentations folder not found: {folder}")
        return processed, citations
    
    for file in os.listdir(folder):
        if not file.endswith(".json"):
            continue
        
        try:
            path = os.path.join(folder, file)
            
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            
            # Upload to MinIO
            minio_path = upload_to_minio(
                path,
                f"presentations/{file}",
                citations,
                f"Presentation {file}",
                tag="postgres:presentations"
            )
            
            # Insert each slide into Postgres
            company = data.get("company", "Unknown")
            quarter = data.get("quarter")
            year = data.get("year")
            slides = data.get("slides", [])
            
            for slide in slides:
                postgres.insert_presentation(
                    company=company,
                    quarter=quarter,
                    year=year,
                    slide_number=slide.get("slide_number"),
                    slide_text=slide.get("text", ""),
                    chart_description=slide.get("chart_description", ""),
                    file_path=minio_path
                )
            
            processed += 1
            logger.info(f"[Ingest] Processed presentation: {file} ({len(slides)} slides)")
        
        except Exception as e:
            logger.error(f"[ERROR] Presentation ingest failed ({file}): {e}")
    
    return processed, citations

def ingest_audio():
    """Upload audio files to MinIO (audio bucket)."""
    folder = os.path.join(settings.DATA_FOLDER, "Audio")
    processed, citations = 0, []
    
    if not os.path.exists(folder):
        logger.warning(f"[Ingest] Audio folder not found: {folder}")
        return processed, citations
    
    for file in os.listdir(folder):
        try:
            path = os.path.join(folder, file)
            
            # Upload to audio bucket
            minio.upload_file(
                path,
                bucket="audio",  # Separate bucket for audio
                object_name=file
            )
            
            citations.append({
                "tag": "minio:audio",
                "label": f"Audio File {file}",
                "minio": f"minio:audio/{file}"
            })
            
            processed += 1
            logger.info(f"[Ingest] Uploaded audio: {file}")
        
        except Exception as e:
            logger.error(f"[ERROR] Audio ingest failed ({file}): {e}")
    
    return processed, citations

def ingest_ppt_json():
    """Convert PPT JSON files to Markdown and upload to MinIO."""
    return convert_and_upload_json_folder(
        os.path.join(settings.DATA_FOLDER, "PPT_Json"),
        "ppt_json",
        "minio:ppt"
    )

def ingest_earnings_release_json():
    """Convert earnings release JSON files to Markdown and upload to MinIO."""
    return convert_and_upload_json_folder(
        os.path.join(settings.DATA_FOLDER, "Earnings_release_JSON"),
        "earnings_release_json",
        "minio:earnings_release"
    )

# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_pipeline():
    """Run complete data ingestion pipeline."""
    logger.info("[Pipeline] ========================================")
    logger.info("[Pipeline] Starting unified ingestion pipeline...")
    logger.info("[Pipeline] ========================================")
    
    total_processed, all_citations = 0, []
    results = {}
    
    tasks = [
        ("Stocks", ingest_stocks),
        ("Transcripts", ingest_transcripts),
        ("Balance Sheets", ingest_balance_sheets),
        ("Presentations", ingest_presentations),
        ("PPT JSON", ingest_ppt_json),
        ("Audio", ingest_audio),
        ("Earnings Release JSON", ingest_earnings_release_json),
    ]
    
    for task_name, task_func in tasks:
        try:
            logger.info(f"[Pipeline] Starting: {task_name}")
            processed, citations = task_func()
            total_processed += processed
            all_citations.extend(citations)
            results[task_name] = {"processed": processed, "citations_count": len(citations)}
            logger.info(f"[Pipeline] Completed: {task_name} ({processed} items)")
        except Exception as e:
            logger.error(f"[Pipeline] ERROR in {task_name}: {e}")
            results[task_name] = {"processed": 0, "error": str(e)}
    
    logger.info("[Pipeline] ========================================")
    logger.info(f"[Pipeline] Pipeline completed. Total files processed: {total_processed}")
    logger.info("[Pipeline] ========================================")
    
    return {
        "status": "completed",
        "total_processed": total_processed,
        "total_citations": len(all_citations),
        "results": results,
        "citations": all_citations,
        "timestamp": datetime.utcnow().isoformat()
    }
