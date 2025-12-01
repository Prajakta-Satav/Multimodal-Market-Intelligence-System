# app/services/index_builder.py

import os
import json
import time
import itertools
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from app.services.postgres_client import PostgresService
from app.services.minio_client import MinioService
from app.services.embeddings import EmbeddingService
from app.services.chroma_client import ChromaService
from app.core.config import settings
from app.core.logging import logger

# Initialize services
postgres = PostgresService()
minio = MinioService()
embedder = EmbeddingService(model_name="e5-large")
chroma = ChromaService(persist_dir=settings.CHROMA_PERSIST_DIR)

# Gemini API key rotation setup
gemini_key_cycle = itertools.cycle(settings.GEMINI_API_KEYS)

# Track seen questions to prevent duplicates
seen_questions: set[str] = set()

# Common output file path for all Q&A pairs
COMMON_QA_OUTPUT_PATH = os.path.join(settings.QA_DATASET_DIR, "qa_dataset.json")


def configure_gemini_key():
    """Rotate to next Gemini API key and configure client."""
    next_key = next(gemini_key_cycle)
    genai.configure(api_key=next_key)
    logger.info(f"[Gemini] Switched to API key: {next_key[:6]}...")
    return next_key


# Initial Gemini configuration
configure_gemini_key()
gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)


def chunk_text(text: str, use_sentence: bool, max_tokens: int, overlap: int) -> List[str]:
    """Chunk text using sentence-based or token-based chunking."""
    return (
        embedder.chunk_by_sentences(text, max_tokens=max_tokens, overlap_sentences=2)
        if use_sentence
        else embedder.chunk_by_tokens(text, max_tokens=max_tokens, overlap_tokens=overlap)
    )


def _normalize_question(question: str) -> str:
    """Normalize question for duplicate detection (lowercase, stripped)."""
    return question.lower().strip()


def _is_duplicate_question(question: str) -> bool:
    """Check if question has been seen before."""
    normalized = _normalize_question(question)
    if normalized in seen_questions:
        logger.warning(f"[QA Generation] Skipping duplicate question: {question[:80]}...")
        return True
    seen_questions.add(normalized)
    return False


def generate_qa_from_chunk(chunk: str, chunk_metadata: Dict) -> List[Dict]:
    """
    Generate 1-2 questions with expected answers from a chunk using Gemini.
    Filters out duplicate questions.
    """
    # Determine number of questions based on chunk size
    chunk_token_count = embedder.count_tokens(chunk)
    num_questions = 2 if chunk_token_count > settings.INDEX_BUILDER_QA_MIN_CHUNK_TOKENS else 1

    # Build the prompt
    prompt = f"""You are a financial data analysis expert. Given the following text chunk, generate {num_questions} question(s) that can be answered using ONLY the information in this chunk.

Context Type: {chunk_metadata.get('type', 'unknown')}
Chunk Text:
{chunk}

For each question:
1. The question should be specific and answerable from the chunk
2. Provide a concise expected answer (2-3 sentences max)
3. The answer must be directly extractable from the chunk

Return your response as a JSON array with this structure:
[
  {{
    "question": "specific question here",
    "expected_answer": "concise answer here"
  }}
]

Generate exactly {num_questions} question(s). Return ONLY the JSON array, no other text."""

    # Initialize response_text to None for safe error handling
    response_text = None

    # Rotate API key before call
    configure_gemini_key()

    # Recreate the model with the current key
    global gemini_model
    gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract JSON from code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Parse the JSON array
        qa_pairs = json.loads(response_text)

        # Add delay to avoid rate limits
        time.sleep(2)

        # Filter out duplicate questions and append source info
        unique_qa_pairs = []
        for qa in qa_pairs:
            question = qa.get('question', '').strip()

            if not question or _is_duplicate_question(question):
                continue

            qa['source'] = chunk
            qa['metadata'] = chunk_metadata
            unique_qa_pairs.append(qa)

        logger.info(f"[QA Generation] Generated {len(unique_qa_pairs)} unique Q&A pairs for chunk")
        return unique_qa_pairs

    except json.JSONDecodeError:
        logger.error(f"[QA Generation] JSON decode error for response: {response_text}")
    except Exception as e:
        logger.error(f"[QA Generation] Error generating Q&A: {e}")

    # Return empty list on failure
    return []


def _load_existing_qa_dataset() -> List[Dict]:
    """Load existing Q&A dataset if it exists."""
    if os.path.exists(COMMON_QA_OUTPUT_PATH):
        try:
            with open(COMMON_QA_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"[QA Generation] Loaded {len(data)} existing Q&A pairs from {COMMON_QA_OUTPUT_PATH}")
                # Populate seen_questions from existing data
                for item in data:
                    question = item.get('question', '').lower().strip()
                    if question:
                        seen_questions.add(question)
                return data
        except Exception as e:
            logger.warning(f"[QA Generation] Could not load existing Q&A dataset: {e}")
    return []


def _save_qa_dataset(qa_dataset: List[Dict]) -> None:
    """Save Q&A dataset to common output file."""
    os.makedirs(os.path.dirname(COMMON_QA_OUTPUT_PATH), exist_ok=True)
    with open(COMMON_QA_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(qa_dataset, f, indent=2, ensure_ascii=False)
    logger.info(f"[QA Generation] Saved {len(qa_dataset)} Q&A pairs to {COMMON_QA_OUTPUT_PATH}")


def index_chunks_with_qa(
    chunks: List[str],
    ids: List[str],
    metas: List[Dict],
    generate_qa: Optional[bool] = None,
    index_type: Optional[str] = None
) -> tuple[int, int, List[Dict]]:
    """
    Embeds chunks, upserts to Chroma, and optionally generates and saves Q&A pairs.
    All Q&A pairs are saved to a common output file.

    Returns number of documents, number of embeddings, and list of Q&A pairs.
    """
    if generate_qa is None:
        generate_qa = settings.INDEX_BUILDER_GENERATE_QA

    logger.info(f"[Index] Embedding {len(chunks)} chunks for {index_type}...")
    embeddings = embedder.embed(chunks, is_query=False)

    logger.info(f"[Index] Upserting to Chroma...")
    chroma.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metas,
        documents=chunks
    )

    qa_dataset = []

    if generate_qa:
        logger.info(f"[QA Generation] Generating Q&A pairs for {len(chunks)} chunks ({index_type})")

        batch_size = 20  # process every 20 chunks together
        chunk_batch_text = []
        chunk_batch_ids = []
        chunk_batch_metas = []

        for idx, (chunk, chunk_id, meta) in enumerate(zip(chunks, ids, metas)):
            chunk_batch_text.append(chunk)
            chunk_batch_ids.append(chunk_id)
            chunk_batch_metas.append(meta)

            if (idx + 1) % batch_size == 0 or (idx + 1) == len(chunks):
                combined_text = "\n\n".join(chunk_batch_text)
                combined_meta = {
                    "type": "combined",
                    "source_chunks": chunk_batch_ids,
                    "index_type": index_type
                }

                # Generate 2 questions from combined text regardless of size
                qa_pairs = generate_qa_from_chunk(combined_text, combined_meta)
                qa_pairs = qa_pairs[:2]  # limit to 2 questions max

                for qa in qa_pairs:
                    qa["combined_chunk_ids"] = chunk_batch_ids
                qa_dataset.extend(qa_pairs)

                chunk_batch_text = []
                chunk_batch_ids = []
                chunk_batch_metas = []

                time.sleep(4)  # moderate delay between batches

        logger.info(f"[QA Generation] Generated {len(qa_dataset)} Q&A pairs for {index_type} index")

    return len(chunks), len(embeddings), qa_dataset


# ========================
# Index builders
# ========================

def build_fundamentals_index(
    max_tokens: int = 256,
    overlap_tokens: int = 30,
    use_sentence: bool = False,
    generate_qa: Optional[bool] = None,
) -> tuple[int, int, List[Dict]]:
    """Build index from stock_fundamentals table with Q&A generation."""
    rows = postgres.fetch_all_fundamentals()
    logger.info(f"[Index] Building fundamentals index from {len(rows)} rows")

    all_chunks, all_ids, all_metas = [], [], []

    for r in rows:
        text = (
            f"Ticker: {r['ticker']}, Year: {r['year']}, "
            f"Revenue: {r['revenue']}, Net Income: {r['net_income']}, "
            f"EPS: {r['eps']}, P/E Ratio: {r['pe_ratio']}"
        )

        chunks = chunk_text(text, use_sentence, max_tokens, overlap_tokens)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"fundamentals_{r['id']}_{i}")
            all_metas.append({
                "type": "fundamentals",
                "ticker": r['ticker'],
                "year": str(r['year']),
                "source": "postgres:stock_fundamentals"
            })

    return index_chunks_with_qa(all_chunks, all_ids, all_metas, generate_qa, "fundamentals")


def build_prices_index(
    max_tokens: int = 256,
    overlap_tokens: int = 30,
    use_sentence: bool = False,
    generate_qa: Optional[bool] = None,
) -> tuple[int, int, List[Dict]]:
    """Build index from stock_prices table with Q&A generation."""
    rows = postgres.fetch_all_prices()
    logger.info(f"[Index] Building prices index from {len(rows)} rows")

    all_chunks, all_ids, all_metas = [], [], []

    for r in rows:
        text = (
            f"Ticker: {r['ticker']}, Date: {r['date']}, "
            f"Open: {r['open']}, High: {r['high']}, Low: {r['low']}, "
            f"Close: {r['close']}, Volume: {r['volume']}"
        )

        chunks = chunk_text(text, use_sentence, max_tokens, overlap_tokens)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"prices_{r['id']}_{i}")
            all_metas.append({
                "type": "prices",
                "ticker": r['ticker'],
                "date": str(r['date']),
                "source": "postgres:stock_prices"
            })

    return index_chunks_with_qa(all_chunks, all_ids, all_metas, generate_qa, "prices")


def build_transcripts_index(
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    use_sentence: bool = True,
    generate_qa: Optional[bool] = None,
) -> tuple[int, int, List[Dict]]:
    """Build index from transcripts table with Q&A generation."""
    rows = postgres.fetch_all_transcripts()
    logger.info(f"[Index] Building transcripts index from {len(rows)} rows")

    all_chunks, all_ids, all_metas = [], [], []

    for r in rows:
        text = r.get("text", "")
        if not text:
            continue

        chunks = chunk_text(text, use_sentence, max_tokens, overlap_tokens)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"transcript_{r['id']}_{i}")
            all_metas.append({
                "type": "transcript",
                "company": r.get("company", ""),
                "quarter": r.get("quarter", ""),
                "year": r.get("year", ""),
                "source": "postgres:transcripts"
            })

    return index_chunks_with_qa(all_chunks, all_ids, all_metas, generate_qa, "transcripts")


def index_minio_markdown(
    prefix: str,
    source_type: str,
    meta_tag: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    use_sentence: bool = True,
    generate_qa: Optional[bool] = None,
) -> tuple[int, int, List[Dict]]:
    """
    Generic function to index markdown files from MinIO with Q&A generation.
    """
    md_files = minio.list_files(bucket=settings.MINIO_BUCKET_DOCS, prefix=prefix, suffix=".md")
    logger.info(f"[Index] Found {len(md_files)} markdown files in MinIO with prefix '{prefix}'")

    all_chunks, all_ids, all_metas = [], [], []

    for path in md_files:
        text = minio.download_as_text(settings.MINIO_BUCKET_DOCS, path)
        chunks = chunk_text(text, use_sentence, max_tokens, overlap_tokens)

        filename = os.path.basename(path)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{source_type}_{filename}_{i}")
            all_metas.append({
                "type": source_type,
                "file": filename,
                "source": meta_tag
            })

    return index_chunks_with_qa(all_chunks, all_ids, all_metas, generate_qa, source_type)


def build_presentations_index(
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    use_sentence: bool = True,
    generate_qa: Optional[bool] = None,
) -> tuple[int, int, List[Dict]]:
    """Build index from presentation markdown files in MinIO with Q&A generation."""
    return index_minio_markdown(
        prefix="ppt_json/",
        source_type="ppt",
        meta_tag="minio:ppt",
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        use_sentence=use_sentence,
        generate_qa=generate_qa,
    )


def build_earnings_release_index(
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    use_sentence: bool = True,
    generate_qa: Optional[bool] = None,
) -> tuple[int, int, List[Dict]]:
    """Build index from earnings release markdown files in MinIO with Q&A generation."""
    return index_minio_markdown(
        prefix="earnings_release_json/",
        source_type="earnings_release",
        meta_tag="minio:earnings_release",
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        use_sentence=use_sentence,
        generate_qa=generate_qa,
    )


def build_all_indexes(generate_qa: bool = True) -> Dict[str, tuple[int, int, List[Dict]]]:
    """Builds all indexes with optional Q&A generation."""
    logger.info("[Index] Starting full index build")

    # Reset duplicate tracker
    global seen_questions
    seen_questions.clear()
    logger.info("[Index] Cleared duplicate question tracker")

    # Check if common Q&A dataset already exists
    if os.path.exists(COMMON_QA_OUTPUT_PATH) and generate_qa:
        logger.info(f"[Index] Common Q&A dataset already exists at {COMMON_QA_OUTPUT_PATH}")
        logger.info("[Index] Skipping Q&A generation for all indexes")
        generate_qa = False
    else:
        # Load existing Q&A pairs to avoid duplicates
        _load_existing_qa_dataset()

    all_qa_pairs = []

    results = {}
    
    logger.info("[Index] Building fundamentals index...")
    results['fundamentals'] = build_fundamentals_index(generate_qa=generate_qa)
    all_qa_pairs.extend(results['fundamentals'][2])

    logger.info("[Index] Building prices index...")
    results['prices'] = build_prices_index(generate_qa=generate_qa)
    all_qa_pairs.extend(results['prices'][2])

    logger.info("[Index] Building transcripts index...")
    results['transcripts'] = build_transcripts_index(generate_qa=generate_qa)
    all_qa_pairs.extend(results['transcripts'][2])

    logger.info("[Index] Building presentations index...")
    results['presentations'] = build_presentations_index(generate_qa=generate_qa)
    all_qa_pairs.extend(results['presentations'][2])

    logger.info("[Index] Building earnings release index...")
    results['earnings_release'] = build_earnings_release_index(generate_qa=generate_qa)
    all_qa_pairs.extend(results['earnings_release'][2])

    total_docs = sum(r[0] for r in results.values())
    total_qa = sum(len(r[2]) for r in results.values())

    # Save all Q&A pairs to common output file
    if generate_qa and all_qa_pairs:
        _save_qa_dataset(all_qa_pairs)

    logger.info(f"[Index] Build complete: {total_docs} total documents, {total_qa} unique Q&A pairs generated")
    logger.info(f"[Index] All Q&A pairs saved to {COMMON_QA_OUTPUT_PATH}")

    return results


def merge_qa_datasets(output_path: Optional[str] = None) -> List[Dict]:
    """
    Merge all Q&A datasets into a single golden dataset for evaluation.
    Since we now have a single common file, this loads and returns it.
    """
    if output_path is None:
        output_path = COMMON_QA_OUTPUT_PATH

    logger.info(f"[QA Merge] Loading Q&A dataset from {output_path}")

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"[QA Merge] Loaded {len(data)} Q&A pairs")
                return data
        except Exception as e:
            logger.error(f"[QA Merge] Could not load Q&A dataset: {e}")
    else:
        logger.warning(f"[QA Merge] Q&A dataset not found at {output_path}")

    return []
