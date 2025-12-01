"""
ChromaDB Q&A Generator V2 - Enhanced
Generates high-quality questions ONLY from ChromaDB embeddings/chunks
Features:
  ✅ Smart Q&A generation from chunks
  ✅ Metadata-based filtering
  ✅ Batch collection querying
  ✅ Quality scoring
  ✅ Multi-format output
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import chromadb
from dotenv import load_dotenv
import google.generativeai as genai


# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Config:
    """ChromaDB Query Configuration"""
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # ChromaDB
    CHROMA_DB_PATH: str = os.getenv(
        "CHROMA_DB_PATH",
        r"D:\Multimodal-Market-Intelligence-System\data\chroma"
    )
    
    # Query parameters
    CHUNKS_PER_COLLECTION: int = int(os.getenv("CHUNKS_PER_COLLECTION", "50"))
    QUESTIONS_PER_CHUNK: int = int(os.getenv("QUESTIONS_PER_CHUNK", "2"))
    QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", "0.75"))
    
    # Output
    OUTPUT_FOLDER: Path = Path(os.getenv(
        "OUTPUT_FOLDER",
        r"D:\Multimodal-Market-Intelligence-System\ground_truth"
    ))
    
    # Filtering (optional)
    FILTER_COMPANY: Optional[str] = os.getenv("FILTER_COMPANY", None)  # e.g., "Amazon"
    FILTER_QUARTER: Optional[str] = os.getenv("FILTER_QUARTER", None)  # e.g., "Q1"
    
    @classmethod
    def validate(cls) -> bool:
        if not os.path.exists(cls.CHROMA_DB_PATH):
            logger.error(f"❌ ChromaDB path not found: {cls.CHROMA_DB_PATH}")
            return False
        
        cls.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        return True


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ChromaDBError(Exception):
    pass


# ============================================================================
# CHROMADB Q&A GENERATOR V2
# ============================================================================

class EnhancedChromaDBGenerator:
    """
    V2: High-quality Q&A generation from ChromaDB embeddings only
    """
    
    # Financial keywords for content analysis
    FINANCIAL_KEYWORDS = {
        'revenue': ['revenue', 'sales', 'net revenue', 'total sales'],
        'profit': ['net income', 'profit', 'earnings', 'net profit'],
        'growth': ['grew', 'growth', 'increased', 'expansion', 'growth rate'],
        'margin': ['margin', 'margin percentage', 'operating margin', 'gross margin'],
        'cash': ['cash flow', 'operating cash', 'free cash flow', 'cash position'],
        'segment': ['segment', 'aws', 'cloud', 'retail', 'advertising', 'services'],
        'guidance': ['guidance', 'outlook', 'forecast', 'expects', 'projects'],
        'ratio': ['ratio', 'pe ratio', 'debt ratio', 'current ratio'],
    }
    
    # Q&A question templates by content type
    QUESTION_TEMPLATES = {
        'revenue': [
            "What was {company}'s revenue in {period}?",
            "How much revenue did {company} generate in {period}?",
            "What were {company}'s total sales in {period}?",
        ],
        'profit': [
            "What was {company}'s net income in {period}?",
            "What profit did {company} report for {period}?",
            "What were {company}'s earnings in {period}?",
        ],
        'growth': [
            "What was {company}'s growth rate in {period}?",
            "How much did {company} grow in {period}?",
            "What was the growth percentage for {company} in {period}?",
        ],
        'margin': [
            "What was {company}'s margin in {period}?",
            "What margin did {company} achieve in {period}?",
        ],
        'segment': [
            "What was {company}'s {segment} performance in {period}?",
            "How did {company}'s {segment} segment perform in {period}?",
        ],
        'guidance': [
            "What guidance did {company} provide in {period}?",
            "What was {company}'s outlook for the upcoming period?",
        ],
    }
    
    def __init__(self):
        self.config = Config
        
        if not self.config.validate():
            raise ChromaDBError("Invalid configuration")
        
        self.client = None
        self.qa_pairs: List[Dict] = []
        self.questions_set: Set[str] = set()
        
        self._init_chromadb()
        self._init_gemini()
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB connection"""
        try:
            self.client = chromadb.PersistentClient(path=self.config.CHROMA_DB_PATH)
            collections = self.client.list_collections()
            
            logger.info(f"✅ ChromaDB connected: {self.config.CHROMA_DB_PATH}")
            logger.info(f"📊 Collections found: {len(collections)}")
            for coll in collections:
                logger.info(f"   - {coll.name} ({coll.count()} chunks)")
        
        except Exception as e:
            logger.error(f"❌ ChromaDB connection failed: {e}")
            raise ChromaDBError(f"Failed to connect to ChromaDB: {e}")
    
    def _init_gemini(self) -> None:
        """Initialize Gemini API"""
        try:
            if self.config.GEMINI_API_KEY:
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
                logger.info(f"✅ Gemini API initialized")
            else:
                logger.warning("⚠️  Gemini API not configured - using basic extraction")
                self.model = None
        except Exception as e:
            logger.warning(f"⚠️  Gemini initialization failed: {e}")
            self.model = None
    
    # ========================================================================
    # CHROMADB RETRIEVAL
    # ========================================================================
    
    def get_all_collections(self) -> List[str]:
        """Get all collection names"""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"❌ Error listing collections: {e}")
            return []
    
    def get_chunks_from_collection(self, collection_name: str, 
                                   limit: int = None,
                                   where: Dict = None) -> List[Dict]:
        """
        Retrieve chunks from a collection with optional filtering
        
        Args:
            collection_name: Name of collection
            limit: Max chunks to retrieve
            where: Optional metadata filter (e.g., {"company": "Amazon"})
        
        Returns:
            List of chunks with metadata and IDs
        """
        limit = limit or self.config.CHUNKS_PER_COLLECTION
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Retrieve with optional metadata filter
            if where:
                results = collection.get(
                    limit=limit,
                    where=where,
                    include=["documents", "metadatas"]
                )
            else:
                results = collection.get(
                    limit=limit,
                    include=["documents", "metadatas"]
                )
            
            # Format results
            chunks = []
            for i, chunk_id in enumerate(results['ids']):
                chunks.append({
                    'id': chunk_id,
                    'chunk_id': chunk_id,
                    'document': results['documents'][i] if results['documents'] else '',
                    'metadata': results['metadatas'][i] if results['metadatas'] else {},
                })
            
            logger.info(f"✅ Retrieved {len(chunks)} chunks from {collection_name}")
            return chunks
        
        except Exception as e:
            logger.error(f"❌ Error retrieving from {collection_name}: {e}")
            return []
    
    def get_chunks_all_collections(self) -> Dict[str, List[Dict]]:
        """Retrieve chunks from all collections"""
        all_chunks = {}
        
        collections = self.get_all_collections()
        logger.info(f"🔍 Retrieving from {len(collections)} collections...")
        
        for coll_name in collections:
            chunks = self.get_chunks_from_collection(coll_name)
            if chunks:
                all_chunks[coll_name] = chunks
        
        return all_chunks
    
    def query_collections_semantic(self, query_text: str, 
                                   collection_names: List[str] = None,
                                   num_results: int = 5) -> List[Dict]:
        """
        Semantic search across collections
        
        Args:
            query_text: Search query (e.g., "revenue growth")
            collection_names: Optional list of collections to search
            num_results: Results per collection
        
        Returns:
            List of relevant chunks
        """
        if not collection_names:
            collection_names = self.get_all_collections()
        
        all_results = []
        
        for coll_name in collection_names:
            try:
                collection = self.client.get_collection(name=coll_name)
                results = collection.query(
                    query_texts=[query_text],
                    n_results=num_results,
                    include=["documents", "metadatas"]
                )
                
                # Format and add results
                if results['ids'] and len(results['ids']) > 0:
                    for i, chunk_id in enumerate(results['ids'][0]):
                        all_results.append({
                            'id': chunk_id,
                            'chunk_id': chunk_id,
                            'document': results['documents'][0][i] if results['documents'] else '',
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'collection': coll_name,
                        })
            
            except Exception as e:
                logger.debug(f"⚠️  Error querying {coll_name}: {e}")
        
        logger.info(f"✅ Found {len(all_results)} semantic matches")
        return all_results
    
    # ========================================================================
    # CONTENT ANALYSIS & CLASSIFICATION
    # ========================================================================
    
    def classify_chunk_content(self, chunk: Dict) -> Tuple[str, float]:
        """
        Classify chunk by financial content type
        
        Returns:
            (content_type, confidence)
        """
        document = chunk.get('document', '').lower()
        
        scores = {}
        for content_type, keywords in self.FINANCIAL_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in document)
            scores[content_type] = matches / len(keywords)
        
        if not scores:
            return ('generic', 0.5)
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return (best_type, confidence)
    
    def extract_period_info(self, chunk: Dict) -> str:
        """Extract period (Q1, Q2, etc.) from metadata or document"""
        metadata = chunk.get('metadata', {})
        
        # Try metadata first
        if 'quarter' in metadata:
            quarter = metadata['quarter']
            year = metadata.get('year', '2024')
            return f"{quarter} {year}"
        
        # Try document
        document = chunk.get('document', '')
        period_match = re.search(r'(Q[1-4])\s+(20\d{2})', document, re.IGNORECASE)
        if period_match:
            return f"{period_match.group(1)} {period_match.group(2)}"
        
        return "2024"
    
    def extract_company_name(self, chunk: Dict) -> str:
        """Extract company name from metadata or document"""
        metadata = chunk.get('metadata', {})
        
        # Try metadata
        if 'company' in metadata:
            return metadata['company']
        
        # Try document for common company names
        document = chunk.get('document', '')
        companies = ['Amazon', 'Microsoft', 'Google', 'Alphabet', 'Apple']
        for company in companies:
            if company.lower() in document.lower():
                return company
        
        return "Company"
    
    # ========================================================================
    # INTELLIGENT Q&A GENERATION
    # ========================================================================
    
    def generate_qa_with_gemini(self, chunk: Dict) -> Optional[Dict]:
        """
        Generate high-quality Q&A using Gemini
        """
        try:
            document = chunk.get('document', '')
            chunk_id = chunk.get('chunk_id', '')
            
            if len(document) < 50:
                return None
            
            company = self.extract_company_name(chunk)
            period = self.extract_period_info(chunk)
            
            prompt = f"""You are a financial analyst. From the chunk below, generate a specific, 
high-quality question-answer pair that can be used to evaluate financial knowledge.

CHUNK:
{document[:800]}

METADATA:
- Company: {company}
- Period: {period}
- Collection: {chunk.get('collection', 'unknown')}

REQUIREMENTS:
1. Question must be specific and answerable from the chunk
2. Answer must be exact quote or key fact from chunk
3. Question should test understanding, not just recall
4. Both must be concise (question <20 words, answer <50 words)

Format as JSON:
{{
  "question": "...",
  "answer": "...",
  "type": "factual|analytical|comparative"
}}
"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                qa_json = json.loads(json_match.group(0))
                return {
                    'question': qa_json.get('question', ''),
                    'answer': qa_json.get('answer', ''),
                    'chunk_id': chunk_id,
                    'company': company,
                    'period': period,
                    'qa_type': qa_json.get('type', 'factual'),
                    'source': chunk.get('collection', ''),
                    'quality_score': 0.90,
                }
        
        except Exception as e:
            logger.debug(f"⚠️  Gemini Q&A generation failed: {e}")
        
        return None
    
    def generate_qa_template_based(self, chunk: Dict) -> Optional[Dict]:
        """
        Generate Q&A using template matching
        """
        try:
            document = chunk.get('document', '')
            chunk_id = chunk.get('chunk_id', '')
            
            if len(document) < 30:
                return None
            
            # Classify content
            content_type, confidence = self.classify_chunk_content(chunk)
            
            # Skip if confidence too low
            if confidence < 0.3:
                return None
            
            company = self.extract_company_name(chunk)
            period = self.extract_period_info(chunk)
            
            # Get templates for this content type
            templates = self.QUESTION_TEMPLATES.get(content_type, [])
            if not templates:
                return None
            
            # Use first template
            question_template = templates[0]
            
            # Extract answer from document (first sentence)
            sentences = [s.strip() for s in re.split(r'[.!?]', document) if s.strip()]
            answer = sentences[0][:150] if sentences else document[:150]
            
            # Format question
            if '{segment}' in question_template:
                # Extract segment name
                segment_keywords = ['aws', 'cloud', 'retail', 'advertising', 'services']
                segment = 'key'
                for kw in segment_keywords:
                    if kw in document.lower():
                        segment = kw.upper()
                        break
                question = question_template.format(
                    company=company,
                    period=period,
                    segment=segment
                )
            else:
                question = question_template.format(
                    company=company,
                    period=period
                )
            
            quality_score = 0.75 + (confidence * 0.15)
            
            return {
                'question': question,
                'answer': answer,
                'chunk_id': chunk_id,
                'company': company,
                'period': period,
                'qa_type': content_type,
                'source': chunk.get('collection', ''),
                'quality_score': quality_score,
            }
        
        except Exception as e:
            logger.debug(f"⚠️  Template-based generation failed: {e}")
        
        return None
    
    def generate_qa_from_chunk(self, chunk: Dict) -> Optional[Dict]:
        """
        Generate Q&A from chunk - tries Gemini first, then template fallback
        """
        # Try Gemini first (better quality)
        if self.model:
            qa = self.generate_qa_with_gemini(chunk)
            if qa and qa.get('question') and qa.get('answer'):
                return qa
        
        # Fallback to template-based
        qa = self.generate_qa_template_based(chunk)
        if qa and qa.get('question') and qa.get('answer'):
            return qa
        
        return None
    
    def add_qa(self, qa: Dict) -> bool:
        """Add Q&A with duplicate checking"""
        q_normalized = qa.get('question', '').lower().strip()
        
        if q_normalized in self.questions_set:
            return False
        
        self.questions_set.add(q_normalized)
        self.qa_pairs.append(qa)
        return True
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def process_chunks_batch(self, chunks: List[Dict], 
                            questions_per_chunk: int = None) -> int:
        """Process batch of chunks and generate Q&A"""
        questions_per_chunk = questions_per_chunk or self.config.QUESTIONS_PER_CHUNK
        count = 0
        
        logger.info(f"📝 Processing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks, 1):
            for _ in range(questions_per_chunk):
                qa = self.generate_qa_from_chunk(chunk)
                
                if qa:
                    quality = qa.get('quality_score', 0.75)
                    
                    # Check quality threshold
                    if quality >= self.config.QUALITY_THRESHOLD:
                        if self.add_qa(qa):
                            count += 1
        
        logger.info(f"   ✅ Generated {count} quality Q&A pairs")
        return count
    
    # ========================================================================
    # METADATA FILTERING
    # ========================================================================
    
    def build_where_filter(self) -> Optional[Dict]:
        """Build ChromaDB where filter from config"""
        where = {}
        
        if self.config.FILTER_COMPANY:
            where['company'] = self.config.FILTER_COMPANY
        
        if self.config.FILTER_QUARTER:
            where['quarter'] = self.config.FILTER_QUARTER
        
        return where if where else None
    
    # ========================================================================
    # OUTPUT & EXPORT
    # ========================================================================
    
    def format_qa_display(self) -> None:
        """Display Q&A pairs"""
        logger.info("\n" + "="*80)
        logger.info("GENERATED Q&A PAIRS FROM CHROMADB")
        logger.info("="*80 + "\n")
        
        for i, qa in enumerate(self.qa_pairs[:20], 1):  # Show first 20
            print(f"{i}. CHUNK ID: {qa['chunk_id']}")
            print(f"   COMPANY: {qa['company']} | PERIOD: {qa['period']}")
            print(f"   TYPE: {qa['qa_type']} | QUALITY: {qa['quality_score']:.0%}")
            print(f"   Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print()
    
    def save_results(self) -> None:
        """Save Q&A pairs to JSON and CSV"""
        output_path = self.config.OUTPUT_FOLDER
        
        # JSON
        json_file = output_path / "chromadb_qa_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved: {json_file}")
        
        # CSV
        import csv
        csv_file = output_path / "chromadb_qa_results.csv"
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'chunk_id', 'company', 'period', 'qa_type', 'question', 'answer', 'quality_score', 'source'
            ])
            writer.writeheader()
            for qa in self.qa_pairs:
                writer.writerow({
                    'chunk_id': qa['chunk_id'],
                    'company': qa['company'],
                    'period': qa['period'],
                    'qa_type': qa['qa_type'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'quality_score': f"{qa['quality_score']:.2%}",
                    'source': qa['source'],
                })
        logger.info(f"✅ Saved: {csv_file}")
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def generate(self, mode: str = 'all', query_text: str = None) -> List[Dict]:
        """
        Main generation pipeline
        
        Args:
            mode: 'all' (all collections), 'semantic' (search), 'filtered' (with filters)
            query_text: Search query (for semantic mode)
        
        Returns:
            List of Q&A pairs
        """
        logger.info("="*80)
        logger.info("CHROMADB Q&A GENERATOR V2 - ENHANCED")
        logger.info("="*80)
        
        chunks = []
        
        if mode == 'all':
            logger.info("\n🔍 Mode: ALL COLLECTIONS")
            all_chunks = self.get_chunks_all_collections()
            for coll_name, coll_chunks in all_chunks.items():
                for chunk in coll_chunks:
                    chunk['collection'] = coll_name
                chunks.extend(coll_chunks)
        
        elif mode == 'semantic' and query_text:
            logger.info(f"\n🔍 Mode: SEMANTIC SEARCH")
            logger.info(f"   Query: {query_text}")
            chunks = self.query_collections_semantic(query_text)
        
        elif mode == 'filtered':
            logger.info(f"\n🔍 Mode: FILTERED")
            where = self.build_where_filter()
            if where:
                logger.info(f"   Filters: {where}")
            collections = self.get_all_collections()
            for coll_name in collections:
                coll_chunks = self.get_chunks_from_collection(coll_name, where=where)
                for chunk in coll_chunks:
                    chunk['collection'] = coll_name
                chunks.extend(coll_chunks)
        
        if not chunks:
            logger.warning("❌ No chunks retrieved")
            return []
        
        logger.info(f"\n📊 Processing {len(chunks)} chunks...")
        self.process_chunks_batch(chunks)
        
        if self.qa_pairs:
            self.format_qa_display()
            self.save_results()
        
        logger.info(f"\n✅ Generated {len(self.qa_pairs)} Q&A pairs")
        logger.info("="*80 + "\n")
        
        return self.qa_pairs


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    try:
        gen = EnhancedChromaDBGenerator()
        
        # Option 1: Query all collections
        qa_pairs = gen.generate(mode='all')
        
        # Option 2: Semantic search
        # qa_pairs = gen.generate(mode='semantic', query_text='revenue growth')
        
        # Option 3: Filtered query
        # qa_pairs = gen.generate(mode='filtered')
        
        if qa_pairs:
            logger.info(f"\n✅ Success! Generated {len(qa_pairs)} Q&A pairs")
            return 0
        else:
            logger.warning("\n⚠️  No Q&A pairs generated")
            return 1
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted")
        return 130
    
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
