"""
Ground Truth Q&A Generator V6 - Multi-Modal Production
Generates 200+ high-quality Q&A pairs from multiple data sources
Supports: CSV, PDF, JSON, TXT formats
Features: .env config, robust error handling, rate limiting, logging
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


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
    """Centralized configuration from environment variables"""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Paths
    DATA_FOLDER: Path = Path(os.getenv(
        "DATA_FOLDER",
        r"D:\\Multimodal-Market-Intelligence-System\\data"
    ))
    OUTPUT_FOLDER: Path = Path(os.getenv(
        "OUTPUT_FOLDER",
        r"D:\\Multimodal-Market-Intelligence-System\\ground_truth"
    ))
    
    # Generation Parameters (Multi-Modal)
    TARGET_QUESTIONS: int = int(os.getenv("TARGET_QUESTIONS", "200"))
    TARGET_CSV_QUESTIONS: int = int(os.getenv("TARGET_CSV_QUESTIONS", "60"))
    TARGET_PDF_QUESTIONS: int = int(os.getenv("TARGET_PDF_QUESTIONS", "60"))
    TARGET_JSON_QUESTIONS: int = int(os.getenv("TARGET_JSON_QUESTIONS", "30"))
    TARGET_TXT_QUESTIONS: int = int(os.getenv("TARGET_TXT_QUESTIONS", "20"))
    GEMINI_QUESTIONS_PER_SOURCE: int = int(os.getenv("GEMINI_QUESTIONS_PER_SOURCE", "8"))
    
    # Multi-Modal Extraction Flags
    EXTRACT_FROM_CSV: bool = os.getenv("EXTRACT_FROM_CSV", "true").lower() == "true"
    EXTRACT_FROM_PDF: bool = os.getenv("EXTRACT_FROM_PDF", "true").lower() == "true"
    EXTRACT_FROM_JSON: bool = os.getenv("EXTRACT_FROM_JSON", "true").lower() == "true"
    EXTRACT_FROM_TXT: bool = os.getenv("EXTRACT_FROM_TXT", "true").lower() == "true"
    
    # Rate Limiting
    API_DELAY_SECONDS: int = int(os.getenv("API_DELAY_SECONDS", "10"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_SECONDS: int = int(os.getenv("RETRY_DELAY_SECONDS", "30"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            logger.error("❌ GEMINI_API_KEY not found in .env file")
            return False
        
        cls.DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        
        return True


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class QuotaExceededException(Exception):
    """Raised when API quota is exceeded"""
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after or Config.RETRY_DELAY_SECONDS
        super().__init__(f"API quota exceeded. Retry after {self.retry_after}s")


class APIError(Exception):
    """Generic API error"""
    pass


# ============================================================================
# MAIN GENERATOR CLASS - MULTI-MODAL
# ============================================================================

class MultiModalGroundTruthGenerator:
    """
    V6: Multi-modal Q&A generator supporting CSV, PDF, JSON, TXT
    """
    
    COMPANY_NAMES = {
        "AMZN": "Amazon",
        "MSFT": "Microsoft",
        "GOOGL": "Google (Alphabet)",
        "Amazon": "Amazon",
        "Microsoft": "Microsoft",
        "Google": "Google (Alphabet)"
    }
    
    # CSV Metric Templates
    METRIC_TEMPLATES = {
        'EPS': "What was {company}'s earnings per share in {quarter}?",
        'PE_Ratio': "What was {company}'s P/E ratio in {quarter}?",
        'ROE': "What was {company}'s return on equity in {quarter}?",
        'ROA': "What was {company}'s return on assets in {quarter}?",
        'Revenue': "What was {company}'s total revenue in {quarter}?",
        'Profit': "What was {company}'s net profit in {quarter}?",
        'MarketCap': "What was {company}'s market capitalization in {quarter}?",
        'PB_Ratio': "What was {company}'s price-to-book ratio in {quarter}?",
        'Beta': "What was {company}'s stock beta in {quarter}?",
        'EBITDA': "What was {company}'s EBITDA in {quarter}?",
        'DebtToEquity': "What was {company}'s debt-to-equity ratio in {quarter}?",
    }
    
    # PDF Extraction Patterns
    PDF_PATTERNS = {
        'net_sales': r'Net sales.*?(?:increased|decreased).*?to\s+\$?([\d.]+)\s*billion',
        'operating_income': r'Operating income.*?(?:increased|decreased).*?to\s+\$?([\d.]+)\s*billion',
        'net_income': r'Net income.*?(?:increased|decreased).*?to\s+\$?([\d.]+)\s*billion',
        'eps': r'(\$?[\d.]+)\s*per diluted share',
        'aws_sales': r'AWS segment sales.*?(?:increased|decreased).*?to\s+\$?([\d.]+)\s*billion',
        'north_america_sales': r'North America segment sales.*?(?:increased|decreased).*?to\s+\$?([\d.]+)\s*billion',
        'international_sales': r'International segment sales.*?(?:increased|decreased).*?to\s+\$?([\d.]+)\s*billion',
        'aws_operating_income': r'AWS segment operating income.*?(?:was).*?\$?([\d.]+)\s*billion',
        'operating_cash_flow': r'Operating cash flow.*?(?:increased|decreased).*?to\s+\$?([\d.]+)\s*billion',
        'free_cash_flow': r'Free cash flow.*?(?:improved|increased).*?to.*?inflow of\s+\$?([\d.]+)\s*billion',
    }
    
    def __init__(self):
        """Initialize multi-modal generator"""
        self.config = Config
        self.qa_pairs: List[Dict] = []
        self.questions_set: Set[str] = set()
        self.use_gemini = False
        
        if not self.config.validate():
            raise ValueError("Invalid configuration. Check .env file.")
        
        self._init_gemini()
        
        # Track extraction stats
        self.stats = {
            'csv': 0,
            'pdf': 0,
            'json': 0,
            'txt': 0,
            'gemini': 0
        }
    
    def _init_gemini(self) -> None:
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
            self.use_gemini = True
            logger.info(f"✅ Gemini API configured ({self.config.GEMINI_MODEL})")
        except Exception as e:
            logger.error(f"⚠️  Failed to configure Gemini API: {e}")
            logger.warning("   Will use local extraction only")
            self.use_gemini = False
    
    def add_qa(
        self,
        question: str,
        answer: str,
        company: str,
        source: str,
        confidence: float = 1.0,
        qa_type: str = "factual"
    ) -> bool:
        """Add Q&A pair with duplicate checking"""
        q_normalized = question.lower().strip()
        
        if q_normalized in self.questions_set:
            return False
        
        self.questions_set.add(q_normalized)
        self.qa_pairs.append({
            "question": question,
            "answer": answer,
            "company": company,
            "source": source,
            "type": qa_type,
            "confidence": confidence,
            "created_date": datetime.now().isoformat()
        })
        
        return True
    
    # ========================================================================
    # CSV EXTRACTION
    # ========================================================================
    
    def extract_from_csv(self) -> int:
        """Extract Q&A from CSV files"""
        if not self.config.EXTRACT_FROM_CSV:
            logger.info("⏭️  CSV extraction disabled")
            return 0
        
        count_before = len(self.qa_pairs)
        logger.info("📊 Extracting from CSV files...")
        
        csv_files = list(self.config.DATA_FOLDER.rglob("*.csv"))
        logger.info(f"   Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            if "daily_stocks" not in csv_file.name.lower():
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                for ticker in df['Ticker'].unique():
                    company = self.COMPANY_NAMES.get(ticker, ticker)
                    ticker_data = df[df['Ticker'] == ticker]
                    
                    for quarter in ticker_data['Quarter'].unique():
                        quarter_data = ticker_data[ticker_data['Quarter'] == quarter]
                        
                        if quarter_data.empty:
                            continue
                        
                        latest = quarter_data.iloc[0]
                        
                        # Generate Q&A for each metric
                        for metric_key, template in self.METRIC_TEMPLATES.items():
                            if metric_key in latest.index and pd.notna(latest[metric_key]):
                                value = latest[metric_key]
                                answer = self._format_answer(metric_key, value)
                                question = template.format(company=company, quarter=quarter)
                                
                                self.add_qa(
                                    question=question,
                                    answer=answer,
                                    company=company,
                                    source=csv_file.name,
                                    confidence=1.0,
                                    qa_type="factual"
                                )
                                
                                if len(self.qa_pairs) >= self.config.TARGET_CSV_QUESTIONS:
                                    break
                        
                        if len(self.qa_pairs) >= self.config.TARGET_CSV_QUESTIONS:
                            break
                    
                    if len(self.qa_pairs) >= self.config.TARGET_CSV_QUESTIONS:
                        break
                
            except Exception as e:
                logger.error(f"⚠️  Error reading {csv_file.name}: {e}")
                continue
        
        generated = len(self.qa_pairs) - count_before
        self.stats['csv'] = generated
        logger.info(f"   ✅ Generated {generated} Q&A from CSV")
        return generated
    
    def _format_answer(self, metric_key: str, value: float) -> str:
        """Format metric value as human-readable answer"""
        if metric_key == 'EPS':
            return f"${value:.2f}"
        elif metric_key in ['PE_Ratio', 'PB_Ratio', 'Beta', 'DebtToEquity']:
            return f"{value:.2f}"
        elif metric_key in ['ROE', 'ROA']:
            return f"{value * 100:.2f}%"
        elif metric_key in ['Revenue', 'Profit', 'MarketCap', 'EBITDA']:
            if value > 1e12:
                return f"${value/1e12:.2f} trillion"
            elif value > 1e9:
                return f"${value/1e9:.2f} billion"
            else:
                return f"${value/1e6:.2f} million"
        else:
            return str(value)
    
    # ========================================================================
    # PDF EXTRACTION
    # ========================================================================
    
    def extract_from_pdf(self) -> int:
        """Extract Q&A from PDF files (earnings releases)"""
        if not self.config.EXTRACT_FROM_PDF:
            logger.info("⏭️  PDF extraction disabled")
            return 0
        
        count_before = len(self.qa_pairs)
        logger.info("📄 Extracting from PDF files...")
        
        pdf_files = list(self.config.DATA_FOLDER.rglob("*.pdf"))
        logger.info(f"   Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            try:
                # Extract company and quarter from filename
                # e.g., "AMZN-Q1-2024-Earnings-Release.pdf"
                filename_parts = pdf_file.stem.split('-')
                company_ticker = filename_parts[0] if filename_parts else "Unknown"
                quarter = filename_parts[1] if len(filename_parts) > 1 else "Q1"
                year = filename_parts[2] if len(filename_parts) > 2 else "2024"
                
                company = self.COMPANY_NAMES.get(company_ticker, company_ticker)
                quarter_full = f"{quarter} {year}"
                
                # Extract text from first 3 pages (summary usually here)
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(min(3, len(reader.pages))):
                    text += reader.pages[page_num].extract_text()
                
                # Extract metrics using patterns
                for metric_name, pattern in self.PDF_PATTERNS.items():
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        value = matches[0]
                        
                        # Generate question
                        question = self._generate_pdf_question(company, quarter_full, metric_name)
                        answer = self._format_pdf_answer(metric_name, value, text, matches)
                        
                        self.add_qa(
                            question=question,
                            answer=answer,
                            company=company,
                            source=pdf_file.name,
                            confidence=0.95,
                            qa_type="factual"
                        )
                
                if len(self.qa_pairs) - count_before >= self.config.TARGET_PDF_QUESTIONS:
                    break
                
            except Exception as e:
                logger.error(f"⚠️  Error reading {pdf_file.name}: {e}")
                continue
        
        generated = len(self.qa_pairs) - count_before
        self.stats['pdf'] = generated
        logger.info(f"   ✅ Generated {generated} Q&A from PDFs")
        return generated
    
    def _generate_pdf_question(self, company: str, quarter: str, metric: str) -> str:
        """Generate question for PDF metric"""
        metric_questions = {
            'net_sales': f"What were {company}'s net sales in {quarter}?",
            'operating_income': f"What was {company}'s operating income in {quarter}?",
            'net_income': f"What was {company}'s net income in {quarter}?",
            'eps': f"What was {company}'s earnings per diluted share in {quarter}?",
            'aws_sales': f"What were {company}'s AWS segment sales in {quarter}?",
            'north_america_sales': f"What were {company}'s North America segment sales in {quarter}?",
            'international_sales': f"What were {company}'s International segment sales in {quarter}?",
            'aws_operating_income': f"What was {company}'s AWS segment operating income in {quarter}?",
            'operating_cash_flow': f"What was {company}'s operating cash flow in {quarter}?",
            'free_cash_flow': f"What was {company}'s free cash flow in {quarter}?",
        }
        return metric_questions.get(metric, f"What was {company}'s {metric} in {quarter}?")
    
    def _format_pdf_answer(self, metric: str, value: str, text: str, matches: list) -> str:
        """Format PDF answer with context"""
        if 'sales' in metric or 'income' in metric or 'flow' in metric:
            # Try to find growth percentage
            growth_pattern = rf"{re.escape(value)}.*?([\d.]+)%"
            growth_match = re.search(growth_pattern, text)
            if growth_match:
                growth = growth_match.group(1)
                return f"${value} billion, growing {growth}% year-over-year"
            return f"${value} billion"
        elif metric == 'eps':
            return f"${value}"
        else:
            return value
    
    # ========================================================================
    # JSON EXTRACTION
    # ========================================================================
    
    def extract_from_json(self) -> int:
        """Extract Q&A from JSON files"""
        if not self.config.EXTRACT_FROM_JSON:
            logger.info("⏭️  JSON extraction disabled")
            return 0
        
        count_before = len(self.qa_pairs)
        logger.info("📋 Extracting from JSON files...")
        
        json_files = list(self.config.DATA_FOLDER.rglob("*.json"))
        logger.info(f"   Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract company from filename
                filename_parts = json_file.stem.split('-')
                company_ticker = filename_parts[0] if filename_parts else "Unknown"
                company = self.COMPANY_NAMES.get(company_ticker, company_ticker)
                
                # Recursive extraction from JSON
                self._extract_from_json_recursive(
                    data, company, json_file.name, prefix=""
                )
                
                if len(self.qa_pairs) - count_before >= self.config.TARGET_JSON_QUESTIONS:
                    break
                
            except Exception as e:
                logger.error(f"⚠️  Error reading {json_file.name}: {e}")
                continue
        
        generated = len(self.qa_pairs) - count_before
        self.stats['json'] = generated
        logger.info(f"   ✅ Generated {generated} Q&A from JSON")
        return generated
    
    def _extract_from_json_recursive(
        self, 
        data: any, 
        company: str, 
        source: str, 
        prefix: str = ""
    ) -> None:
        """Recursively extract Q&A from nested JSON"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    # Generate Q&A for leaf values
                    question = f"According to {source}, what is {prefix}{key}?"
                    answer = str(value)
                    
                    self.add_qa(
                        question=question,
                        answer=answer,
                        company=company,
                        source=source,
                        confidence=0.90,
                        qa_type="factual"
                    )
                elif isinstance(value, (dict, list)):
                    # Recurse into nested structures
                    new_prefix = f"{prefix}{key} "
                    self._extract_from_json_recursive(value, company, source, new_prefix)
        
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    self._extract_from_json_recursive(item, company, source, prefix)
    
    # ========================================================================
    # TXT EXTRACTION
    # ========================================================================
    
    def extract_from_txt(self) -> int:
        """Extract Q&A from TXT files (transcripts)"""
        if not self.config.EXTRACT_FROM_TXT:
            logger.info("⏭️  TXT extraction disabled")
            return 0
        
        count_before = len(self.qa_pairs)
        logger.info("📝 Extracting from TXT files...")
        
        txt_files = list(self.config.DATA_FOLDER.rglob("*.txt"))
        logger.info(f"   Found {len(txt_files)} TXT files")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Extract company from filename
                filename_parts = txt_file.stem.split('-')
                company_ticker = filename_parts[0] if filename_parts else "Unknown"
                company = self.COMPANY_NAMES.get(company_ticker, company_ticker)
                
                # Extract key sentences with financial metrics
                sentences = self._extract_financial_sentences(text)
                
                for sentence in sentences[:10]:  # Limit to top 10 per file
                    question = f"According to the {txt_file.stem} transcript, what was mentioned about financial performance?"
                    answer = sentence
                    
                    self.add_qa(
                        question=question,
                        answer=answer,
                        company=company,
                        source=txt_file.name,
                        confidence=0.85,
                        qa_type="analytical"
                    )
                
                if len(self.qa_pairs) - count_before >= self.config.TARGET_TXT_QUESTIONS:
                    break
                
            except Exception as e:
                logger.error(f"⚠️  Error reading {txt_file.name}: {e}")
                continue
        
        generated = len(self.qa_pairs) - count_before
        self.stats['txt'] = generated
        logger.info(f"   ✅ Generated {generated} Q&A from TXT")
        return generated
    
    def _extract_financial_sentences(self, text: str) -> List[str]:
        """Extract sentences containing financial keywords"""
        financial_keywords = [
            'revenue', 'income', 'profit', 'growth', 'billion', 'million',
            'margin', 'sales', 'cash flow', 'operating', 'earnings'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        financial_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in financial_keywords):
                if len(sentence) > 50 and len(sentence) < 300:
                    financial_sentences.append(sentence)
        
        return financial_sentences[:20]  # Top 20
    
    # ========================================================================
    # GEMINI-POWERED ANALYTICAL Q&A
    # ========================================================================
    
    def generate_gemini_qa(self, all_sources_summary: str) -> int:
        """Generate analytical Q&A using Gemini across all sources"""
        if not self.use_gemini:
            logger.warning("⚠️  Gemini API not configured - skipping AI generation")
            return 0
        
        count_before = len(self.qa_pairs)
        logger.info("🤖 Generating Gemini-powered analytical Q&A...")
        logger.info(f"   Rate limit: {self.config.API_DELAY_SECONDS}s delay between requests")
        
        # Sample questions based on already extracted data
        sample_data = self.qa_pairs[:50]  # Use first 50 as context
        
        prompt = self._build_gemini_multi_source_prompt(sample_data)
        
        try:
            response_text = self._call_gemini_with_retry(prompt)
            json_text = self._extract_json_from_response(response_text)
            qa_list = json.loads(json_text)
            
            for item in qa_list:
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()
                qa_type = item.get('type', 'analytical')
                company = item.get('company', 'Multiple')
                
                if question and answer:
                    self.add_qa(
                        question=question,
                        answer=answer,
                        company=company,
                        source="gemini-multi-source",
                        confidence=0.88,
                        qa_type=qa_type
                    )
            
            logger.info(f"   ✅ Generated {len(qa_list)} Gemini Q&A")
            time.sleep(self.config.API_DELAY_SECONDS)
            
        except Exception as e:
            logger.error(f"   ⚠️  Gemini generation failed: {e}")
        
        generated = len(self.qa_pairs) - count_before
        self.stats['gemini'] = generated
        return generated
    
    def _build_gemini_multi_source_prompt(self, sample_data: List[Dict]) -> str:
        """Build prompt for Gemini using multi-source context"""
        # Summarize available data
        sources = set([qa['source'] for qa in sample_data])
        companies = set([qa['company'] for qa in sample_data])
        
        context_summary = "\n".join([
            f"- {qa['question']}: {qa['answer']}"
            for qa in sample_data[:20]
        ])
        
        return f"""You are a financial analyst creating complex ground truth Q&A pairs for evaluating AI systems.

Given the following financial data extracted from multiple sources:

Sources available: {', '.join(sources)}
Companies: {', '.join(companies)}

Sample data:
{context_summary}

Generate {self.config.GEMINI_QUESTIONS_PER_SOURCE} diverse, high-quality analytical question-answer pairs that:
- Require synthesis across multiple data points
- Ask about trends, comparisons, and insights
- Use proper financial terminology
- Are answerable from the provided context

Requirements:
- Mix question types: comparative, trend analysis, interpretation
- Answers should be 1-3 sentences
- Each question should be unique and non-repetitive
- Include company name in response

Return as valid JSON array:
[
  {{"question": "...", "answer": "...", "type": "analytical", "company": "Amazon"}},
  {{"question": "...", "answer": "...", "type": "comparative", "company": "Multiple"}},
  ...
]
"""
    
    def _call_gemini_with_retry(self, prompt: str) -> str:
        """Call Gemini API with retry logic"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except ResourceExhausted:
                logger.warning(f"⚠️  Quota exceeded (attempt {attempt + 1}/{self.config.MAX_RETRIES})")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY_SECONDS)
                else:
                    raise QuotaExceededException()
            except Exception as e:
                logger.error(f"⚠️  API error: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(5)
                else:
                    raise APIError(f"Failed after {self.config.MAX_RETRIES} attempts")
        
        raise APIError("Unexpected error in retry logic")
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from markdown code blocks"""
        if "```json" in response_text:
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                return match.group(1)
        if "```" in response_text:
            match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                return match.group(1)
        return response_text
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def generate(self) -> List[Dict]:
        """Main multi-modal generation pipeline"""
        logger.info("="*80)
        logger.info("MULTI-MODAL GROUND TRUTH Q&A GENERATOR V6")
        logger.info("="*80)
        logger.info(f"🎯 Target: {self.config.TARGET_QUESTIONS} Q&A pairs")
        logger.info(f"📁 Data: {self.config.DATA_FOLDER}")
        logger.info(f"🤖 Gemini: {'Enabled' if self.use_gemini else 'Disabled'}")
        logger.info(f"📊 Multi-Modal: CSV={self.config.EXTRACT_FROM_CSV}, "
                   f"PDF={self.config.EXTRACT_FROM_PDF}, "
                   f"JSON={self.config.EXTRACT_FROM_JSON}, "
                   f"TXT={self.config.EXTRACT_FROM_TXT}")
        
        # Phase 1: CSV
        if len(self.qa_pairs) < self.config.TARGET_QUESTIONS:
            self.extract_from_csv()
        
        # Phase 2: PDF
        if len(self.qa_pairs) < self.config.TARGET_QUESTIONS:
            self.extract_from_pdf()
        
        # Phase 3: JSON
        if len(self.qa_pairs) < self.config.TARGET_QUESTIONS:
            self.extract_from_json()
        
        # Phase 4: TXT
        if len(self.qa_pairs) < self.config.TARGET_QUESTIONS:
            self.extract_from_txt()
        
        # Phase 5: Gemini analytical
        if self.use_gemini and len(self.qa_pairs) < self.config.TARGET_QUESTIONS:
            self.generate_gemini_qa("")
        
        # Trim to target
        self.qa_pairs = self.qa_pairs[:self.config.TARGET_QUESTIONS]
        
        logger.info(f"\n📊 Generation Summary:")
        logger.info(f"   CSV: {self.stats['csv']} Q&A")
        logger.info(f"   PDF: {self.stats['pdf']} Q&A")
        logger.info(f"   JSON: {self.stats['json']} Q&A")
        logger.info(f"   TXT: {self.stats['txt']} Q&A")
        logger.info(f"   Gemini: {self.stats['gemini']} Q&A")
        logger.info(f"   Total: {len(self.qa_pairs)} Q&A pairs")
        
        return self.qa_pairs
    
    def save(self) -> Dict:
        """Save Q&A pairs"""
        output_path = self.config.OUTPUT_FOLDER
        
        # Save JSON
        json_file = output_path / "ground_truth_qa_pairs.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        
        # Save CSV
        csv_file = output_path / "ground_truth_qa_pairs.csv"
        df = pd.DataFrame(self.qa_pairs)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info("="*80)
        logger.info("✅ SAVED MULTI-MODAL GROUND TRUTH DATASET")
        logger.info("="*80)
        logger.info(f"📄 JSON: {json_file}")
        logger.info(f"📊 CSV: {csv_file}")
        logger.info(f"📈 Total: {len(self.qa_pairs)} Q&A pairs")
        
        if self.qa_pairs:
            df_qa = pd.DataFrame(self.qa_pairs)
            logger.info("\n📋 Summary:")
            logger.info(f"  Companies: {', '.join(df_qa['company'].unique())}")
            logger.info(f"  Types: {dict(df_qa['type'].value_counts())}")
            logger.info(f"  Sources: {dict(df_qa['source'].value_counts())}")
            logger.info(f"  Avg Confidence: {df_qa['confidence'].mean():.2%}")
        
        logger.info("="*80)
        
        return {
            "total_pairs": len(self.qa_pairs),
            "json_file": str(json_file),
            "csv_file": str(csv_file),
            "stats": self.stats
        }
    
    def print_samples(self, n: int = 10) -> None:
        """Print sample Q&A pairs by source"""
        if not self.qa_pairs:
            return
        
        logger.info(f"\n📋 Sample Q&A Pairs (first {n}):\n")
        for i, qa in enumerate(self.qa_pairs[:n], 1):
            print(f"{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print(f"   Source: {qa['source']} | Type: {qa['type']} | Confidence: {qa['confidence']:.0%}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    try:
        gen = MultiModalGroundTruthGenerator()
        qa_pairs = gen.generate()
        
        if not qa_pairs:
            logger.error("❌ No Q&A pairs generated")
            return 1
        
        gen.print_samples(n=15)
        gen.save()
        
        logger.info("\n✅ Multi-modal generation completed successfully!")
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Generation interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
