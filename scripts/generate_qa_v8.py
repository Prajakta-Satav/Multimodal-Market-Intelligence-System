"""
Ground Truth Q&A Generator V7 - Enhanced Extraction
Fixes: JSON files and 10-K/10-Q PDF extraction
Handles multiple PDF formats and nested JSON structures
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
    """Centralized configuration"""
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    DATA_FOLDER: Path = Path(os.getenv(
        "DATA_FOLDER",
        r"D:\Multimodal-Market-Intelligence-System\data"
    ))
    OUTPUT_FOLDER: Path = Path(os.getenv(
        "OUTPUT_FOLDER",
        r"D:\Multimodal-Market-Intelligence-System\ground_truth"
    ))
    
    QUESTIONS_PER_CSV: int = int(os.getenv("QUESTIONS_PER_CSV", "15"))
    QUESTIONS_PER_PDF: int = int(os.getenv("QUESTIONS_PER_PDF", "12"))
    QUESTIONS_PER_JSON: int = int(os.getenv("QUESTIONS_PER_JSON", "10"))
    QUESTIONS_PER_TXT: int = int(os.getenv("QUESTIONS_PER_TXT", "8"))
    
    EXTRACT_FROM_CSV: bool = os.getenv("EXTRACT_FROM_CSV", "true").lower() == "true"
    EXTRACT_FROM_PDF: bool = os.getenv("EXTRACT_FROM_PDF", "true").lower() == "true"
    EXTRACT_FROM_JSON: bool = os.getenv("EXTRACT_FROM_JSON", "true").lower() == "true"
    EXTRACT_FROM_TXT: bool = os.getenv("EXTRACT_FROM_TXT", "true").lower() == "true"
    
    API_DELAY_SECONDS: int = int(os.getenv("API_DELAY_SECONDS", "10"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_SECONDS: int = int(os.getenv("RETRY_DELAY_SECONDS", "30"))
    
    @classmethod
    def validate(cls) -> bool:
        if not cls.GEMINI_API_KEY:
            logger.error("❌ GEMINI_API_KEY not found in .env file")
            return False
        cls.DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        return True


# ============================================================================
# EXCEPTIONS
# ============================================================================

class QuotaExceededException(Exception):
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after or Config.RETRY_DELAY_SECONDS
        super().__init__(f"API quota exceeded. Retry after {self.retry_after}s")


class APIError(Exception):
    pass


# ============================================================================
# ENHANCED GENERATOR - V7
# ============================================================================

class EnhancedGroundTruthGenerator:
    """
    V7: Enhanced extraction for JSON and 10-K/10-Q PDFs
    """
    
    COMPANY_NAMES = {
        "AMZN": "Amazon", "MSFT": "Microsoft", "GOOGL": "Google (Alphabet)",
        "Amazon": "Amazon", "Microsoft": "Microsoft", "Google": "Google (Alphabet)",
        "Alphabet": "Google (Alphabet)", "GOOG": "Google (Alphabet)"
    }
    
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
    }
    
    # V7: EXPANDED PDF patterns for 10-K/10-Q
    PDF_PATTERNS = {
        # Earnings releases
        'net_sales': r'Net sales.*?(?:increased|decreased|was).*?\$?([\d,.]+)\s*(?:billion|million)',
        'operating_income': r'Operating income.*?(?:increased|decreased|was).*?\$?([\d,.]+)\s*(?:billion|million)',
        'net_income': r'Net income.*?(?:increased|decreased|was|totaled).*?\$?([\d,.]+)\s*(?:billion|million)',
        'eps': r'(\$?[\d.]+)\s*per diluted share',
        'aws_sales': r'AWS.*?sales.*?(?:increased|decreased|was).*?\$?([\d,.]+)\s*(?:billion|million)',
        'aws_revenue': r'AWS.*?revenue.*?(?:increased|decreased|was).*?\$?([\d,.]+)\s*(?:billion|million)',
        
        # 10-K/10-Q specific
        'total_revenue': r'Total.*?revenue|Total net revenues.*?\$?([\d,.]+)',
        'cost_of_revenue': r'Cost of revenue|Cost of net revenues.*?\$?([\d,.]+)',
        'operating_expenses': r'Operating expenses|Total operating expenses.*?\$?([\d,.]+)',
        'income_from_operations': r'Income from operations.*?\$?([\d,.]+)',
        'income_before_taxes': r'Income before (?:provision|income taxes).*?\$?([\d,.]+)',
        'net_income_10k': r'Net income.*?\$?([\d,.]+)',
        'eps_basic': r'Basic earnings per share.*?\$?([\d.]+)',
        'eps_diluted': r'Diluted earnings per share.*?\$?([\d.]+)',
        'operating_cash_flow': r'Operating cash flows.*?\$?([\d,.]+)',
        'investing_cash_flow': r'Investing cash flows.*?\$?([\d,.]+)',
        'financing_cash_flow': r'Financing cash flows.*?\$?([\d,.]+)',
    }
    
    def __init__(self):
        self.config = Config
        self.qa_pairs: List[Dict] = []
        self.questions_set: Set[str] = set()
        self.use_gemini = False
        
        if not self.config.validate():
            raise ValueError("Invalid configuration")
        
        self._init_gemini()
        self.file_stats = {}
    
    def _init_gemini(self) -> None:
        try:
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
            self.use_gemini = True
            logger.info(f"✅ Gemini API configured ({self.config.GEMINI_MODEL})")
        except Exception as e:
            logger.error(f"⚠️  Failed to configure Gemini: {e}")
            self.use_gemini = False
    
    def add_qa(self, question: str, answer: str, company: str, source: str,
               confidence: float = 1.0, qa_type: str = "factual") -> bool:
        """Add Q&A with duplicate checking"""
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
    # FILE DISCOVERY
    # ========================================================================
    
    def discover_all_files(self) -> Dict[str, List[Path]]:
        """Discover ALL files in data folder"""
        files = {'csv': [], 'pdf': [], 'json': [], 'txt': []}
        
        logger.info(f"🔍 Scanning data folder: {self.config.DATA_FOLDER}")
        
        for file_path in self.config.DATA_FOLDER.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext == '.csv':
                    files['csv'].append(file_path)
                elif ext == '.pdf':
                    files['pdf'].append(file_path)
                elif ext == '.json':
                    files['json'].append(file_path)
                elif ext == '.txt':
                    files['txt'].append(file_path)
        
        logger.info(f"   📊 Found: {len(files['csv'])} CSV, {len(files['pdf'])} PDF, "
                   f"{len(files['json'])} JSON, {len(files['txt'])} TXT")
        return files
    
    # ========================================================================
    # CSV EXTRACTION
    # ========================================================================
    
    def extract_from_single_csv(self, csv_path: Path) -> int:
        """Extract from ONE CSV file"""
        count_before = len(self.qa_pairs)
        
        try:
            df = pd.read_csv(csv_path)
            
            for ticker in df['Ticker'].unique():
                company = self.COMPANY_NAMES.get(ticker, ticker)
                ticker_data = df[df['Ticker'] == ticker]
                
                for quarter in ticker_data['Quarter'].unique():
                    quarter_data = ticker_data[ticker_data['Quarter'] == quarter]
                    if quarter_data.empty:
                        continue
                    
                    latest = quarter_data.iloc[0]
                    
                    for metric_key, template in self.METRIC_TEMPLATES.items():
                        if metric_key in latest.index and pd.notna(latest[metric_key]):
                            value = latest[metric_key]
                            answer = self._format_answer(metric_key, value)
                            question = template.format(company=company, quarter=quarter)
                            
                            self.add_qa(question, answer, company, csv_path.name, 1.0, "factual")
                    
                    if len(self.qa_pairs) - count_before >= self.config.QUESTIONS_PER_CSV:
                        break
        
        except Exception as e:
            logger.error(f"   ⚠️  Error in {csv_path.name}: {e}")
        
        generated = len(self.qa_pairs) - count_before
        self.file_stats[csv_path.name] = generated
        return generated
    
    def extract_from_all_csv(self, csv_files: List[Path]) -> int:
        """Extract from ALL CSV files"""
        if not self.config.EXTRACT_FROM_CSV or not csv_files:
            return 0
        
        logger.info(f"📊 Extracting from {len(csv_files)} CSV files...")
        total = 0
        
        for csv_path in csv_files:
            count = self.extract_from_single_csv(csv_path)
            if count > 0:
                logger.info(f"   ✅ {csv_path.name}: {count} Q&A")
            total += count
        
        logger.info(f"   📊 Total from CSV: {total} Q&A")
        return total
    
    def _format_answer(self, metric_key: str, value: float) -> str:
        """Format metric value"""
        if metric_key == 'EPS':
            return f"${value:.2f}"
        elif metric_key in ['PE_Ratio', 'PB_Ratio', 'Beta']:
            return f"{value:.2f}"
        elif metric_key in ['ROE', 'ROA']:
            return f"{value * 100:.2f}%"
        elif metric_key in ['Revenue', 'Profit', 'MarketCap']:
            if value > 1e12:
                return f"${value/1e12:.2f} trillion"
            elif value > 1e9:
                return f"${value/1e9:.2f} billion"
            else:
                return f"${value/1e6:.2f} million"
        return str(value)
    
    # ========================================================================
    # PDF EXTRACTION - ENHANCED FOR 10-K/10-Q
    # ========================================================================
    
    def extract_from_single_pdf(self, pdf_path: Path) -> int:
        """Extract from ONE PDF (handles multiple formats)"""
        count_before = len(self.qa_pairs)
        
        try:
            # Parse filename for metadata
            filename = pdf_path.stem
            filename_parts = filename.split('-') if '-' in filename else filename.split('_')
            
            # Detect company from filename
            company_ticker = None
            for part in filename_parts:
                if part.upper() in self.COMPANY_NAMES or part.upper() in ['AMZN', 'MSFT', 'GOOGL', 'GOOG']:
                    company_ticker = part.upper()
                    break
            
            if not company_ticker:
                # Try to detect from file content
                company_ticker = "UNKNOWN"
            
            company = self.COMPANY_NAMES.get(company_ticker, company_ticker)
            
            # Detect quarter/year
            quarter_year = self._extract_quarter_year(filename)
            
            # Extract text from PDF (all pages for 10-K/10-Q, limited for earnings)
            reader = PyPDF2.PdfReader(pdf_path)
            text = ""
            
            # For 10-K/10-Q: extract more pages
            # For earnings releases: extract fewer pages
            max_pages = 10 if ('10-k' in filename.lower() or '10-q' in filename.lower() or '10k' in filename.lower() or '10q' in filename.lower()) else 3
            
            for page_num in range(min(max_pages, len(reader.pages))):
                try:
                    text += reader.pages[page_num].extract_text()
                except Exception:
                    continue
            
            # Extract using patterns
            extracted_count = 0
            for metric_name, pattern in self.PDF_PATTERNS.items():
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    for match in matches[:1]:  # Take first match
                        value = match if isinstance(match, str) else str(match)
                        
                        # Clean value
                        value = value.replace(',', '').strip()
                        if not value:
                            continue
                        
                        question = self._generate_pdf_question(company, quarter_year, metric_name)
                        answer = self._format_pdf_answer(metric_name, value)
                        
                        if self.add_qa(question, answer, company, pdf_path.name, 0.90, "factual"):
                            extracted_count += 1
                        
                        if extracted_count >= self.config.QUESTIONS_PER_PDF:
                            break
                
                if extracted_count >= self.config.QUESTIONS_PER_PDF:
                    break
        
        except Exception as e:
            logger.debug(f"   ⚠️  Error in {pdf_path.name}: {e}")
        
        generated = len(self.qa_pairs) - count_before
        self.file_stats[pdf_path.name] = generated
        return generated
    
    def extract_from_all_pdf(self, pdf_files: List[Path]) -> int:
        """Extract from ALL PDF files"""
        if not self.config.EXTRACT_FROM_PDF or not pdf_files:
            return 0
        
        logger.info(f"📄 Extracting from {len(pdf_files)} PDF files...")
        total = 0
        
        for pdf_path in pdf_files:
            count = self.extract_from_single_pdf(pdf_path)
            if count > 0:
                logger.info(f"   ✅ {pdf_path.name}: {count} Q&A")
            total += count
        
        logger.info(f"   📄 Total from PDF: {total} Q&A")
        return total
    
    def _extract_quarter_year(self, filename: str) -> str:
        """Extract quarter and year from filename"""
        # Look for Q1, Q2, Q3, Q4
        for i in range(1, 5):
            if f'Q{i}' in filename or f'q{i}' in filename:
                quarter = f'Q{i}'
                break
        else:
            quarter = "Q1"
        
        # Look for year
        year_match = re.search(r'20\d{2}', filename)
        year = year_match.group(0) if year_match else "2024"
        
        return f"{quarter} {year}"
    
    def _generate_pdf_question(self, company: str, quarter: str, metric: str) -> str:
        """Generate question for PDF metric"""
        metric_questions = {
            'net_sales': f"What were {company}'s net sales in {quarter}?",
            'operating_income': f"What was {company}'s operating income in {quarter}?",
            'net_income': f"What was {company}'s net income in {quarter}?",
            'eps': f"What was {company}'s earnings per diluted share in {quarter}?",
            'aws_sales': f"What were {company}'s AWS segment sales in {quarter}?",
            'aws_revenue': f"What was {company}'s AWS revenue in {quarter}?",
            'total_revenue': f"What was {company}'s total revenue in {quarter}?",
            'cost_of_revenue': f"What was {company}'s cost of revenue in {quarter}?",
            'operating_expenses': f"What were {company}'s operating expenses in {quarter}?",
            'income_from_operations': f"What was {company}'s income from operations in {quarter}?",
            'income_before_taxes': f"What was {company}'s income before taxes in {quarter}?",
            'net_income_10k': f"What was {company}'s net income in {quarter}?",
            'eps_basic': f"What was {company}'s basic earnings per share in {quarter}?",
            'eps_diluted': f"What was {company}'s diluted earnings per share in {quarter}?",
            'operating_cash_flow': f"What was {company}'s operating cash flow in {quarter}?",
            'investing_cash_flow': f"What was {company}'s investing cash flow in {quarter}?",
            'financing_cash_flow': f"What was {company}'s financing cash flow in {quarter}?",
        }
        return metric_questions.get(metric, f"What was {company}'s {metric} in {quarter}?")
    
    def _format_pdf_answer(self, metric: str, value: str) -> str:
        """Format PDF answer"""
        try:
            # Try to convert to float and format
            val_float = float(value)
            if val_float > 1e9:
                return f"${val_float/1e9:.2f} billion"
            elif val_float > 1e6:
                return f"${val_float/1e6:.2f} million"
            elif 'eps' in metric.lower() or 'share' in metric.lower():
                return f"${val_float:.2f}"
            else:
                return f"{val_float:.2f}"
        except ValueError:
            return f"${value}"
    
    # ========================================================================
    # JSON EXTRACTION - ENHANCED
    # ========================================================================
    
    def extract_from_single_json(self, json_path: Path) -> int:
        """Extract from ONE JSON file - ENHANCED"""
        count_before = len(self.qa_pairs)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract company from filename
            filename = json_path.stem
            company_ticker = None
            for ticker in ['AMZN', 'MSFT', 'GOOGL', 'GOOG', 'Amazon', 'Microsoft', 'Google', 'Alphabet']:
                if ticker.lower() in filename.lower():
                    company_ticker = ticker
                    break
            
            company = self.COMPANY_NAMES.get(company_ticker, company_ticker or "Unknown")
            quarter_year = self._extract_quarter_year(filename)
            
            # V7: Enhanced JSON extraction
            self._extract_json_with_context(
                data, company, json_path.name, quarter_year,
                max_questions=self.config.QUESTIONS_PER_JSON
            )
        
        except Exception as e:
            logger.debug(f"   ⚠️  Error in {json_path.name}: {e}")
        
        generated = len(self.qa_pairs) - count_before
        self.file_stats[json_path.name] = generated
        return generated
    
    def extract_from_all_json(self, json_files: List[Path]) -> int:
        """Extract from ALL JSON files"""
        if not self.config.EXTRACT_FROM_JSON or not json_files:
            return 0
        
        logger.info(f"📋 Extracting from {len(json_files)} JSON files...")
        total = 0
        
        for json_path in json_files:
            count = self.extract_from_single_json(json_path)
            if count > 0:
                logger.info(f"   ✅ {json_path.name}: {count} Q&A")
            total += count
        
        logger.info(f"   📋 Total from JSON: {total} Q&A")
        return total
    
    def _extract_json_with_context(self, data: any, company: str, source: str,
                                   quarter_year: str, max_questions: int = 10) -> None:
        """V7: Extract JSON with financial context"""
        extracted = 0
        
        def extract_metrics(obj, prefix=""):
            nonlocal extracted
            if extracted >= max_questions:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if extracted >= max_questions:
                        return
                    
                    key_lower = key.lower()
                    
                    # Check if this is a financial metric
                    financial_keywords = [
                        'revenue', 'sales', 'income', 'profit', 'earnings',
                        'ebitda', 'cashflow', 'assets', 'liabilities',
                        'equity', 'shares', 'eps', 'margin'
                    ]
                    
                    is_metric = any(kw in key_lower for kw in financial_keywords)
                    
                    if is_metric and isinstance(value, (int, float, str)):
                        question = f"According to {source}, what was {company}'s {key} in {quarter_year}?"
                        answer = str(value)
                        
                        if self.add_qa(question, answer, company, source, 0.85, "factual"):
                            extracted += 1
                    
                    elif isinstance(value, (dict, list)):
                        extract_metrics(value, f"{prefix}{key} ")
            
            elif isinstance(obj, list):
                for item in obj:
                    if extracted >= max_questions:
                        return
                    if isinstance(item, (dict, list)):
                        extract_metrics(item, prefix)
        
        extract_metrics(data)
    
    # ========================================================================
    # TXT EXTRACTION
    # ========================================================================
    
    def extract_from_single_txt(self, txt_path: Path) -> int:
        """Extract from ONE TXT file"""
        count_before = len(self.qa_pairs)
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            filename = txt_path.stem
            company_ticker = None
            for ticker in ['AMZN', 'MSFT', 'GOOGL', 'GOOG', 'Amazon', 'Microsoft', 'Google', 'Alphabet']:
                if ticker.lower() in filename.lower():
                    company_ticker = ticker
                    break
            
            company = self.COMPANY_NAMES.get(company_ticker, company_ticker or "Unknown")
            
            # Extract financial sentences
            sentences = self._extract_financial_sentences(text)
            
            for sentence in sentences[:self.config.QUESTIONS_PER_TXT]:
                question = f"According to {txt_path.stem}, what was mentioned about financial performance?"
                answer = sentence
                
                self.add_qa(question, answer, company, txt_path.name, 0.85, "analytical")
        
        except Exception as e:
            logger.debug(f"   ⚠️  Error in {txt_path.name}: {e}")
        
        generated = len(self.qa_pairs) - count_before
        self.file_stats[txt_path.name] = generated
        return generated
    
    def extract_from_all_txt(self, txt_files: List[Path]) -> int:
        """Extract from ALL TXT files"""
        if not self.config.EXTRACT_FROM_TXT or not txt_files:
            return 0
        
        logger.info(f"📝 Extracting from {len(txt_files)} TXT files...")
        total = 0
        
        for txt_path in txt_files:
            count = self.extract_from_single_txt(txt_path)
            if count > 0:
                logger.info(f"   ✅ {txt_path.name}: {count} Q&A")
            total += count
        
        logger.info(f"   📝 Total from TXT: {total} Q&A")
        return total
    
    def _extract_financial_sentences(self, text: str) -> List[str]:
        """Extract sentences with financial keywords"""
        keywords = [
            'revenue', 'income', 'profit', 'growth', 'billion', 'million',
            'margin', 'sales', 'cash flow', 'operating', 'earnings'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        financial = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(kw in sentence.lower() for kw in keywords):
                if 50 < len(sentence) < 300:
                    financial.append(sentence)
        
        return financial
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def generate(self) -> List[Dict]:
        """Main generation pipeline"""
        logger.info("="*80)
        logger.info("GROUND TRUTH GENERATOR V7 - ENHANCED EXTRACTION")
        logger.info("="*80)
        logger.info(f"📁 Data folder: {self.config.DATA_FOLDER}")
        
        files = self.discover_all_files()
        
        self.extract_from_all_csv(files['csv'])
        self.extract_from_all_pdf(files['pdf'])
        self.extract_from_all_json(files['json'])
        self.extract_from_all_txt(files['txt'])
        
        logger.info(f"\n📊 Total generated: {len(self.qa_pairs)} Q&A pairs")
        logger.info(f"📁 Files processed: {len(self.file_stats)}")
        
        return self.qa_pairs
    
    def save(self) -> Dict:
        """Save Q&A pairs"""
        output_path = self.config.OUTPUT_FOLDER
        
        json_file = output_path / "ground_truth_qa_pairs.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        
        csv_file = output_path / "ground_truth_qa_pairs.csv"
        df = pd.DataFrame(self.qa_pairs)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info("="*80)
        logger.info("✅ SAVED ENHANCED GROUND TRUTH DATASET")
        logger.info("="*80)
        logger.info(f"📄 JSON: {json_file}")
        logger.info(f"📊 CSV: {csv_file}")
        logger.info(f"📈 Total: {len(self.qa_pairs)} Q&A pairs")
        
        if self.qa_pairs:
            df_qa = pd.DataFrame(self.qa_pairs)
            logger.info("\n📋 Summary:")
            logger.info(f"  Companies: {', '.join(df_qa['company'].unique())}")
            logger.info(f"  Types: {dict(df_qa['type'].value_counts())}")
            logger.info(f"  Avg Confidence: {df_qa['confidence'].mean():.2%}")
            
            logger.info("\n📁 Per-File Breakdown (Top 20):")
            for filename, count in sorted(self.file_stats.items(), key=lambda x: -x[1])[:20]:
                if count > 0:
                    logger.info(f"  {filename}: {count} Q&A")
        
        logger.info("="*80)
        
        return {
            "total_pairs": len(self.qa_pairs),
            "json_file": str(json_file),
            "csv_file": str(csv_file),
            "file_stats": self.file_stats
        }
    
    def print_samples(self, n: int = 15) -> None:
        """Print sample Q&A pairs"""
        if not self.qa_pairs:
            return
        
        logger.info(f"\n📋 Sample Q&A Pairs (first {n}):\n")
        for i, qa in enumerate(self.qa_pairs[:n], 1):
            print(f"{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print(f"   Source: {qa['source']} | Type: {qa['type']} | Confidence: {qa['confidence']:.0%}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        gen = EnhancedGroundTruthGenerator()
        qa_pairs = gen.generate()
        
        if not qa_pairs:
            logger.error("❌ No Q&A pairs generated")
            return 1
        
        gen.print_samples(n=15)
        gen.save()
        
        logger.info("\n✅ V7 enhanced extraction finished!")
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
