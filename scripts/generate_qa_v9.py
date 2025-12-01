"""
Ground Truth Q&A Generator V8.1 - CSV + JSON + TXT
Extracts high-quality Q&A from:
  ✅ CSV: Structured financial metrics
  ✅ JSON: Parsed financial data
  ✅ TXT: Management insights, earnings calls, transcripts
  ❌ PDF: Skipped (inconsistent extraction quality)

Smart TXT extraction: Identifies and extracts key financial statements
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
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
    
    # Questions per file
    QUESTIONS_PER_CSV: int = int(os.getenv("QUESTIONS_PER_CSV", "20"))
    QUESTIONS_PER_JSON: int = int(os.getenv("QUESTIONS_PER_JSON", "15"))
    QUESTIONS_PER_TXT: int = int(os.getenv("QUESTIONS_PER_TXT", "12"))
    
    # V8.1: CSV, JSON, and TXT
    EXTRACT_FROM_CSV: bool = True
    EXTRACT_FROM_JSON: bool = True
    EXTRACT_FROM_TXT: bool = True
    EXTRACT_FROM_PDF: bool = False
    
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
# MAIN GENERATOR - V8.1 WITH TXT
# ============================================================================

class EnhancedFocusedGenerator:
    """
    V8.1: High-quality extraction from CSV, JSON, and TXT
    CSV: Metrics (100% confidence)
    JSON: Structured data (95% confidence)
    TXT: Management insights (85-90% confidence)
    """
    
    COMPANY_NAMES = {
        "AMZN": "Amazon", "MSFT": "Microsoft", "GOOGL": "Google (Alphabet)",
        "Amazon": "Amazon", "Microsoft": "Microsoft", "Google": "Google (Alphabet)",
        "Alphabet": "Google (Alphabet)", "GOOG": "Google (Alphabet)"
    }
    
    # CSV Metrics
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
        'CurrentRatio': "What was {company}'s current ratio in {quarter}?",
    }
    
    # TXT Financial Keywords & Patterns (V8.1)
    TXT_FINANCIAL_PATTERNS = {
        'revenue_growth': r'revenue[s]?.*?(?:grew|increased|declined|decreased).*?(?:by\s+)?(\d+(?:\.\d+)?%)',
        'earnings_per_share': r'earnings per share.*?\$?([\d.]+)',
        'net_income': r'net income.*?\$?([\d,.]+)\s*(?:billion|million)',
        'operating_income': r'operating income.*?\$?([\d,.]+)\s*(?:billion|million)',
        'operating_margin': r'operating margin.*?(\d+(?:\.\d+)?%)',
        'gross_margin': r'gross margin.*?(\d+(?:\.\d+)?%)',
        'profit_growth': r'(?:profit|earnings).*?(?:grew|increased|up).*?(?:by\s+)?(\d+(?:\.\d+)?%)',
        'segment_growth': r'(?:aws|cloud|services?).*?(?:grew|increased).*?(?:by\s+)?(\d+(?:\.\d+)?%)',
        'guidance': r'guidance.*?\$?([\d.]+)\s*(?:billion|million)',
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
        self.metrics_data = []
    
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
        """Discover CSV, JSON, and TXT files"""
        files = {'csv': [], 'json': [], 'txt': []}
        
        logger.info(f"🔍 Scanning data folder: {self.config.DATA_FOLDER}")
        logger.info(f"   Focus: CSV + JSON + TXT (structured + insights)")
        
        for file_path in self.config.DATA_FOLDER.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext == '.csv':
                    files['csv'].append(file_path)
                elif ext == '.json':
                    files['json'].append(file_path)
                elif ext == '.txt':
                    files['txt'].append(file_path)
        
        logger.info(f"   📊 Found: {len(files['csv'])} CSV, {len(files['json'])} JSON, {len(files['txt'])} TXT")
        logger.info(f"   ⏭️  Skipping: PDF (extraction quality issues)")
        
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
                    row_metrics = {
                        'company': company,
                        'ticker': ticker,
                        'quarter': quarter,
                        'data': {}
                    }
                    
                    for metric_key, template in self.METRIC_TEMPLATES.items():
                        if metric_key in latest.index and pd.notna(latest[metric_key]):
                            value = latest[metric_key]
                            answer = self._format_answer(metric_key, value)
                            question = template.format(company=company, quarter=quarter)
                            
                            self.add_qa(question, answer, company, csv_path.name, 1.0, "factual")
                            row_metrics['data'][metric_key] = value
                            
                            if len(self.qa_pairs) - count_before >= self.config.QUESTIONS_PER_CSV:
                                break
                        
                        if len(self.qa_pairs) - count_before >= self.config.QUESTIONS_PER_CSV:
                            break
                    
                    self.metrics_data.append(row_metrics)
                    
                    if len(self.qa_pairs) - count_before >= self.config.QUESTIONS_PER_CSV:
                        break
                
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
        elif metric_key in ['PE_Ratio', 'PB_Ratio', 'Beta', 'DebtToEquity', 'CurrentRatio']:
            return f"{value:.2f}"
        elif metric_key in ['ROE', 'ROA']:
            return f"{value * 100:.2f}%"
        elif metric_key in ['Revenue', 'Profit', 'MarketCap', 'EBITDA']:
            if value > 1e12:
                return f"${value/1e12:.2f} trillion"
            elif value > 1e9:
                return f"${value/1e9:.2f} billion"
            elif value > 1e6:
                return f"${value/1e6:.2f} million"
            else:
                return f"${value:.2f}"
        return str(value)
    
    # ========================================================================
    # COMPARATIVE QUESTIONS FROM CSV
    # ========================================================================
    
    def generate_comparative_qa(self) -> int:
        """Generate comparative questions from CSV metrics"""
        count_before = len(self.qa_pairs)
        
        logger.info(f"🔄 Generating comparative Q&A from CSV data...")
        
        by_quarter = {}
        for metric in self.metrics_data:
            quarter = metric['quarter']
            if quarter not in by_quarter:
                by_quarter[quarter] = []
            by_quarter[quarter].append(metric)
        
        for quarter, companies in by_quarter.items():
            if len(companies) < 2:
                continue
            
            for i in range(len(companies)):
                for j in range(i + 1, len(companies)):
                    comp1 = companies[i]
                    comp2 = companies[j]
                    
                    # Revenue comparison
                    if 'Revenue' in comp1['data'] and 'Revenue' in comp2['data']:
                        rev1 = comp1['data']['Revenue']
                        rev2 = comp2['data']['Revenue']
                        higher = comp1['company'] if rev1 > rev2 else comp2['company']
                        val1 = self._format_answer('Revenue', rev1)
                        val2 = self._format_answer('Revenue', rev2)
                        
                        q = f"Which company had higher revenue in {quarter}: {comp1['company']} or {comp2['company']}?"
                        a = f"{higher} had higher revenue ({val1} vs {val2})"
                        self.add_qa(q, a, "Multiple", "CSV comparison", 0.99, "comparative")
                    
                    # ROE comparison
                    if 'ROE' in comp1['data'] and 'ROE' in comp2['data']:
                        roe1 = comp1['data']['ROE'] * 100
                        roe2 = comp2['data']['ROE'] * 100
                        higher = comp1['company'] if roe1 > roe2 else comp2['company']
                        
                        q = f"Which company had better return on equity in {quarter}: {comp1['company']} or {comp2['company']}?"
                        a = f"{higher} had higher ROE ({roe1:.2f}% vs {roe2:.2f}%)"
                        self.add_qa(q, a, "Multiple", "CSV comparison", 0.99, "comparative")
        
        generated = len(self.qa_pairs) - count_before
        logger.info(f"   ✅ Generated {generated} comparative Q&A")
        return generated
    
    # ========================================================================
    # JSON EXTRACTION
    # ========================================================================
    
    def extract_from_single_json(self, json_path: Path) -> int:
        """Extract from ONE JSON file"""
        count_before = len(self.qa_pairs)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = json_path.stem
            company_ticker = None
            for ticker in ['AMZN', 'MSFT', 'GOOGL', 'GOOG', 'Amazon', 'Microsoft', 'Google', 'Alphabet']:
                if ticker.lower() in filename.lower():
                    company_ticker = ticker
                    break
            
            company = self.COMPANY_NAMES.get(company_ticker, company_ticker or "Unknown")
            quarter_year = self._extract_quarter_year(filename)
            
            self._extract_json_high_quality(
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
    
    def _extract_json_high_quality(self, data: any, company: str, source: str,
                                   quarter_year: str, max_questions: int = 15) -> None:
        """Extract JSON with financial context"""
        extracted = 0
        
        financial_keywords = [
            'revenue', 'sales', 'income', 'profit', 'earnings', 'ebitda',
            'cashflow', 'cash_flow', 'assets', 'liabilities', 'equity',
            'shares', 'eps', 'margin', 'ratio', 'rate', 'price'
        ]
        
        def extract_metrics(obj, path=""):
            nonlocal extracted
            if extracted >= max_questions:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if extracted >= max_questions:
                        return
                    
                    key_lower = key.lower()
                    is_metric = any(kw in key_lower for kw in financial_keywords)
                    
                    if is_metric and isinstance(value, (int, float)):
                        question = f"According to {source}, what was {company}'s {key} in {quarter_year}?"
                        answer = self._format_json_answer(key, value)
                        
                        if self.add_qa(question, answer, company, source, 0.95, "factual"):
                            extracted += 1
                    
                    elif isinstance(value, dict) and len(path) < 3:
                        extract_metrics(value, f"{path}/{key}")
            
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    if extracted >= max_questions:
                        return
                    if isinstance(item, dict):
                        extract_metrics(item, f"{path}[{idx}]")
        
        extract_metrics(data)
    
    def _format_json_answer(self, key: str, value: float) -> str:
        """Format JSON answer"""
        key_lower = key.lower()
        
        if 'eps' in key_lower or 'per_share' in key_lower:
            return f"${value:.2f}"
        elif any(x in key_lower for x in ['ratio', 'rate', 'margin', 'beta']):
            return f"{value:.2f}"
        elif any(x in key_lower for x in ['revenue', 'sales', 'income', 'profit', 'earnings', 'ebitda']):
            if value > 1e9:
                return f"${value/1e9:.2f} billion"
            elif value > 1e6:
                return f"${value/1e6:.2f} million"
            else:
                return f"${value:.2f}"
        else:
            return str(value)
    
    # ========================================================================
    # TXT EXTRACTION - V8.1 SMART EXTRACTION
    # ========================================================================
    
    def extract_from_single_txt(self, txt_path: Path) -> int:
        """Extract from ONE TXT file - SMART extraction"""
        count_before = len(self.qa_pairs)
        
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            filename = txt_path.stem
            company_ticker = None
            for ticker in ['AMZN', 'MSFT', 'GOOGL', 'GOOG', 'Amazon', 'Microsoft', 'Google', 'Alphabet']:
                if ticker.lower() in filename.lower():
                    company_ticker = ticker
                    break
            
            company = self.COMPANY_NAMES.get(company_ticker, company_ticker or "Unknown")
            quarter_year = self._extract_quarter_year(filename)
            
            extracted_count = 0
            
            # V8.1: Extract using financial patterns
            for pattern_name, pattern in self.TXT_FINANCIAL_PATTERNS.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                if matches:
                    for match in matches[:1]:  # Take first match
                        match_text = match if isinstance(match, str) else str(match)
                        if not match_text:
                            continue
                        
                        # Generate contextual question
                        question = self._generate_txt_question(company, quarter_year, pattern_name, match_text)
                        answer = self._format_txt_answer(pattern_name, match_text)
                        
                        if self.add_qa(question, answer, company, txt_path.name, 0.88, "analytical"):
                            extracted_count += 1
                        
                        if extracted_count >= self.config.QUESTIONS_PER_TXT:
                            break
                
                if extracted_count >= self.config.QUESTIONS_PER_TXT:
                    break
            
            # Fallback: Extract key financial sentences
            if extracted_count < self.config.QUESTIONS_PER_TXT:
                sentences = self._extract_financial_sentences(text)
                
                for sentence in sentences[:self.config.QUESTIONS_PER_TXT - extracted_count]:
                    question = f"According to {txt_path.stem}, what was mentioned about {company}'s financial performance?"
                    answer = sentence
                    
                    if self.add_qa(question, answer, company, txt_path.name, 0.82, "analytical"):
                        extracted_count += 1
        
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
    
    def _generate_txt_question(self, company: str, quarter: str, pattern_name: str, value: str) -> str:
        """Generate contextual question from TXT pattern"""
        questions = {
            'revenue_growth': f"What was {company}'s revenue growth rate in {quarter}?",
            'earnings_per_share': f"What were {company}'s earnings per share in {quarter}?",
            'net_income': f"What was {company}'s net income in {quarter}?",
            'operating_income': f"What was {company}'s operating income in {quarter}?",
            'operating_margin': f"What was {company}'s operating margin in {quarter}?",
            'gross_margin': f"What was {company}'s gross margin in {quarter}?",
            'profit_growth': f"What was {company}'s profit growth rate in {quarter}?",
            'segment_growth': f"What was the growth rate of {company}'s key segments in {quarter}?",
            'guidance': f"What was {company}'s guidance for the period?",
        }
        return questions.get(pattern_name, f"What financial metric was mentioned for {company} in {quarter}?")
    
    def _format_txt_answer(self, pattern_name: str, value: str) -> str:
        """Format TXT extracted answer"""
        # Value often includes % or other formatting already
        if '%' in value:
            return value
        elif any(unit in value.lower() for unit in ['billion', 'million']):
            return f"${value}"
        else:
            try:
                float_val = float(value)
                if float_val > 100:
                    return f"${value} million"
                else:
                    return f"{value}%"
            except ValueError:
                return value
    
    def _extract_financial_sentences(self, text: str) -> List[str]:
        """Extract sentences with financial keywords"""
        keywords = [
            'revenue', 'income', 'profit', 'growth', 'billion', 'million',
            'margin', 'sales', 'cash flow', 'operating', 'earnings', 'guidance'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        financial = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(kw in sentence.lower() for kw in keywords):
                if 40 < len(sentence) < 250:  # Reasonable length
                    financial.append(sentence)
        
        return financial[:20]  # Top 20
    
    def _extract_quarter_year(self, filename: str) -> str:
        """Extract quarter and year"""
        for i in range(1, 5):
            if f'Q{i}' in filename or f'q{i}' in filename:
                quarter = f'Q{i}'
                break
        else:
            quarter = "Q1"
        
        year_match = re.search(r'20\d{2}', filename)
        year = year_match.group(0) if year_match else "2024"
        
        return f"{quarter} {year}"
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def generate(self) -> List[Dict]:
        """Main generation pipeline - CSV + JSON + TXT"""
        logger.info("="*80)
        logger.info("GROUND TRUTH GENERATOR V8.1 - HIGH QUALITY EXTRACTION")
        logger.info("CSV + JSON + TXT (structured data + management insights)")
        logger.info("="*80)
        logger.info(f"📁 Data folder: {self.config.DATA_FOLDER}")
        logger.info(f"🎯 CSV: {self.config.QUESTIONS_PER_CSV} per file (100% confidence)")
        logger.info(f"🎯 JSON: {self.config.QUESTIONS_PER_JSON} per file (95% confidence)")
        logger.info(f"🎯 TXT: {self.config.QUESTIONS_PER_TXT} per file (85-90% confidence)")
        
        files = self.discover_all_files()
        
        # Extract from all sources
        self.extract_from_all_csv(files['csv'])
        self.extract_from_all_json(files['json'])
        self.extract_from_all_txt(files['txt'])
        
        # Generate comparative questions
        if self.metrics_data:
            self.generate_comparative_qa()
        
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
        logger.info("✅ SAVED HIGH-QUALITY GROUND TRUTH DATASET")
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
            
            logger.info("\n📁 Per-File Breakdown:")
            for filename, count in sorted(self.file_stats.items(), key=lambda x: -x[1]):
                if count > 0:
                    logger.info(f"  {filename}: {count} Q&A")
        
        logger.info("="*80)
        
        return {
            "total_pairs": len(self.qa_pairs),
            "json_file": str(json_file),
            "csv_file": str(csv_file),
            "file_stats": self.file_stats
        }
    
    def print_samples(self, n: int = 20) -> None:
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
        gen = EnhancedFocusedGenerator()
        qa_pairs = gen.generate()
        
        if not qa_pairs:
            logger.error("❌ No Q&A pairs generated")
            return 1
        
        gen.print_samples(n=20)
        gen.save()
        
        logger.info("\n✅ V8.1 high-quality extraction finished!")
        logger.info("   ✨ All Q&A from CSV + JSON + TXT")
        logger.info("   🎯 High confidence, diverse sources")
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
