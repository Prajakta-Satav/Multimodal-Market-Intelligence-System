"""
Ground Truth Q&A Generator v5 - Production Ready
Generates 150+ high-quality, diverse Q&A pairs using Gemini API
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

# Load environment variables
load_dotenv()

# Configure logging
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
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # Paths
    DATA_FOLDER: Path = Path(os.getenv(
        "DATA_FOLDER",
        r"D:\Multimodal-Market-Intelligence-System\data"
    ))
    OUTPUT_FOLDER: Path = Path(os.getenv(
        "OUTPUT_FOLDER",
        r"D:\Multimodal-Market-Intelligence-System\ground_truth"
    ))
    
    # Generation Parameters
    TARGET_QUESTIONS: int = int(os.getenv("TARGET_QUESTIONS", "150"))
    QUESTIONS_PER_COMPANY: int = int(os.getenv("QUESTIONS_PER_COMPANY", "10"))
    
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
        
        # Create directories
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
# MAIN GENERATOR CLASS
# ============================================================================

class GeminiGroundTruthGenerator:
    """
    Production-ready Q&A generator with robust error handling
    """
    
    # Company name mapping
    COMPANY_NAMES = {
        "AMZN": "Amazon",
        "MSFT": "Microsoft",
        "GOOGL": "Google (Alphabet)"
    }
    
    # Metric templates for question generation
    METRIC_TEMPLATES = {
        'EPS': [
            "What was {company}'s earnings per share in {quarter}?",
            "What EPS did {company} report for {quarter}?",
        ],
        'PE_Ratio': [
            "What was {company}'s P/E ratio in {quarter}?",
            "What was the price-to-earnings ratio for {company} in {quarter}?",
        ],
        'ROE': [
            "What was {company}'s return on equity in {quarter}?",
            "What ROE did {company} achieve in {quarter}?",
        ],
        'ROA': [
            "What was {company}'s return on assets in {quarter}?",
        ],
        'Revenue': [
            "What was {company}'s total revenue in {quarter}?",
            "How much revenue did {company} generate in {quarter}?",
        ],
        'Profit': [
            "What was {company}'s net profit in {quarter}?",
        ],
        'MarketCap': [
            "What was {company}'s market capitalization in {quarter}?",
        ],
        'PB_Ratio': [
            "What was {company}'s price-to-book ratio in {quarter}?",
        ],
        'Beta': [
            "What was {company}'s stock beta in {quarter}?",
        ],
        'EBITDA': [
            "What was {company}'s EBITDA in {quarter}?",
        ],
        'DebtToEquity': [
            "What was {company}'s debt-to-equity ratio in {quarter}?",
        ],
    }
    
    def __init__(self):
        """Initialize generator with config validation"""
        self.config = Config
        self.qa_pairs: List[Dict] = []
        self.questions_set: Set[str] = set()
        self.use_gemini = False
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration. Check .env file.")
        
        # Initialize Gemini API
        self._init_gemini()
    
    def _init_gemini(self) -> None:
        """Initialize Gemini API with error handling"""
        try:
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
            self.use_gemini = True
            logger.info(f"✅ Gemini API configured ({self.config.GEMINI_MODEL})")
        except Exception as e:
            logger.error(f"⚠️  Failed to configure Gemini API: {e}")
            logger.warning("   Will use local generation only")
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
        """
        Add Q&A pair with duplicate checking
        
        Returns:
            bool: True if added, False if duplicate
        """
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
    
    def extract_metrics_from_csv(self) -> List[Dict]:
        """
        Extract financial metrics from CSV files
        
        Returns:
            List of dicts containing company/quarter metrics
        """
        metrics_data = []
        csv_files = list(self.config.DATA_FOLDER.rglob("*.csv"))
        
        logger.info(f"📊 Found {len(csv_files)} CSV files")
        
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
                        
                        # Collect non-null metrics
                        metrics = {
                            col: latest[col]
                            for col in latest.index
                            if col not in ['Date', 'Ticker', 'Quarter']
                            and pd.notna(latest[col])
                        }
                        
                        metrics_data.append({
                            "company": company,
                            "ticker": ticker,
                            "quarter": quarter,
                            "metrics": metrics,
                            "source": csv_file.name
                        })
                
            except Exception as e:
                logger.error(f"⚠️  Error reading {csv_file.name}: {e}")
                continue
        
        logger.info(f"   Found {len(metrics_data)} company/quarter combinations")
        return metrics_data
    
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
    
    def generate_local_qa(self, metrics_data: List[Dict]) -> int:
        """Generate factual Q&A from local data"""
        count_before = len(self.qa_pairs)
        logger.info("📊 Generating local factual Q&A...")
        
        for data in metrics_data:
            if len(self.qa_pairs) >= self.config.TARGET_QUESTIONS:
                break
            
            company = data['company']
            quarter = data['quarter']
            metrics = data['metrics']
            source = data['source']
            
            # Generate Q&A for each metric
            for metric_key, templates in self.METRIC_TEMPLATES.items():
                if metric_key in metrics:
                    value = metrics[metric_key]
                    answer = self._format_answer(metric_key, value)
                    question = templates[0].format(company=company, quarter=quarter)
                    
                    self.add_qa(
                        question=question,
                        answer=answer,
                        company=company,
                        source=source,
                        confidence=1.0,
                        qa_type="factual"
                    )
        
        generated = len(self.qa_pairs) - count_before
        logger.info(f"   ✅ Generated {generated} factual Q&A")
        return generated
    
    def _call_gemini_with_retry(self, prompt: str) -> str:
        """
        Call Gemini API with retry logic
        
        Raises:
            QuotaExceededException: When quota is exceeded
            APIError: For other API errors
        """
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            
            except ResourceExhausted as e:
                logger.warning(f"⚠️  Quota exceeded (attempt {attempt + 1}/{self.config.MAX_RETRIES})")
                if attempt < self.config.MAX_RETRIES - 1:
                    logger.info(f"   Waiting {self.config.RETRY_DELAY_SECONDS}s before retry...")
                    time.sleep(self.config.RETRY_DELAY_SECONDS)
                else:
                    raise QuotaExceededException()
            
            except Exception as e:
                logger.error(f"⚠️  API error (attempt {attempt + 1}/{self.config.MAX_RETRIES}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(5)
                else:
                    raise APIError(f"Failed after {self.config.MAX_RETRIES} attempts: {e}")
        
        raise APIError("Unexpected error in retry logic")
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from markdown code blocks"""
        # Try ```json block first
        if "```json" in response_text:
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                return match.group(1)
        
        # Try generic ``` block
        if "```" in response_text:
            match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                return match.group(1)
        
        return response_text
    
    def generate_gemini_qa(self, metrics_data: List[Dict]) -> int:
        """Generate complex analytical Q&A using Gemini API"""
        if not self.use_gemini:
            logger.warning("⚠️  Gemini API not configured - skipping AI generation")
            return 0
        
        count_before = len(self.qa_pairs)
        logger.info("🤖 Generating Gemini-powered complex Q&A...")
        logger.info(f"   Rate limit: {self.config.API_DELAY_SECONDS}s delay between requests")
        
        for idx, data in enumerate(metrics_data, 1):
            if len(self.qa_pairs) >= self.config.TARGET_QUESTIONS:
                break
            
            company = data['company']
            quarter = data['quarter']
            ticker = data['ticker']
            metrics = data['metrics']
            
            # Build prompt
            metrics_str = json.dumps(metrics, indent=2)
            prompt = self._build_gemini_prompt(
                company, ticker, quarter, metrics_str,
                self.config.QUESTIONS_PER_COMPANY
            )
            
            try:
                # Call API with retry
                response_text = self._call_gemini_with_retry(prompt)
                
                # Extract and parse JSON
                json_text = self._extract_json_from_response(response_text)
                qa_list = json.loads(json_text)
                
                # Add Q&A pairs
                added = 0
                for item in qa_list:
                    if len(self.qa_pairs) >= self.config.TARGET_QUESTIONS:
                        break
                    
                    question = item.get('question', '').strip()
                    answer = item.get('answer', '').strip()
                    qa_type = item.get('type', 'analytical')
                    
                    if question and answer:
                        if self.add_qa(
                            question=question,
                            answer=answer,
                            company=company,
                            source=f"gemini-{quarter}",
                            confidence=0.90,
                            qa_type=qa_type
                        ):
                            added += 1
                
                logger.info(f"   ✅ [{idx}/{len(metrics_data)}] {company} {quarter}: {added} Q&A")
                
                # Rate limiting delay (10 seconds)
                if idx < len(metrics_data):
                    logger.debug(f"   ⏳ Waiting {self.config.API_DELAY_SECONDS}s...")
                    time.sleep(self.config.API_DELAY_SECONDS)
            
            except QuotaExceededException:
                logger.error("❌ API quota exceeded. Cannot continue.")
                break
            
            except json.JSONDecodeError as e:
                logger.error(f"   ⚠️  JSON parse error for {company} {quarter}: {e}")
                logger.debug(f"   Response: {response_text[:200]}...")
                continue
            
            except APIError as e:
                logger.error(f"   ⚠️  {e}")
                continue
            
            except Exception as e:
                logger.error(f"   ⚠️  Unexpected error for {company} {quarter}: {e}")
                continue
        
        generated = len(self.qa_pairs) - count_before
        logger.info(f"   ✅ Generated {generated} Gemini-powered Q&A")
        return generated
    
    def _build_gemini_prompt(
        self,
        company: str,
        ticker: str,
        quarter: str,
        metrics_str: str,
        num_questions: int
    ) -> str:
        """Build Gemini API prompt"""
        return f"""You are a financial analyst creating ground truth Q&A pairs for evaluating AI systems.

Given the following financial data for {company} ({ticker}) in {quarter}:

{metrics_str}

Generate {num_questions} diverse, high-quality question-answer pairs. Include:
- 3 analytical/interpretive questions (e.g., "What does a P/E of X suggest?")
- 3 comparative questions (e.g., "How does this compare to industry average?")
- 2 trend/insight questions (e.g., "What could explain the ROE increase?")
- 2 complex calculation questions (e.g., "What's the profit margin?")

Requirements:
- Questions must be specific, precise, and answerable from the data
- Answers should be concise (1-3 sentences max)
- Use proper financial terminology
- Vary question structure and complexity
- Each question should be unique and non-repetitive

Return as valid JSON array:
[
  {{"question": "...", "answer": "...", "type": "analytical"}},
  {{"question": "...", "answer": "...", "type": "comparative"}},
  ...
]
"""
    
    def generate_comparative_qa(self, metrics_data: List[Dict]) -> int:
        """Generate cross-company comparative Q&A"""
        count_before = len(self.qa_pairs)
        logger.info("📈 Generating comparative Q&A...")
        
        # Group by quarter
        by_quarter: Dict[str, List[Dict]] = {}
        for data in metrics_data:
            quarter = data['quarter']
            by_quarter.setdefault(quarter, []).append(data)
        
        # Generate comparisons
        for quarter, companies in by_quarter.items():
            if len(companies) < 2:
                continue
            
            for i in range(len(companies)):
                for j in range(i + 1, len(companies)):
                    if len(self.qa_pairs) >= self.config.TARGET_QUESTIONS:
                        break
                    
                    comp1, comp2 = companies[i], companies[j]
                    
                    # Revenue comparison
                    if 'Revenue' in comp1['metrics'] and 'Revenue' in comp2['metrics']:
                        self._add_comparison_qa(comp1, comp2, 'Revenue', quarter, 'revenue')
                    
                    # ROE comparison
                    if 'ROE' in comp1['metrics'] and 'ROE' in comp2['metrics']:
                        self._add_comparison_qa(comp1, comp2, 'ROE', quarter, 'return on equity')
        
        generated = len(self.qa_pairs) - count_before
        logger.info(f"   ✅ Generated {generated} comparative Q&A")
        return generated
    
    def _add_comparison_qa(
        self,
        comp1: Dict,
        comp2: Dict,
        metric: str,
        quarter: str,
        metric_name: str
    ) -> None:
        """Helper to add comparison Q&A"""
        val1 = comp1['metrics'][metric]
        val2 = comp2['metrics'][metric]
        
        if metric == 'Revenue':
            higher = comp1['company'] if val1 > val2 else comp2['company']
            formatted1 = f"${val1/1e9:.2f}B"
            formatted2 = f"${val2/1e9:.2f}B"
        elif metric == 'ROE':
            val1_pct = val1 * 100
            val2_pct = val2 * 100
            higher = comp1['company'] if val1 > val2 else comp2['company']
            formatted1 = f"{val1_pct:.2f}%"
            formatted2 = f"{val2_pct:.2f}%"
        else:
            return
        
        question = f"Which company had higher {metric_name} in {quarter}: {comp1['company']} or {comp2['company']}?"
        answer = f"{higher} had higher {metric_name} ({formatted1} vs {formatted2})"
        
        self.add_qa(
            question=question,
            answer=answer,
            company="Multiple",
            source="comparative",
            confidence=0.95,
            qa_type="comparative"
        )
    
    def generate(self) -> List[Dict]:
        """Main generation pipeline"""
        logger.info("="*80)
        logger.info("GEMINI-POWERED GROUND TRUTH Q&A GENERATOR V5")
        logger.info("="*80)
        logger.info(f"🎯 Target: {self.config.TARGET_QUESTIONS} Q&A pairs")
        logger.info(f"📁 Data: {self.config.DATA_FOLDER}")
        logger.info(f"🤖 Gemini: {'Enabled' if self.use_gemini else 'Disabled'}")
        
        # Extract data
        metrics_data = self.extract_metrics_from_csv()
        
        if not metrics_data:
            logger.error("❌ No data found. Check DATA_FOLDER path.")
            return []
        
        # Phase 1: Local factual Q&A
        self.generate_local_qa(metrics_data)
        
        # Phase 2: Comparative Q&A
        if len(self.qa_pairs) < self.config.TARGET_QUESTIONS:
            self.generate_comparative_qa(metrics_data)
        
        # Phase 3: Gemini-powered Q&A
        if self.use_gemini and len(self.qa_pairs) < self.config.TARGET_QUESTIONS:
            self.generate_gemini_qa(metrics_data)
        
        # Trim to target
        self.qa_pairs = self.qa_pairs[:self.config.TARGET_QUESTIONS]
        
        logger.info(f"📊 Total generated: {len(self.qa_pairs)} Q&A pairs")
        return self.qa_pairs
    
    def save(self) -> Dict[str, any]:
        """Save Q&A pairs to files"""
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
        logger.info("✅ SAVED GROUND TRUTH DATASET")
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
        
        logger.info("="*80)
        
        return {
            "total_pairs": len(self.qa_pairs),
            "json_file": str(json_file),
            "csv_file": str(csv_file)
        }
    
    def print_samples(self, n: int = 10) -> None:
        """Print sample Q&A pairs"""
        if not self.qa_pairs:
            return
        
        logger.info(f"\n📋 Sample Q&A Pairs (first {n}):\n")
        for i, qa in enumerate(self.qa_pairs[:n], 1):
            print(f"{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print(f"   Type: {qa['type']} | Confidence: {qa['confidence']:.0%}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    try:
        # Initialize generator
        gen = GeminiGroundTruthGenerator()
        
        # Generate Q&A pairs
        qa_pairs = gen.generate()
        
        if not qa_pairs:
            logger.error("❌ No Q&A pairs generated")
            return 1
        
        # Print samples
        gen.print_samples(n=10)
        
        # Save results
        gen.save()
        
        logger.info("\n✅ Generation completed successfully!")
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Generation interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
