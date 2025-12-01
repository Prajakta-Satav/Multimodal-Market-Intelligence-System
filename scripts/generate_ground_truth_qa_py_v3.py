"""
Ground Truth Q&A Generator v2
Extract exactly 100 unique Q&A pairs from PDFs, JSONs, and CSVs
No generic/placeholder questions - only real data extraction
"""

import json
import pandas as pd
import PyPDF2
import os
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime
import re


class GroundTruthQAGeneratorV2:
    """Generate 100 unique, real Q&A pairs from actual data files"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.qa_pairs = []
        self.questions_set: Set[str] = set()  # Track unique questions
        self.max_questions = 100
    
    def add_qa(self, question: str, answer: str, company: str, source: str, 
               confidence: float = 1.0, qa_type: str = "factual") -> bool:
        """Add Q&A pair, ensuring no duplicates. Returns True if added."""
        
        # Check if question already exists
        if question.lower() in self.questions_set:
            return False
        
        self.questions_set.add(question.lower())
        
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
    
    def extract_from_csv(self) -> int:
        """Extract Q&A from stock fundamentals CSV"""
        count_before = len(self.qa_pairs)
        
        csv_files = list(self.data_folder.rglob("*.csv"))
        print(f"\n📊 Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            print(f"\n   Processing: {csv_file.name}")
            
            try:
                df = pd.read_csv(csv_file)
                
                # Extract from daily_stocks_fundamentals_2024.csv
                if "daily_stocks" in csv_file.name.lower():
                    
                    # Get unique tickers
                    tickers = df['Ticker'].unique()
                    
                    for ticker in tickers:
                        ticker_data = df[df['Ticker'] == ticker]
                        
                        # Get latest quarter data
                        if 'Quarter' in ticker_data.columns:
                            latest_quarter = ticker_data.iloc[0]['Quarter']
                        else:
                            latest_quarter = "Q1 2024"
                        
                        # Extract key metrics
                        try:
                            latest = ticker_data.iloc[0]
                            
                            # Q&A pairs from CSV
                            if pd.notna(latest.get('EPS', None)):
                                company_name = f"Company {ticker}"
                                eps = latest['EPS']
                                q = f"What was the EPS for {ticker} in {latest_quarter}?"
                                a = f"${eps:.2f}"
                                self.add_qa(q, a, company_name, f"{csv_file.name}", 0.95)
                            
                            if pd.notna(latest.get('PE_Ratio', None)):
                                pe = latest['PE_Ratio']
                                q = f"What was the P/E ratio for {ticker} in {latest_quarter}?"
                                a = f"{pe:.2f}"
                                self.add_qa(q, a, f"Company {ticker}", f"{csv_file.name}", 0.95)
                            
                            if pd.notna(latest.get('PB_Ratio', None)):
                                pb = latest['PB_Ratio']
                                q = f"What was the price-to-book ratio for {ticker}?"
                                a = f"{pb:.2f}"
                                self.add_qa(q, a, f"Company {ticker}", f"{csv_file.name}", 0.95)
                            
                            if pd.notna(latest.get('ROE', None)):
                                roe = latest['ROE'] * 100
                                q = f"What was the return on equity (ROE) for {ticker}?"
                                a = f"{roe:.2f}%"
                                self.add_qa(q, a, f"Company {ticker}", f"{csv_file.name}", 0.95)
                            
                            if pd.notna(latest.get('ROA', None)):
                                roa = latest['ROA'] * 100
                                q = f"What was the return on assets (ROA) for {ticker}?"
                                a = f"{roa:.2f}%"
                                self.add_qa(q, a, f"Company {ticker}", f"{csv_file.name}", 0.95)
                            
                            if pd.notna(latest.get('Revenue', None)):
                                revenue = latest['Revenue']
                                q = f"What was the revenue for {ticker} in {latest_quarter}?"
                                a = f"${revenue/1e9:.2f}B"
                                self.add_qa(q, a, f"Company {ticker}", f"{csv_file.name}", 0.95)
                            
                            if pd.notna(latest.get('Profit', None)):
                                profit = latest['Profit']
                                q = f"What was the profit for {ticker}?"
                                a = f"${profit/1e9:.2f}B"
                                self.add_qa(q, a, f"Company {ticker}", f"{csv_file.name}", 0.95)
                            
                            # Comparative questions (YoY)
                            if len(ticker_data) > 1:
                                latest_eps = ticker_data.iloc[0].get('EPS')
                                earlier_eps = ticker_data.iloc[-1].get('EPS')
                                
                                if pd.notna(latest_eps) and pd.notna(earlier_eps) and earlier_eps != 0:
                                    growth = ((latest_eps - earlier_eps) / earlier_eps) * 100
                                    q = f"What was the EPS growth for {ticker} between the earliest and latest data points?"
                                    a = f"{growth:.1f}%"
                                    self.add_qa(q, a, f"Company {ticker}", f"{csv_file.name}", 0.80, "comparative")
                        
                        except Exception as e:
                            continue
                        
                        # Stop if we've reached max questions
                        if len(self.qa_pairs) >= self.max_questions:
                            return len(self.qa_pairs) - count_before
                
            except Exception as e:
                print(f"      ⚠️  Error: {e}")
                continue
        
        return len(self.qa_pairs) - count_before
    
    def extract_from_pdfs(self) -> int:
        """Extract Q&A from earnings release PDFs"""
        count_before = len(self.qa_pairs)
        
        pdf_files = list(self.data_folder.rglob("*.pdf"))
        print(f"\n📄 Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            print(f"\n   Processing: {pdf_file.name}")
            
            try:
                # Extract company and quarter from filename
                filename = pdf_file.stem  # AMZN-Q1-2024-Earnings-Release
                parts = filename.split("-")
                
                if len(parts) >= 2:
                    ticker = parts[0]
                    quarter = parts[1] if len(parts) > 1 else "Q1"
                    company_name = self._get_company_name(ticker)
                else:
                    continue
                
                # Extract text from PDF
                text = self._extract_pdf_text(pdf_file)
                
                if not text:
                    continue
                
                # Extract financial metrics from text
                metrics = self._extract_metrics_from_text(text, ticker, quarter, company_name)
                
                for metric in metrics:
                    if len(self.qa_pairs) >= self.max_questions:
                        break
                    self.add_qa(**metric)
                
                if len(self.qa_pairs) >= self.max_questions:
                    break
                    
            except Exception as e:
                print(f"      ⚠️  Error: {e}")
                continue
        
        return len(self.qa_pairs) - count_before
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages[:5]:  # Read first 5 pages only
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"      Error reading PDF: {e}")
            return ""
    
    def _get_company_name(self, ticker: str) -> str:
        """Get full company name from ticker"""
        company_map = {
            "AMZN": "Amazon.com, Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
        }
        return company_map.get(ticker, ticker)
    
    def _extract_metrics_from_text(self, text: str, ticker: str, quarter: str, 
                                   company: str) -> List[Dict]:
        """Extract financial metrics and create Q&A from PDF text"""
        metrics = []
        
        # Revenue extraction
        revenue_pattern = r'(?:net\s+)?sales?\s+(?:increased|were|totaled|of)\s+(?:\$)?(\d+(?:\.\d+)?)\s*(?:billion|million|B|M)'
        revenue_matches = re.findall(revenue_pattern, text, re.IGNORECASE)
        
        if revenue_matches:
            revenue = float(revenue_matches[0])
            unit = "B" if "billion" in text[max(0, text.find(revenue_matches[0])-50):text.find(revenue_matches[0])+50].lower() else "M"
            
            metrics.append({
                "question": f"What were {company}'s net sales in {quarter}?",
                "answer": f"${revenue} {unit}",
                "company": company,
                "source": f"{ticker}-{quarter}-Earnings.pdf",
                "confidence": 0.85,
                "qa_type": "factual"
            })
        
        # Operating income extraction
        op_income_pattern = r'operating\s+income\s+(?:increased|was|of|totaled)\s+(?:\$)?(\d+(?:\.\d+)?)\s*(?:billion|million)'
        op_income_matches = re.findall(op_income_pattern, text, re.IGNORECASE)
        
        if op_income_matches:
            op_income = float(op_income_matches[0])
            
            metrics.append({
                "question": f"What was {company}'s operating income in {quarter}?",
                "answer": f"${op_income}B",
                "company": company,
                "source": f"{ticker}-{quarter}-Earnings.pdf",
                "confidence": 0.85,
                "qa_type": "factual"
            })
        
        # EPS extraction
        eps_pattern = r'(?:diluted|basic)?\s*(?:earnings?|EPS)\s+(?:per\s+share)?\s+(?:of|was|were|totaled)\s+(?:\$)?(\d+(?:\.\d+)?)'
        eps_matches = re.findall(eps_pattern, text, re.IGNORECASE)
        
        if eps_matches:
            eps = float(eps_matches[0])
            
            metrics.append({
                "question": f"What was {company}'s diluted EPS in {quarter}?",
                "answer": f"${eps:.2f}",
                "company": company,
                "source": f"{ticker}-{quarter}-Earnings.pdf",
                "confidence": 0.80,
                "qa_type": "factual"
            })
        
        # YoY growth extraction
        yoy_pattern = r'(?:increased|grew|growth)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)'
        yoy_matches = re.findall(yoy_pattern, text, re.IGNORECASE)
        
        if yoy_matches:
            growth = float(yoy_matches[0])
            
            metrics.append({
                "question": f"What was the year-over-year growth rate for {company} in {quarter}?",
                "answer": f"{growth}% YoY",
                "company": company,
                "source": f"{ticker}-{quarter}-Earnings.pdf",
                "confidence": 0.75,
                "qa_type": "comparative"
            })
        
        # AWS/segment revenue (Amazon specific)
        if ticker == "AMZN":
            aws_pattern = r'AWS\s+(?:segment\s+)?sales?\s+(?:increased|were|of)\s+(?:\$)?(\d+(?:\.\d+)?)\s*(?:billion|B)'
            aws_matches = re.findall(aws_pattern, text, re.IGNORECASE)
            
            if aws_matches:
                aws_revenue = float(aws_matches[0])
                
                metrics.append({
                    "question": f"What was Amazon's AWS segment revenue in {quarter}?",
                    "answer": f"${aws_revenue}B",
                    "company": company,
                    "source": f"{ticker}-{quarter}-Earnings.pdf",
                    "confidence": 0.85,
                    "qa_type": "factual"
                })
        
        return metrics
    
    def extract_from_json(self) -> int:
        """Extract Q&A from earnings JSON files"""
        count_before = len(self.qa_pairs)
        
        json_files = list(self.data_folder.rglob("*.json"))
        print(f"\n📋 Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            # Skip non-earnings JSONs
            if "embedding" in json_file.name.lower() or "config" in json_file.name.lower():
                continue
            
            print(f"\n   Processing: {json_file.name}")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract company info
                if isinstance(data, dict):
                    metadata = data.get('metadata', {})
                    company = metadata.get('company_name', {}).get('value', 'Unknown')
                    ticker = metadata.get('ticker', {}).get('value', 'UNKNOWN')
                    quarter = metadata.get('quarter', {}).get('value', 'Q1')
                    
                    # Extract income statement data
                    income_stmt = data.get('income_statement', {})
                    
                    if 'total_revenues' in income_stmt:
                        value = income_stmt['total_revenues'].get('value', 0)
                        q = f"What was {company} total revenue in {quarter}?"
                        a = f"${value:.2f}B" if value > 100 else f"${value:.2f}M"
                        self.add_qa(q, a, company, json_file.name, 1.0)
                    
                    if 'net_income' in income_stmt:
                        value = income_stmt['net_income'].get('value', 0)
                        q = f"What was {company} net income in {quarter}?"
                        a = f"${value:.2f}B" if value > 10 else f"${value:.2f}M"
                        self.add_qa(q, a, company, json_file.name, 1.0)
                    
                    if 'operating_income' in income_stmt:
                        value = income_stmt['operating_income'].get('value', 0)
                        q = f"What was {company} operating income in {quarter}?"
                        a = f"${value:.2f}B"
                        self.add_qa(q, a, company, json_file.name, 1.0)
                    
                    # Extract segments
                    segments = data.get('segments', {})
                    for segment_name, segment_data in segments.items():
                        if isinstance(segment_data, dict):
                            if 'revenue' in segment_data:
                                value = segment_data['revenue'].get('value', 0)
                                q = f"What was {company} {segment_name} segment revenue?"
                                a = f"${value:.2f}B"
                                self.add_qa(q, a, company, json_file.name, 0.95)
                
                if len(self.qa_pairs) >= self.max_questions:
                    break
                    
            except Exception as e:
                print(f"      ⚠️  Error: {e}")
                continue
        
        return len(self.qa_pairs) - count_before
    
    def generate(self) -> List[Dict]:
        """Generate all Q&A pairs"""
        print("="*80)
        print("GROUND TRUTH Q&A GENERATOR V2")
        print("="*80)
        print(f"\n🎯 Target: 100 unique Q&A pairs")
        print(f"📁 Data folder: {self.data_folder}")
        
        # Extract from different sources
        csv_count = self.extract_from_csv()
        print(f"   ✅ Extracted {csv_count} Q&A from CSV files")
        
        if len(self.qa_pairs) < self.max_questions:
            pdf_count = self.extract_from_pdfs()
            print(f"   ✅ Extracted {pdf_count} Q&A from PDF files")
        
        if len(self.qa_pairs) < self.max_questions:
            json_count = self.extract_from_json()
            print(f"   ✅ Extracted {json_count} Q&A from JSON files")
        
        # Trim to exactly 100
        self.qa_pairs = self.qa_pairs[:self.max_questions]
        
        return self.qa_pairs
    
    def save(self, output_folder: str = "ground_truth") -> Dict:
        """Save Q&A pairs to files"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Save as JSON
        json_file = output_path / "ground_truth_qa_pairs.json"
        with open(json_file, 'w') as f:
            json.dump(self.qa_pairs, f, indent=2)
        
        # Save as CSV
        csv_file = output_path / "ground_truth_qa_pairs.csv"
        df = pd.DataFrame(self.qa_pairs)
        df.to_csv(csv_file, index=False)
        
        print("\n" + "="*80)
        print("✅ SAVED GROUND TRUTH DATASET")
        print("="*80)
        print(f"📄 JSON: {json_file}")
        print(f"📊 CSV: {csv_file}")
        print(f"📈 Total Q&A pairs: {len(self.qa_pairs)}")
        
        # Show summary
        if self.qa_pairs:
            df_qa = pd.DataFrame(self.qa_pairs)
            print(f"\n📋 Summary:")
            print(f"  Companies: {df_qa['company'].nunique()}")
            print(f"  Types: {df_qa['type'].value_counts().to_dict()}")
            print(f"  Avg Confidence: {df_qa['confidence'].mean():.2f}")
        
        print("\n" + "="*80)
        
        return {
            "total_pairs": len(self.qa_pairs),
            "json_file": str(json_file),
            "csv_file": str(csv_file)
        }


if __name__ == "__main__":
    # Initialize and generate
    gen = GroundTruthQAGeneratorV2(data_folder="data")
    
    # Generate Q&A pairs
    qa_pairs = gen.generate()
    
    # Show samples
    if qa_pairs:
        print("\n📋 Sample Q&A Pairs:")
        for i, qa in enumerate(qa_pairs[:5], 1):
            print(f"\n{i}. Q: {qa['question']}")
            print(f"   A: {qa['answer']}")
            print(f"   Source: {qa['source']}")
    
    # Save
    gen.save()