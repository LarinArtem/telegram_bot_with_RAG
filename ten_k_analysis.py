import openai
import numpy as np
import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import tiktoken
import time
import random
from edgar import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
API_KEY = "YOUR_API_KEY"

@dataclass
class AnalysisResult:
    """Data class to store analysis results"""
    risk_factors_summary: str
    risk_factors_grade: int
    financial_statements_summary: str
    financial_statements_grade: int
    mda_summary: str
    mda_grade: int
    overall_grade: float
    overall_summary: str
    recommendation: str
    future_perspective: str
    timestamp: str

class TenKAnalyzer:
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1", embedding_model: str = "text-embedding-3-large"):

        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.max_tokens_per_chunk = 8000  
        self.max_analysis_tokens = 25000  
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def chunk_document(self, text: str, max_tokens: int = None) -> List[str]:
        if max_tokens is None:
            max_tokens = self.max_tokens_per_chunk
            
        chunks = []

        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            if paragraph_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                temp_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    
                    if temp_tokens + sentence_tokens <= max_tokens:
                        temp_chunk += sentence + " "
                        temp_tokens += sentence_tokens
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + " "
                        temp_tokens = sentence_tokens
                
                if temp_chunk:
                    chunks.append(temp_chunk.strip())

            elif current_tokens + paragraph_tokens <= max_tokens:
                current_chunk += paragraph + "\n\n"
                current_tokens += paragraph_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
                current_tokens = paragraph_tokens
        

        if current_chunk:
            chunks.append(current_chunk.strip())

        valid_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = self.count_tokens(chunk)
            if chunk_tokens <= max_tokens:
                valid_chunks.append(chunk)
                logger.info(f"Chunk {i+1}: {chunk_tokens} tokens")
            else:
                logger.warning(f"Chunk {i+1} still too large ({chunk_tokens} tokens), truncating...")
                tokens = self.tokenizer.encode(chunk)
                truncated_tokens = tokens[:max_tokens]
                truncated_chunk = self.tokenizer.decode(truncated_tokens)
                valid_chunks.append(truncated_chunk)
                logger.info(f"Chunk {i+1}: truncated to {len(truncated_tokens)} tokens")
            
        return valid_chunks
    
    def create_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        embeddings = []
        
        for i, chunk in enumerate(text_chunks):
            try:
                if i > 0:
                    time.sleep(0.1) 

                token_count = self.count_tokens(chunk)
                logger.info(f"Creating embedding for chunk {i+1}/{len(text_chunks)} ({token_count} tokens)")
                
                if token_count > 8192: 
                    logger.warning(f"Chunk {i+1} exceeds 8192 tokens ({token_count}), truncating...")
                    tokens = self.tokenizer.encode(chunk)
                    truncated_tokens = tokens[:8000]  
                    chunk = self.tokenizer.decode(truncated_tokens)
                    logger.info(f"Truncated chunk {i+1} to {len(truncated_tokens)} tokens")
                
                response = self._make_api_request_with_retry(
                    lambda: self.client.embeddings.create(
                        input=chunk,
                        model=self.embedding_model
                    )
                )
                embeddings.append(response.data[0].embedding)
                
            except Exception as e:
                logger.error(f"Error creating embedding for chunk {i+1}: {e}")
                embedding_dim = self._get_embedding_dimension()
                embeddings.append([0.0] * embedding_dim)
                
        return embeddings
    
    def _get_embedding_dimension(self) -> int:
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        return model_dimensions.get(self.embedding_model, 1536)
    
    def _make_api_request_with_retry(self, request_func, max_retries: int = 5):
        for attempt in range(max_retries):
            try:
                return request_func()
            except Exception as e:
                if attempt == max_retries - 1:  
                    raise e

                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  
                    logger.warning(f"Rate limit hit, waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def _extract_key_sections(self, text: str) -> Dict[str, str]:
        sections = {
            'risk_factors': '',
            'financial_statements': '',
            'mda': '',
            'business_overview': ''
        }
        text_lower = text.lower()

        patterns = {
            'risk_factors': [
                r'item\s*1a[\.\s]*risk\s*factors',
                r'risk\s*factors',
                r'principal\s*risks'
            ],
            'mda': [
                r'item\s*7[\.\s]*management[\s\']*s\s*discussion\s*and\s*analysis',
                r'management[\s\']*s\s*discussion\s*and\s*analysis',
                r'md&a'
            ],
            'financial_statements': [
                r'item\s*8[\.\s]*financial\s*statements',
                r'consolidated\s*statements',
                r'financial\s*statements'
            ],
            'business_overview': [
                r'item\s*1[\.\s]*business',
                r'business\s*overview',
                r'company\s*overview'
            ]
        }

        for section_name, section_patterns in patterns.items():
            for pattern in section_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    start_pos = match.start()
                    end_patterns = [r'item\s*\d+[a-z]*[\.\s]', r'\n\s*\n\s*[A-Z][A-Z\s]{10,}']
                    end_pos = len(text)
                    
                    for end_pattern in end_patterns:
                        end_matches = list(re.finditer(end_pattern, text_lower[start_pos + 1000:]))
                        if end_matches:
                            end_pos = start_pos + 1000 + end_matches[0].start()
                            break
                    
                    section_text = text[start_pos:end_pos]
                    if len(section_text) > len(sections[section_name]):
                        sections[section_name] = section_text
                    break
        
        return sections
    
    def _create_focused_analysis_content(self, document_text: str) -> str:
        sections = self._extract_key_sections(document_text)

        focused_content = ""
        if sections['business_overview']:
            business_tokens = self.tokenizer.encode(sections['business_overview'])
            if len(business_tokens) > 2000:
                business_tokens = business_tokens[:2000]
            focused_content += "BUSINESS OVERVIEW:\n" + self.tokenizer.decode(business_tokens) + "\n\n"

        if sections['risk_factors']:
            risk_tokens = self.tokenizer.encode(sections['risk_factors'])
            if len(risk_tokens) > 4000:
                risk_tokens = risk_tokens[:4000]
            focused_content += "RISK FACTORS:\n" + self.tokenizer.decode(risk_tokens) + "\n\n"

        if sections['mda']:
            mda_tokens = self.tokenizer.encode(sections['mda'])
            if len(mda_tokens) > 4000:
                mda_tokens = mda_tokens[:4000]
            focused_content += "MANAGEMENT DISCUSSION & ANALYSIS:\n" + self.tokenizer.decode(mda_tokens) + "\n\n"

        if sections['financial_statements']:
            financial_tokens = self.tokenizer.encode(sections['financial_statements'])
            if len(financial_tokens) > 6000:
                financial_tokens = financial_tokens[:6000]
            focused_content += "FINANCIAL STATEMENTS:\n" + self.tokenizer.decode(financial_tokens) + "\n\n"

        if not focused_content.strip():
            logger.warning("Key sections not found, using first part of document")
            doc_tokens = self.tokenizer.encode(document_text)
            if len(doc_tokens) > 15000:
                doc_tokens = doc_tokens[:15000]
            focused_content = self.tokenizer.decode(doc_tokens)

        content_tokens = self.count_tokens(focused_content)
        if content_tokens > self.max_analysis_tokens:
            logger.warning(f"Focused content still too large ({content_tokens} tokens), truncating to {self.max_analysis_tokens}")
            tokens = self.tokenizer.encode(focused_content)
            truncated_tokens = tokens[:self.max_analysis_tokens]
            focused_content = self.tokenizer.decode(truncated_tokens)
        
        logger.info(f"Created focused analysis content: {self.count_tokens(focused_content)} tokens")
        return focused_content
    
    def get_enhanced_prompt(self) -> str:
        return """
        Act as a Senior Financial Analyst with expertise in SEC filings analysis. 
        
        Analyze the provided 10-K filing content with focus on investment decision-making. 
        Your analysis should be thorough yet concise, suitable for institutional investors.
        
        IMPORTANT: Structure your response EXACTLY as follows with clear section headers:
        
        ## 1. RISK FACTORS ANALYSIS
        [Analyze the most critical business risks (regulatory, operational, market, financial)]
        [Assess risk severity and potential impact on business operations]
        [Highlight any new or escalating risks]
        Grade: [X]/5 (where 1=High Risk/Poor Disclosure, 5=Well-Managed Risks/Excellent Disclosure)
        
        ## 2. FINANCIAL STATEMENTS ANALYSIS  
        [Revenue trends, profitability metrics, and growth patterns]
        [Balance sheet strength: debt levels, liquidity, working capital]
        [Cash flow analysis and key financial ratios]
        Grade: [X]/5 (where 1=Poor Financial Health, 5=Excellent Financial Performance)
        
        ## 3. MANAGEMENT DISCUSSION & ANALYSIS (MD&A)
        [Management's strategic vision and execution capability]
        [Transparency in addressing challenges and forward-looking guidance quality]
        [Capital allocation strategy and operational efficiency initiatives]
        Grade: [X]/5 (where 1=Poor Management Insight, 5=Excellent Strategic Communication)
        
        ## 4. OVERALL ASSESSMENT
        Overall Grade: [X.X]/5 (Weighted: Risk 25%, Financial 50%, MD&A 25%)
        
        Investment Recommendation: [BUY/HOLD/SELL]
        
        ## 5. FUTURE PERSPECTIVE
        [12-month outlook with key catalysts and risks to watch]
        
        ## 6. EXECUTIVE SUMMARY
        [3-4 sentences capturing the investment thesis]
        
        CRITICAL: Always include specific grades (1-5) for each section and use the exact headers above.
        Base all assessments on quantitative data and qualitative factors present in the filing.
        """
    
    def analyze_with_embeddings(self, document_text: str) -> AnalysisResult:

        logger.info("Starting 10-K analysis...")
        
        logger.info("Chunking document...")
        chunks = self.chunk_document(document_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        logger.info("Creating embeddings...")
        embeddings = self.create_embeddings(chunks)

        logger.info("Creating focused analysis content...")
        analysis_content = self._create_focused_analysis_content(document_text)
        
        prompt = self.get_enhanced_prompt()

        prompt_tokens = self.count_tokens(prompt)
        content_tokens = self.count_tokens(analysis_content)
        total_input_tokens = prompt_tokens + content_tokens
        
        logger.info(f"Token usage - Prompt: {prompt_tokens}, Content: {content_tokens}, Total: {total_input_tokens}")
        
        if total_input_tokens > self.max_analysis_tokens:
            logger.error(f"Total tokens ({total_input_tokens}) exceed limit ({self.max_analysis_tokens})")
            raise ValueError(f"Content too large for analysis: {total_input_tokens} tokens")
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Please analyze this 10-K filing content:\n\n{analysis_content}"}
        ]

        logger.info("Sending to GPT for analysis...")
        try:
            response = self._make_api_request_with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.3  
                )
            )
            
            analysis_text = response.choices[0].message.content
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error during GPT analysis: {e}")
            analysis_text = f"Analysis failed due to API error: {str(e)}"

        result = self._parse_analysis_response(analysis_text)
        
        return result
    
    def _parse_analysis_response(self, analysis_text: str) -> AnalysisResult:
        logger.info("Parsing GPT analysis response...")
        logger.debug(f"Raw analysis text: {analysis_text[:500]}...")  

        risk_summary = "Risk analysis not available"
        risk_grade = 3
        financial_summary = "Financial analysis not available"
        financial_grade = 3
        mda_summary = "MD&A analysis not available"
        mda_grade = 3
        overall_summary = analysis_text if analysis_text else "Analysis summary not available"
        recommendation = "HOLD"
        future_perspective = "Future outlook not available"
        
        if not analysis_text or "Analysis failed" in analysis_text:
            logger.warning("Analysis text is empty or contains failure message")
            overall_summary = analysis_text if analysis_text else "Analysis failed - no response received"
        else:
            sections_dict = {}

            section_patterns = {
                'risk': r'##\s*1\.\s*RISK\s+FACTORS?\s+ANALYSIS',
                'financial': r'##\s*2\.\s*FINANCIAL\s+STATEMENTS?\s+ANALYSIS',
                'mda': r'##\s*3\.\s*MANAGEMENT\s+DISCUSSION\s*[&\s]*\s*ANALYSIS',
                'overall': r'##\s*4\.\s*OVERALL\s+ASSESSMENT',
                'future': r'##\s*5\.\s*FUTURE\s+PERSPECTIVE',
                'executive': r'##\s*6\.\s*EXECUTIVE\s+SUMMARY'
            }

            section_positions = []
            for section_name, pattern in section_patterns.items():
                matches = list(re.finditer(pattern, analysis_text, re.IGNORECASE))
                for match in matches:
                    section_positions.append((match.start(), section_name, match.end()))

            section_positions.sort()

            for i, (start_pos, section_name, header_end) in enumerate(section_positions):
                if i + 1 < len(section_positions):
                    end_pos = section_positions[i + 1][0]
                else:
                    end_pos = len(analysis_text)

                section_content = analysis_text[header_end:end_pos].strip()
                sections_dict[section_name] = section_content
                logger.debug(f"Extracted {section_name} section: {len(section_content)} chars")

            if 'risk' in sections_dict:
                risk_summary = sections_dict['risk']
            if 'financial' in sections_dict:
                financial_summary = sections_dict['financial']
            if 'mda' in sections_dict:
                mda_summary = sections_dict['mda']
            if 'executive' in sections_dict:
                overall_summary = sections_dict['executive']
            if 'future' in sections_dict:
                future_perspective = sections_dict['future']

            grade_patterns = [
                r'grade[:\s]*\*?\*?([1-5](?:\.[05])?)\s*/\s*5\*?\*?',  
                r'grade[:\s]*\*?\*?([1-5](?:\.[05])?)\*?\*?',  
                r'([1-5](?:\.[05])?)\s*/\s*5', 
                r'rating[:\s]*([1-5](?:\.[05])?)',  
                r'score[:\s]*([1-5](?:\.[05])?)',  
            ]

            for pattern in grade_patterns:
                matches = re.findall(pattern, risk_summary.lower())
                if matches:
                    try:
                        grade_value = float(matches[-1])
                        risk_grade = int(grade_value) if grade_value == int(grade_value) else grade_value
                        logger.debug(f"Found risk grade: {risk_grade} from pattern: {pattern}")
                        break
                    except (ValueError, IndexError):
                        pass

            for pattern in grade_patterns:
                matches = re.findall(pattern, financial_summary.lower())
                if matches:
                    try:
                        grade_value = float(matches[-1])
                        financial_grade = int(grade_value) if grade_value == int(grade_value) else grade_value
                        logger.debug(f"Found financial grade: {financial_grade} from pattern: {pattern}")
                        break
                    except (ValueError, IndexError):
                        pass

            for pattern in grade_patterns:
                matches = re.findall(pattern, mda_summary.lower())
                if matches:
                    try:
                        grade_value = float(matches[-1])
                        mda_grade = int(grade_value) if grade_value == int(grade_value) else grade_value
                        logger.debug(f"Found MDA grade: {mda_grade} from pattern: {pattern}")
                        break
                    except (ValueError, IndexError):
                        pass

            recommendation_text = sections_dict.get('overall', '') + ' ' + overall_summary
            recommendation_patterns = [
                r'recommendation[:\s]*(BUY|STRONG BUY|SELL|STRONG SELL|HOLD)',
                r'\b(BUY|STRONG BUY)\b',
                r'\b(SELL|STRONG SELL)\b', 
                r'\b(HOLD)\b'
            ]
            
            for pattern in recommendation_patterns:
                match = re.search(pattern, recommendation_text.upper())
                if match:
                    rec = match.group(1) if 'recommendation' in pattern else match.group(0)
                    if 'BUY' in rec:
                        recommendation = 'BUY'
                    elif 'SELL' in rec:
                        recommendation = 'SELL'
                    elif 'HOLD' in rec:
                        recommendation = 'HOLD'
                    logger.debug(f"Found recommendation: {recommendation}")
                    break

            def clean_section_content(content):
                lines = content.split('\n')
                cleaned_lines = []
                for line in lines:
                    if not re.match(r'##\s*\d+\.', line.strip()):
                        cleaned_lines.append(line)
                return '\n'.join(cleaned_lines).strip()
            
            risk_summary = clean_section_content(risk_summary)
            financial_summary = clean_section_content(financial_summary)
            mda_summary = clean_section_content(mda_summary)
            overall_summary = clean_section_content(overall_summary)
            future_perspective = clean_section_content(future_perspective)

        overall_grade = round((risk_grade * 0.25 + financial_grade * 0.50 + mda_grade * 0.25), 1)

        logger.info(f"Parsed grades - Risk: {risk_grade}, Financial: {financial_grade}, MD&A: {mda_grade}, Overall: {overall_grade}")
        logger.info(f"Recommendation: {recommendation}")
        logger.info(f"Section lengths - Risk: {len(risk_summary)}, Financial: {len(financial_summary)}, MD&A: {len(mda_summary)}")
        
        return AnalysisResult(
            risk_factors_summary=risk_summary,
            risk_factors_grade=risk_grade,
            financial_statements_summary=financial_summary,
            financial_statements_grade=financial_grade,
            mda_summary=mda_summary,
            mda_grade=mda_grade,
            overall_grade=overall_grade,
            overall_summary=overall_summary,
            recommendation=recommendation,
            future_perspective=future_perspective,
            timestamp=datetime.now().isoformat()
        )
    
    def save_results(self, result: AnalysisResult, filename: str = None):
        if filename is None:
            filename = f"10k_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        result_dict = {
            'risk_factors': {
                'summary': result.risk_factors_summary,
                'grade': result.risk_factors_grade
            },
            'financial_statements': {
                'summary': result.financial_statements_summary,
                'grade': result.financial_statements_grade
            },
            'management_discussion_analysis': {
                'summary': result.mda_summary,
                'grade': result.mda_grade
            },
            'overall_assessment': {
                'grade': result.overall_grade,
                'summary': result.overall_summary,
                'recommendation': result.recommendation,
                'future_perspective': result.future_perspective
            },
            'metadata': {
                'timestamp': result.timestamp,
                'model_used': self.model_name
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

def main():
    """Example usage of the 10-K analyzer"""

    analyzer = TenKAnalyzer(api_key=API_KEY)

    set_identity("YOUR_EMAIL")
    company = Company("INTC")

    filings = company.get_filings(form = "10-K").latest()
    sample_10k_text = filings.text()
    
    try:
        result = analyzer.analyze_with_embeddings(sample_10k_text)
        
        print("=== 10-K ANALYSIS RESULTS ===")
        print(f"Risk Factors Grade: {result.risk_factors_grade}/5")
        print(f"Financial Statements Grade: {result.financial_statements_grade}/5")
        print(f"MD&A Grade: {result.mda_grade}/5")
        print(f"Overall Grade: {result.overall_grade}/5")
        print(f"Recommendation: {result.recommendation}")
        print(f"Analysis completed at: {result.timestamp}")
        
        analyzer.save_results(result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")


def get_result_summary(result: AnalysisResult) -> str:
    """Generate a summary of the analysis results."""
    summary = (
        f"Risk Factors Grade: {result.risk_factors_grade}/5\n"
        f"Financial Statements Grade: {result.financial_statements_grade}/5\n"
        f"MD&A Grade: {result.mda_grade}/5\n"
        f"Overall Grade: {result.overall_grade}/5\n"
        f"Recommendation: {result.recommendation}\n"
        f"Analysis completed at: {result.timestamp}"
    )
    return summary


def form_type(ticker, form="10-K"):
    """Fetch the latest 10-K or 10-Q filing text for a given ticker."""
    set_identity("YOUR_EMAIL")
    company = Company(f"{ticker}")

    if form == "10-K":
        filings = company.get_filings(form="10-K").latest()
        company_10k = filings.text()
        analyzer = TenKAnalyzer(api_key=API_KEY)
        result = analyzer.analyze_with_embeddings(company_10k)
        analyzer.save_results(result)
        final_10k_rating = get_result_summary(result)
        return final_10k_rating

    elif form == "10-Q":
        filings = company.get_filings(form="10-Q").latest()
        company_10q = filings.text()
        analyzer = TenKAnalyzer(api_key=API_KEY)
        result = analyzer.analyze_with_embeddings(company_10q)
        analyzer.save_results(result)
        final_10q_rating = get_result_summary(result)
        return final_10q_rating

    else:
        raise ValueError("Invalid form type. Please use '10-K' or '10-Q'.")


if __name__ == "__main__":
    main()