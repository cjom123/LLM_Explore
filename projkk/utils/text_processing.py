"""
Text processing utilities for Excel RAG analysis.
Handles text cleaning, preprocessing, and normalization.
"""

import re
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles text processing and cleaning for RAG analysis."""
    
    def __init__(self):
        self.common_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing."""
        try:
            if not isinstance(text, str):
                return str(text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep important ones
            text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\{\}]', '', text)
            
            # Clean up multiple punctuation
            text = re.sub(r'[\.\,\:\;]+', '.', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return str(text)
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text for better analysis."""
        try:
            # Clean text
            clean_text = self.clean_text(text)
            
            # Split into words
            words = re.findall(r'\b\w+\b', clean_text)
            
            # Remove stopwords and short words
            keywords = [
                word for word in words 
                if word.lower() not in self.common_stopwords 
                and len(word) > 2
            ]
            
            # Count frequency
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, freq in sorted_keywords[:max_keywords]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_numerical_values(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        try:
            # Find all numbers (including decimals)
            numbers = re.findall(r'\b\d+\.?\d*\b', text)
            
            # Convert to float
            numerical_values = []
            for num_str in numbers:
                try:
                    numerical_values.append(float(num_str))
                except ValueError:
                    continue
            
            return numerical_values
            
        except Exception as e:
            logger.error(f"Error extracting numerical values: {str(e)}")
            return []
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract date patterns from text."""
        try:
            # Common date patterns
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or M/D/YY
                r'\b\d{4}-\d{1,2}-\d{1,2}\b',    # YYYY-MM-DD
                r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
                r'\b\w+ \d{1,2},? \d{4}\b',      # Month DD, YYYY
                r'\b\d{1,2} \w+ \d{4}\b'         # DD Month YYYY
            ]
            
            dates = []
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                dates.extend(matches)
            
            return list(set(dates))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting dates: {str(e)}")
            return []
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using Jaccard similarity."""
        try:
            # Clean and tokenize texts
            words1 = set(self.clean_text(text1).split())
            words2 = set(self.clean_text(text2).split())
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Create a summary of text by extracting key sentences."""
        try:
            if len(text) <= max_length:
                return text
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Score sentences by word frequency
            word_freq = {}
            for sentence in sentences:
                words = re.findall(r'\b\w+\b', sentence.lower())
                for word in words:
                    if word not in self.common_stopwords:
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences
            sentence_scores = {}
            for sentence in sentences:
                score = 0
                words = re.findall(r'\b\w+\b', sentence.lower())
                for word in words:
                    if word in word_freq:
                        score += word_freq[word]
                sentence_scores[sentence] = score
            
            # Select top sentences
            sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            
            summary = ""
            for sentence, score in sorted_sentences:
                if len(summary + sentence) <= max_length:
                    summary += sentence + ". "
                else:
                    break
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data patterns from text."""
        try:
            structured_data = {
                "keywords": self.extract_keywords(text),
                "numbers": self.extract_numerical_values(text),
                "dates": self.extract_dates(text),
                "word_count": len(text.split()),
                "character_count": len(text),
                "has_numbers": bool(self.extract_numerical_values(text)),
                "has_dates": bool(self.extract_dates(text))
            }
            
            # Detect data types
            if any(char.isdigit() for char in text):
                structured_data["contains_digits"] = True
            
            if any(char.isupper() for char in text):
                structured_data["contains_uppercase"] = True
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}")
            return {}
    
    def normalize_column_names(self, column_names: List[str]) -> List[str]:
        """Normalize Excel column names for better processing."""
        try:
            normalized_names = []
            
            for name in column_names:
                # Convert to string
                name_str = str(name)
                
                # Remove special characters
                clean_name = re.sub(r'[^\w\s]', '', name_str)
                
                # Replace spaces with underscores
                clean_name = re.sub(r'\s+', '_', clean_name)
                
                # Convert to lowercase
                clean_name = clean_name.lower()
                
                # Ensure it's not empty
                if clean_name:
                    normalized_names.append(clean_name)
                else:
                    normalized_names.append(f"column_{len(normalized_names)}")
            
            return normalized_names
            
        except Exception as e:
            logger.error(f"Error normalizing column names: {str(e)}")
            return column_names
    
    def detect_data_patterns(self, text: str) -> Dict[str, Any]:
        """Detect patterns in text that might indicate data structure."""
        try:
            patterns = {
                "is_table": False,
                "has_headers": False,
                "column_count": 0,
                "row_count": 0,
                "separator": None,
                "data_types": []
            }
            
            lines = text.split('\n')
            
            if len(lines) > 1:
                # Check if it looks like a table
                first_line = lines[0].strip()
                second_line = lines[1].strip()
                
                # Detect separators
                separators = [',', '\t', '|', ';']
                for sep in separators:
                    if sep in first_line and sep in second_line:
                        patterns["separator"] = sep
                        patterns["is_table"] = True
                        patterns["column_count"] = len(first_line.split(sep))
                        patterns["row_count"] = len([l for l in lines if l.strip()])
                        break
                
                # Check for headers (first line different from others)
                if patterns["is_table"]:
                    first_cols = first_line.split(patterns["separator"])
                    second_cols = second_line.split(patterns["separator"])
                    
                    # If first line has text and second has numbers, likely headers
                    if (any(not col.replace('.', '').isdigit() for col in first_cols) and
                        any(col.replace('.', '').isdigit() for col in second_cols)):
                        patterns["has_headers"] = True
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting data patterns: {str(e)}")
            return {"is_table": False, "has_headers": False, "column_count": 0, "row_count": 0} 