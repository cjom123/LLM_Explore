"""
Excel file processing module for the RAG system.
Handles loading, parsing, and text extraction from various Excel formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from config.settings import SUPPORTED_FORMATS, MAX_FILE_SIZE, SHEET_NAMES_TO_SKIP

logger = logging.getLogger(__name__)

class ExcelProcessor:
    """Handles Excel file processing and text extraction for RAG analysis."""
    
    def __init__(self):
        self.supported_formats = SUPPORTED_FORMATS
        self.max_file_size = MAX_FILE_SIZE
        self.sheet_names_to_skip = SHEET_NAMES_TO_SKIP
        
    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate Excel file format and size."""
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                return False, f"File size exceeds maximum limit of {self.max_file_size / (1024*1024):.1f}MB"
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                return False, f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
            
            return True, "File is valid"
            
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    def load_excel_file(self, file_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """Load Excel file and return dictionary of sheet names and DataFrames."""
        try:
            file_path = Path(file_path)
            
            # Load all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                if sheet_name not in self.sheet_names_to_skip:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        if not df.empty:
                            sheets[sheet_name] = df
                            logger.info(f"Loaded sheet: {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
                    except Exception as e:
                        logger.warning(f"Failed to load sheet {sheet_name}: {str(e)}")
                        continue
            
            return sheets
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise
    
    def extract_text_from_dataframe(self, df: pd.DataFrame, sheet_name: str = "") -> str:
        """Extract meaningful text from DataFrame for RAG processing."""
        try:
            text_parts = []
            
            # Add sheet name if provided
            if sheet_name:
                text_parts.append(f"Sheet: {sheet_name}")
            
            # Add column information
            columns_info = f"Columns: {', '.join(df.columns.tolist())}"
            text_parts.append(columns_info)
            
            # Add data type information
            dtype_info = f"Data types: {dict(df.dtypes)}"
            text_parts.append(dtype_info)
            
            # Add basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats = df[numeric_cols].describe()
                stats_text = f"Numerical statistics:\n{stats.to_string()}"
                text_parts.append(stats_text)
            
            # Add sample data (first few rows)
            sample_size = min(5, len(df))
            sample_data = df.head(sample_size).to_string()
            sample_text = f"Sample data (first {sample_size} rows):\n{sample_data}"
            text_parts.append(sample_text)
            
            # Add row count
            row_count = f"Total rows: {len(df)}"
            text_parts.append(row_count)
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from DataFrame: {str(e)}")
            return f"Error processing sheet {sheet_name}: {str(e)}"
    
    def get_dataframe_summary(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive summary of DataFrame for analysis."""
        try:
            summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "duplicates": df.duplicated().sum()
            }
            
            # Add numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
            
            # Add categorical column information
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                summary["categorical_info"] = {
                    col: {
                        "unique_count": df[col].nunique(),
                        "top_values": df[col].value_counts().head(5).to_dict()
                    }
                    for col in categorical_cols
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating DataFrame summary: {str(e)}")
            return {"error": str(e)}
    
    def process_excel_file(self, file_path: Union[str, Path]) -> Dict:
        """Main method to process Excel file and return structured data for RAG."""
        try:
            # Validate file
            is_valid, message = self.validate_file(file_path)
            if not is_valid:
                return {"error": message}
            
            # Load Excel file
            sheets = self.load_excel_file(file_path)
            
            if not sheets:
                return {"error": "No valid sheets found in Excel file"}
            
            # Process each sheet
            processed_data = {
                "file_path": str(file_path),
                "total_sheets": len(sheets),
                "sheets": {},
                "overall_summary": {}
            }
            
            all_text_chunks = []
            
            for sheet_name, df in sheets.items():
                # Extract text for RAG
                sheet_text = self.extract_text_from_dataframe(df, sheet_name)
                all_text_chunks.append(sheet_text)
                
                # Get summary
                sheet_summary = self.get_dataframe_summary(df)
                
                processed_data["sheets"][sheet_name] = {
                    "dataframe": df,
                    "text": sheet_text,
                    "summary": sheet_summary
                }
            
            # Combine all text for global analysis
            processed_data["combined_text"] = "\n\n---\n\n".join(all_text_chunks)
            
            # Generate overall summary
            total_rows = sum(len(sheet["dataframe"]) for sheet in processed_data["sheets"].values())
            total_cols = sum(len(sheet["dataframe"].columns) for sheet in processed_data["sheets"].values())
            
            processed_data["overall_summary"] = {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "sheet_names": list(sheets.keys())
            }
            
            logger.info(f"Successfully processed Excel file with {len(sheets)} sheets")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            return {"error": f"Failed to process Excel file: {str(e)}"}
    
    def export_processed_data(self, processed_data: Dict, output_path: Union[str, Path]) -> bool:
        """Export processed data to JSON for later use."""
        try:
            import json
            
            # Remove DataFrame objects (not JSON serializable)
            export_data = processed_data.copy()
            for sheet_name in export_data.get("sheets", {}):
                if "dataframe" in export_data["sheets"][sheet_name]:
                    del export_data["sheets"][sheet_name]["dataframe"]
            
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported processed data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting processed data: {str(e)}")
            return False 