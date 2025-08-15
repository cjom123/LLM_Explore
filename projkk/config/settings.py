"""
Configuration settings for the Excel RAG Analysis project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LANGUAGE_MODEL = "microsoft/DialoGPT-medium"
ALTERNATIVE_MODELS = {
    "small": "microsoft/DialoGPT-small",
    "medium": "microsoft/DialoGPT-medium",
    "large": "microsoft/DialoGPT-large",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

# RAG parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_K = 5
TOP_P = 0.9

# Vector database settings
VECTOR_DIMENSION = 384  # For all-MiniLM-L6-v2
SIMILARITY_METRIC = "cosine"
INDEX_TYPE = "Flat"  # or "IVFFlat" for larger datasets

# Excel processing settings
SUPPORTED_FORMATS = ['.xlsx', '.xls', '.csv']
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SHEET_NAMES_TO_SKIP = ['Sheet1', 'Sheet2', 'Sheet3']  # Generic sheet names to skip

# Prompt engineering settings
MAX_PROMPT_LENGTH = 2000
SYSTEM_PROMPT_TEMPLATE = """
You are an expert data analyst specializing in Excel data analysis. 
Your task is to provide clear, actionable insights based on the data provided.
Always support your analysis with specific data points and suggest visualizations when appropriate.
"""

# Analysis categories
ANALYSIS_CATEGORIES = {
    "summary": "Data overview and key statistics",
    "trends": "Time series and pattern analysis",
    "anomalies": "Outlier and anomaly detection",
    "correlations": "Relationship analysis between variables",
    "insights": "Business intelligence and actionable insights",
    "forecasting": "Predictive analysis and trends"
}

# Visualization settings
CHART_TYPES = {
    "line": "Line charts for trends",
    "bar": "Bar charts for comparisons",
    "scatter": "Scatter plots for correlations",
    "histogram": "Histograms for distributions",
    "heatmap": "Heatmaps for correlation matrices",
    "box": "Box plots for outlier detection"
}

# Performance settings
BATCH_SIZE = 32
USE_GPU = False  # Set to True if CUDA is available
CACHE_DIR = PROJECT_ROOT / ".cache"
MODEL_CACHE_DIR = CACHE_DIR / "models"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Create necessary directories
CACHE_DIR.mkdir(exist_ok=True)
MODEL_CACHE_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR.mkdir(exist_ok=True) 