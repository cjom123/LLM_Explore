# Excel Analysis with RAG and Prompt Engineering

A powerful Excel analysis tool that combines Retrieval-Augmented Generation (RAG) with advanced prompt engineering using open-source models from Hugging Face.

## Features

- **Intelligent Excel Analysis**: Automatically understand and analyze Excel files using AI
- **RAG Implementation**: Retrieve relevant context from Excel data for better analysis
- **Prompt Engineering**: Optimized prompts for various analysis tasks
- **Open Source Models**: Uses models from Hugging Face (no API costs)
- **Interactive Web Interface**: Streamlit-based UI for easy interaction
- **Multiple Analysis Types**: Data insights, trend analysis, anomaly detection, and more

## Architecture

The project uses a RAG pipeline with:
1. **Document Loading**: Excel file parsing and text extraction
2. **Text Chunking**: Intelligent splitting of Excel content
3. **Embedding Generation**: Using sentence-transformers for vector representations
4. **Vector Storage**: FAISS for efficient similarity search
5. **Retrieval**: Context-aware information retrieval
6. **Generation**: AI-powered analysis using open-source language models

## Models Used

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (lightweight, fast)
- **Language Model**: `microsoft/DialoGPT-medium` (for text generation)
- **Alternative Models**: Configurable to use other Hugging Face models

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd excel-rag-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Upload Excel File**: Use the web interface to upload your Excel file
2. **Select Analysis Type**: Choose from predefined analysis categories
3. **Ask Questions**: Interact with your data using natural language
4. **Get Insights**: Receive AI-generated analysis and visualizations

## Project Structure

```
├── app.py                          # Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── excel_processor.py         # Excel file processing
│   ├── rag_engine.py              # RAG implementation
│   ├── prompt_engineer.py         # Prompt optimization
│   └── model_manager.py           # Model loading and management
├── utils/
│   ├── __init__.py
│   ├── text_processing.py         # Text preprocessing utilities
│   └── visualization.py           # Chart and plot generation
├── prompts/
│   ├── analysis_prompts.py        # Predefined analysis prompts
│   └── system_prompts.py          # System-level prompts
├── config/
│   └── settings.py                # Configuration settings
├── examples/
│   └── sample_data.xlsx           # Sample Excel file for testing
└── requirements.txt                # Python dependencies
```

## Configuration

Edit `config/settings.py` to customize:
- Model selection
- Chunk sizes
- Embedding dimensions
- Analysis parameters

## Examples

The project includes sample prompts for:
- Data summary and insights
- Trend analysis
- Anomaly detection
- Statistical analysis
- Business intelligence queries

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is open source and available under the MIT License. 