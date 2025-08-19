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



















### Ingestion & Processing
- **Upload Excel**: Input source (.xlsx/.xls/.csv) provided by user.
- `core/excel_processor.ExcelProcessor`
  - **Purpose**: Validate, load, and convert Excel sheets to analyzable text and summaries.
  - **Functioning**: Uses pandas to read sheets, cleans empty rows/cols, extracts schema, dtypes, samples, and numeric stats.

- **Text Chunking**
  - **Purpose**: Split long text into manageable, retrievable units.
  - **Functioning**: Creates overlapping chunks using `CHUNK_SIZE` (1000) and `CHUNK_OVERLAP` (200) to preserve context.

### Indexing
- `sentence-transformers/all-MiniLM-L6-v2` (Embedding Model)
  - **Purpose**: Convert chunks and queries into 384-d semantic vectors.
  - **Functioning**: Batch encodes text via `SentenceTransformer.encode`; device-aware (CPU/GPU/MPS).

- **FAISS Vector DB**
  - **Purpose**: Store vectors and enable fast similarity search.
  - **Functioning**: Builds index (`IndexFlatIP` exact or `IndexIVFFlat` approximate). Uses inner product for cosine similarity.

- **Chunk Metadata**
  - **Purpose**: Track provenance and structure of chunks.
  - **Functioning**: Stores `sheet_name`, `chunk_index`, `total_chunks`, `source`.

- **Save/Load Index**
  - **Purpose**: Persist and restore retrieval state.
  - **Functioning**: Writes `.faiss` binary and `.json` metadata; can reload to avoid reprocessing.

### Query & Retrieval
- **Query Embedding**
  - **Purpose**: Represent the user’s question as a vector in the same space as chunks.
  - **Functioning**: Encoded by the same embedding model; ensures comparable similarity.

- **Top-K Retrieval**
  - **Purpose**: Find most relevant context for generation.
  - **Functioning**: FAISS search returns top `K=TOP_K (5)` chunks with similarity scores.

- **Retrieved Chunks + Scores**
  - **Purpose**: Provide grounded context for the LLM.
  - **Functioning**: Aggregates the best-matching chunk texts and their metadata for the prompt.

### Prompting & Generation
- `core/prompt_engineer.PromptEngineer`
  - **Purpose**: Craft effective prompts for different analysis tasks.
  - **Functioning**: Selects templates (summary/trends/etc.), inserts context, optimizes phrasing per model.

- `core/model_manager.ModelManager`
  - **Purpose**: Load/cache models and run inference.
  - **Functioning**: Device selection, 4-bit quant (CUDA), tokenizer setup, embedding and text generation APIs.

- `microsoft/DialoGPT-medium` (LLM)
  - **Purpose**: Generate natural-language, context-aware analysis.
  - **Functioning**: Receives enhanced prompt and produces the response; alternatives: small/large.

- **Enhanced Prompt**
  - **Purpose**: Combine system role, retrieved context, and user query.
  - **Functioning**: Structured prompt guiding the LLM to produce grounded, actionable insights.

- **RAG Response**
  - **Purpose**: Final answer grounded in retrieved Excel content.
  - **Functioning**: Returns generated text plus context used and lengths for traceability.

- `utils/visualization.DataVisualizer`
  - **Purpose**: Turn insights into charts (line/bar/scatter/heatmap/box).
  - **Functioning**: Generates Matplotlib/Plotly figures and dashboards; can export images/HTML.

### Configuration & Storage
- `config/settings.py`
  - **Purpose**: Central control of models, chunking, retrieval, and performance.
  - **Functioning**: Sets `EMBEDDING_MODEL`, `LANGUAGE_MODEL`, `CHUNK_SIZE`, `TOP_K`, `INDEX_TYPE`, device flags, cache paths.

- **Model Cache (`.cache/models`)**
  - **Purpose**: Avoid re-downloading Hugging Face models.
  - **Functioning**: Local storage used by `SentenceTransformer` and `transformers`.

- **Artifacts**
  - **Purpose**: Persisted outputs for reuse.
  - **Functioning**: `.faiss` index + `metadata.json` for retrieval; exported figures for reporting.

- **Files/Classes Involved**
  - `core/excel_processor.py`, `core/rag_engine.py`, `core/model_manager.py`, `core/prompt_engineer.py`
  - `utils/visualization.py`, `config/settings.py`

- RAG flow summary:
  - Excel → Text → Chunking → Embeddings → FAISS Index → Query Embedding → Top-K Context → Prompt Engineering → DialoGPT Response → Visualizations
