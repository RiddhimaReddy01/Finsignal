# FinSignal: Financial Analysis Tool

FinSignal is an advanced, AI-driven financial analysis and orchestration tool designed to process, analyze, and deliver comprehensive insights on market data, financial transcripts, and SEC filings.

## Architecture & Structure

The codebase has recently been refactored to align with best-practice coding standards, emphasizing modularity and clean separation of concerns. The directory structure is as follows:

- **`src/`**: Contains the core application logic, including the FastAPI backend server (`server.py`), the decision engine, orchestrators, and AI-driven signal pipelines.
- **`tests/`**: Contains all unit, integration, and end-to-end testing suites. Run these via `pytest`.
- **`scripts/`**: Houses utility, debugging, and index-building scripts used for maintenance and data cache warming.
- **`docs/`**: Markdown documentation detailing specific features, such as the Decision Tab, the dynamic weighting implementation, and evidence quality logic.

## Core Capabilities

1. **Market Data Integration**: Connects dynamically with Yahoo Finance and other APIs for real-time data.
2. **Document Analysis**: Parses and scores SEC item texts (e.g., Item 1A risk factors) and management discussion/analysis transcripts.
3. **Valuation Engine**: Implements Quantitative DCF models, extracting free cash flow and growth rates to perform robust intrinsic value calculations.
4. **LLM Orchestration**: Utilizes the Gemini models to analyze tone, reconcile market data conflicts, and build fully automated, structured financial research reports.

## Installation

Ensure you have Python installed. It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

You must configure your `.env` file with the appropriate API keys (e.g., `GEMINI_API_KEY`) to leverage the language model capabilities.

## Usage

Start the backend API server:
```bash
uvicorn src.server:app --reload
```
Once started, the API docs are accessible at `http://127.0.0.1:8000/docs`.

To run the orchestration CLI:
```bash
python src/main.py -q "What is the valuation for AAPL?"
```

## Testing

Run the automated test suite to verify end-to-end functionality and model consistency:
```bash
pytest tests/
```
