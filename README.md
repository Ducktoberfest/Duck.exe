# CleanVision

Automated data cleaning, profiling, and visual insight generation powered by local AI and deterministic execution.

## What the Project Is

Dirty CSV and Excel spreadsheets often require repetitive, manual cleaning before they can be used for reporting or analytics. CleanVision ingests messy tabular datasets, profiles their quality, and prompts a local LLM to produce a structured JSON cleaning plan. Python deterministically executes the plan, computing all numerical metrics and safeguards so the LLM never writes or executes arbitrary code. The system exports clean data alongside interactive charts and plain-language explanations, ensuring full privacy with all computation kept on the local machine.

## How It Works

1. Ingest raw CSV or Excel file and parse delimiters and encodings.
2. Profile baseline data quality (types, nulls, duplicates, outliers, whitespace).
3. Build a structured JSON cleaning plan using local Ollama LLM (or rule-based fallback).
4. Execute cleaning actions deterministically in Python with step-by-step audit logging.
5. Profile cleaned data, evaluate deltas, and export cleaned CSV with UTF-8 BOM encoding.
6. Generate responsive Plotly visualizations for distributions, categories, and trends.
7. Synthesize plain-language insights in English or Azerbaijani based strictly on verified numbers.

## Tools

| Category | Technology |
| --- | --- |
| Core Language | Python |
| Data Processing | pandas, numpy, scikit-learn, scipy, openpyxl |
| Visualizations | Plotly |
| Backend & Server | FastAPI, Uvicorn |
| Frontend | Tailwind CSS, vanilla JS |
| Local AI | Ollama with llama3.2:3b |
| Testing | pytest |

## How to Open

Run `run.bat` (Windows) or `./run.sh` (Linux/macOS), then open http://localhost:8000 in your browser; API documentation is available at http://localhost:8000/docs. Without Ollama installed or running, the application seamlessly operates in rule-based fallback mode.

## Team Roles

- Lead Fullstack Engineer: Alex Mercer
- Backend & Pipeline Engineer: Jordan Vance
- AI & Prompt Engineer: Taylor Morgan
- Frontend & UI Designer: Morgan Riley

## License

MIT
