# CleanVision

Upload a messy CSV or Excel file and CleanVision will find the problems, clean the data and explain the results in plain language.

Canva Presentation Link: https://canva.link/7tgffjfd15t1e1b

## Problem and solution

Real-world data is rarely clean. Duplicate rows, empty cells, stray whitespace, wrong data types and outliers can take hours of manual work before any analysis can start.

CleanVision automates this work. A local open-source LLM (Ollama, `llama3.2:3b`) reads the data and writes **a JSON plan, not code**. Python executes the plan and computes every number, and the LLM only constructs the sentences that explain the result. This way the model cannot make up figures, and your data never leaves your computer.

## How it works

1. **Load:** `load_dataframe` turns the file into a pandas DataFrame.
2. **Profile:** `profile_dataframe` collects statistics on rows, columns, null cells, duplicates and outliers.
3. **Plan:** `build_plan` gives the profile to the LLM, which returns a JSON cleaning plan. If the LLM is unavailable, a rule-based fallback plan is used.
4. **Execute:** `execute_plan` runs the plan steps and writes a log entry for each step.
5. **Re-profile:** the profile is calculated again for the cleaned data.
6. **Charts:** `generate_charts` creates up to 3 Plotly charts.
7. **Insights:** `write_insights` explains the result in plain language, and the response is returned as JSON.

## Technologies

| Area | Technology |
|---|---|
| Language | Python |
| Data | pandas, numpy, scikit-learn, scipy, openpyxl |
| Charts | Plotly |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| LLM | Ollama (`llama3.2:3b`) |
| Testing | pytest |

## Installation

1. Clone the repository:
```bash
   git clone <repo-url>
   cd cleanvision
```
2. Create a virtual environment:
```bash
   python -m venv .venv
   source .venv/bin/activate
```
   On Windows: `.venv\Scripts\activate`
3. Install the dependencies:
```bash
   pip install -r requirements.txt
```
4. Install [Ollama](https://ollama.com) and pull the model:
```bash
   ollama pull llama3.2:3b
   ollama serve
```
5. Start the project:
   - Linux/macOS: `chmod +x run.sh && ./run.sh`
   - Windows: `run.bat`

The backend runs at `http://localhost:8000` and the interface at `http://localhost:8501`. If Ollama is not running, the script prints a warning and the app works with the fallback plan.


## Limitations and future plans

**Limitations**
- The file size limit is 20 MB.
- `llama3.2:3b` is a small model, so plan quality may drop on complex tables. In that case the fallback plan is used.
- The old `.xls` format needs an extra `pip install xlrd`.
- Results are stored only in the `jobs/` folder and are not cleaned up automatically.
- Jobs run synchronously, so large files may cause waiting.

**Future plans**
- Option to choose stronger models.
- Job queue and progress indicator.
- Letting users edit the cleaning plan by hand.
- More file formats (JSON, Parquet).
- Automatic deletion of old job folders.

## License

MIT License. See the `LICENSE` file for details.
