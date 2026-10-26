# 🧠 DataMind — LLM-Based Data Analysis System

A conversational data analysis system powered by Large Language Models (LLMs) and LangChain. Ask questions about your database in natural language and get SQL queries, visualizations, and predictive analytics — automatically.

> Master's thesis project — Applied Computer Science, Hochschule Fulda  

---

## Features

- **Natural Language to SQL** — Ask questions in plain English or German; the agent generates and executes SQL automatically
- **Automatic Visualizations** — Charts and diagrams generated on demand via Matplotlib
- **Predictive Analytics** — Automatic model selection from 10 forecasting models (ARIMA, SARIMA, Prophet, XGBoost, Random Forest, and more)
- **Multi-Database Support** — Connect and query multiple PostgreSQL databases simultaneously
- **Hallucination Detection** — Numeric claims in agent responses are automatically verified against the database
- **Query History** — All queries are logged with response times and ratings; exportable as CSV or Excel
- **Dark / Light Mode** — Full theme support
- **Multilingual UI** — English and German interface

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (claude-sonnet-4-6) / OpenAI GPT-4o |
| Agent Framework | LangChain (`create_tool_calling_agent`) |
| Database | PostgreSQL + psycopg2 + SQLAlchemy |
| Frontend | Streamlit |
| Forecasting | ARIMA, SARIMA, Prophet, XGBoost, Random Forest, Scikit-learn |
| Language | Python 3.11 |

---

## Project Structure

```
llm_data_analysis/
├── app/
│   └── main.py                  # Streamlit UI
├── agent/
│   └── analyst_agent.py         # LangChain Tool-Calling Agent
├── database/
│   ├── db_connector.py          # Single DB connector
│   └── multi_db_connector.py    # Multi-DB connector
├── tools/
│   ├── sql_tool.py              # SQL execution tool (single DB)
│   ├── multi_db_sql_tool.py     # SQL execution tool (multi-DB)
│   ├── python_tool.py           # Python code execution tool
│   └── predictive_tool.py       # Forecasting & predictive analytics tool
├── utils/
│   ├── prompt_templates.py      # System prompts
│   ├── hallucination_detector.py
│   ├── query_history.py
│   ├── visualizer.py            # Chart generation
│   ├── logger.py                # Logging
│   └── file_uploader.py
├── tests/
│   ├── evaluation.py            # Agent evaluation suite
│   ├── prompt_testing.py        # Prompt quality tests
│   └── tests_db.py              # Database connection tests
├── logs/                        # Query history & agent logs
├── evaluation_results/          # Evaluation CSV/Excel exports
├── .env.example                 # Environment variable template
├── .gitignore
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/doriane-koudjou/LLM-Data-Analysis-System.git
cd LLM-Data-Analysis-System
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env

ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

### 5. Start the application

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Connect a database** — Enter your PostgreSQL credentials in the sidebar and click *Connect*
2. **Ask a question** — Type a question in the chat input or click one of the example questions
3. **View results** — The agent returns an answer, the generated SQL query, and optionally a chart or forecast table
4. **Export** — Download results as CSV or export the full query history

---

## Supported Forecasting Models

| # | Model |
|---|---|
| 1 | Linear Regression |
| 2 | Polynomial Regression |
| 3 | Moving Average |
| 4 | Exponential Smoothing |
| 5 | ARIMA |
| 6 | SARIMA |
| 7 | Ensemble |
| 8 | Random Forest |
| 9 | XGBoost |
| 10 | Prophet |

The system automatically selects the most suitable model based on the statistical properties of the time series (seasonality, trend, number of data points).

---

## License

This project was developed as part of a Master's thesis at Hochschule Fulda. For academic use only.
