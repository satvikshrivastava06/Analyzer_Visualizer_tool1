# ⬡ Data Analyzer & Visualizer Tool

<div align="center">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourapp.streamlit.app)
[![CI](https://github.com/satvikshrivastava06/Analyzer_Visualizer_tool1/actions/workflows/ci.yml/badge.svg)](https://github.com/satvikshrivastava06/Analyzer_Visualizer_tool1/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![DuckDB](https://img.shields.io/badge/DuckDB-1.4.4-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)

**A premium, AI-powered, full-stack data analytics platform.**

*Ingest → Profile → Clean → Transform → Visualize → Report — all in one browser tab.*

</div>

---

## 🧠 What Is This?

The **Data Analyzer & Visualizer Tool** is a production-grade, modular Streamlit application that replaces the workflows of multiple expensive enterprise tools — Tableau, Power BI, Alteryx, and Microsoft Copilot — with a single, open-source, zero-cost platform.

It combines the speed of **DuckDB** (an in-process OLAP database), the intelligence of **multi-provider LLMs** (Gemini, Groq), and a comprehensive suite of visualization engines (Plotly, Altair, ECharts, PyGWalker) into a cohesive, production-hardened analytics environment.

---

## ✨ Feature Overview

### 📥 1. Ingestion Lab
- Upload **CSV, Excel, JSON, Parquet** files directly
- Pull live data from **Google Sheets** via URL
- Execute custom **SQL queries** against existing DuckDB tables
- Connect to **MotherDuck** (cloud DuckDB) for team-shared datasets
- AI-powered **Dataset Hub** auto-discovers and indexes local CSV/ZIP datasets

### 🏥 2. Smart Clean
- Auto-generate **statistical profile** across all data types (`describe(include='all')`)
- **Missing field heatmap** — identify null columns at a glance
- **Smart Imputation** — median fill for numeric, mode fill for categorical
- **PII Anonymizer** — detects and redacts email, phone, SSN, IP columns before they reach any AI or chart
- **Duplicate removal, NA dropping** with one click
- **Z-score anomaly scanner** — flags statistical outliers (|z| > 3)
- **Sweetviz deep profiling** — generates a scrollable interactive HTML report
- **Data Quality Audit** — checks for negative-numeric logical violations, malformed emails, and schema inconsistencies

### 📊 3. Visualizations
- Comprehensive **manual chart builder** (Bar, Line, Area, Scatter, Histogram, Box, Violin, Pie, Treemap, Sunburst, Heatmap, Bubble, Strip)
- **Statistical context panel** anchored to every chart (Min / Max / Mean / STD)
- **Smart sampling** — auto-truncates datasets > 10,000 rows to preserve UI performance
- **PyGWalker BI Explorer** — full drag-and-drop Tableau-style canvas with **Geo Map Mode** (world map plotting for lat/lon datasets)
- **ECharts Premium** — animated Gauge, Radar, Sankey, Candlestick, 3D Bar, and Heatmap charts
- **Vega-Altair** declarative charts with grammar-of-graphics precision
- **Generative Viz** — describe a chart in plain English; the AI generates it

### 🤖 4. AI Assistant (Agentic)
- Full **conversational chat** interface over your loaded dataset
- **Perform Live Surgery** — AI recommendations come with executable SQL blocks that mutate the live DuckDB view in place (no code required)
- **AI Data Dictionary** — auto-generate column meanings and strategic KPI suggestions
- **Semantic Search** — find data by *concept* ("high revenue customers") rather than exact keywords
- **Text-to-SQL** using a WrenAI-inspired semantic context manifest for deterministic, hallucination-resistant queries
- **Multi-model fallback**: Gemini 2.5-flash → Gemini 2.0-flash → Groq/Llama 3.3 70B → Groq/Mixtral — zero-cost, 100% uptime

### 🔄 5. ETL / Pipelines
- Visual **multi-step SQL pipeline builder** powered by DuckDB
- Add, reorder, and delete sequential transformation steps
- Preview output at each step before committing
- dbt-style **modular SQL modeling** — write reusable `CREATE VIEW` statements
- Full pipeline **execution log** with row counts at each stage

### 📈 6. Forecasting Lab
- Train **Polynomial Regression** models on any numeric time series
- Configure degree and train/test split via sliders
- Overlay **AI forecasts vs. expert baselines** (e.g., UN population projections)
- Export model predictions as CSV

### 📰 7. Journalist Lab
- Apply **FiveThirtyEight-style aesthetics** — authoritative typography, subdued palettes, and precise grid hierarchies
- **Rolling average smoothing** for noisy polling or market data
- Generate publication-ready narrative charts

### ⚖️ 8. Compare
- **Delta analysis** between two datasets — value distributions, row counts, column overlap
- Identify schema drift and data pipeline regressions

### 📋 9. Executive Summary
- High-level **KPI dashboard** styled for leadership teams
- Auto-detects `datetime64` columns to compute **% vs. Previous Period** deltas
- Separates operational cleaning UI from strategic business reporting (IJCDS Best Practice 9)

### 📤 10. Export Options
- Download cleaned data as **CSV, Excel, JSON, or Parquet**
- Export a **reproducible Python script** — a standalone `.py` file containing the exact DuckDB connection, cleaning logic, and session state
- Auto-generate a session-matched `requirements.txt`
- One-click **PDF report** export via FPDF2

### 🔒 Governance & Lineage (Sidebar)
- Real-time **connection status** (local DuckDB / MotherDuck)
- Active source table, schema preview, and cache limits
- **Git-Lite Version History** — immutable audit log of every transformation, query, and AI surgery with timestamps

---

## 🏗️ Architecture

```
data-analyzer-visualizer-tool/
│
├── app.py                      # Main entry point, session state, sidebar, tab routing
├── ui_theme.py                 # Design system — CSS tokens, helpers, layout components
├── pyg_viz.py                  # PyGWalker BI Explorer with Geo Map Mode
├── premium_viz.py              # ECharts premium chart library
├── visualization.py            # Legacy Plotly chart builders
│
├── modules/                    # Pure business logic (no Streamlit)
│   ├── ai_engine.py            # Multi-model LLM routing, safe_ai_call, SQL validation
│   ├── analysis.py             # Statistical analysis, anomaly detection
│   ├── cleaning.py             # Smart imputation, PII masking, Sweetviz profiling
│   ├── ingestion.py            # File upload, URL ingestion, Google Sheets
│   ├── export.py               # Script generation, PDF export
│   └── visualization.py       # Chart helper utilities
│
├── views/                      # Streamlit UI tab components
│   ├── tab_ingestion.py        # Ingestion Lab UI
│   ├── tab_cleaning.py         # Smart Clean UI
│   ├── tab_visualizations.py   # Manual chart builder UI
│   ├── tab_generative_viz.py   # NL-to-chart AI UI
│   ├── tab_assistant.py        # AI chat + Live Surgery UI
│   ├── tab_etl.py              # ETL Pipeline builder UI
│   ├── tab_forecasting.py      # Forecasting Lab UI
│   ├── tab_journalist.py       # Journalist Lab UI
│   ├── tab_comparison.py       # Dataset comparison UI
│   ├── tab_summary.py          # Executive KPI summary UI
│   └── tab_export.py           # Export options UI
│
├── connectors/
│   └── sql_connector.py        # DuckDB / MotherDuck connection abstraction
│
├── tests/                      # pytest test suite
│   └── test_cleaning.py
│
└── .streamlit/
    └── secrets.toml            # API keys (gitignored)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip or conda

### 1. Clone the Repository
```bash
git clone https://github.com/satvikshrivastava06/Analyzer_Visualizer_tool1.git
cd Analyzer_Visualizer_tool1
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create `.streamlit/secrets.toml`:
```toml
[gemini]
api_key = "YOUR_GEMINI_API_KEY"     # Free tier — https://aistudio.google.com

[groq]
api_key = "YOUR_GROQ_API_KEY"       # Free tier — https://console.groq.com

[motherduck]
token    = "YOUR_MOTHERDUCK_TOKEN"   # Optional — cloud DuckDB
db_name  = "my_db"                   # Optional
```
> **Note:** The app runs fully without any API keys — AI features gracefully degrade to informational prompts.

### 5. Run the App
```bash
streamlit run app.py
```

Navigate to **http://localhost:8501** in your browser.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

The test suite covers smart imputation, PII masking, duplicate detection, and anomaly detection logic.

---

## 🖥️ Tech Stack

| Layer | Technology |
|:---|:---|
| **Framework** | Streamlit |
| **Data Engine** | DuckDB 1.4.4 (local + MotherDuck cloud) |
| **AI / LLM** | Google Gemini (2.5-flash, 2.0-flash), Groq (Llama 3.3 70B, Mixtral 8x7B) |
| **Visualization** | Plotly, Altair, ECharts (streamlit-echarts), PyGWalker |
| **Data Grid** | Streamlit AG-Grid (Excel-style editing) |
| **ML** | scikit-learn (Polynomial Regression) |
| **Profiling** | Sweetviz |
| **PDF Export** | FPDF2 |
| **Chart Image Export** | Kaleido |
| **Design System** | Custom CSS (DM Sans, Space Grotesk, JetBrains Mono) |

---

## 🔐 Security & Governance

- **SQL Injection Guard**: Every query (user or AI-generated) is scanned for `DROP`, `DELETE`, `TRUNCATE`, `ALTER`, `INSERT`, `UPDATE` before execution
- **PII Redaction**: Heuristic scanner masks sensitive columns before data reaches any external API or chart
- **Read-Only Mode**: The DuckDB engine operates in an analytical (read-only) context by default
- **Audit Log**: Every transformation is timestamped and stored in the immutable version history sidebar

---

## 📦 Deployment

### Streamlit Community Cloud (Free)
1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `app.py` and add your secrets via the Secrets Manager UI

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ by [Satvik Shrivastava](https://github.com/satvikshrivastava06)*
