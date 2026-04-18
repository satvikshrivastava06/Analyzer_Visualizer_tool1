<!--
Copyright (c) 2026 Satvik Shrivastava. All rights reserved.
Unauthorized copying, modification, or distribution is strictly prohibited.
-->

# Data Analyzer & Visualizer Tool — Comprehensive Project Report

**Author:** Satvik Shrivastava
**Platform Version:** v2.1
**Report Date:** April 2026
**Repository:** [github.com/satvikshrivastava06/Analyzer_Visualizer_tool1](https://github.com/satvikshrivastava06/Analyzer_Visualizer_tool1)

---

## 1. Executive Summary

The **Data Analyzer & Visualizer Tool** is a production-grade, open-source analytics platform built on the Streamlit web framework. The project bridges a critical gap in the modern data tooling landscape: the prohibitive cost and technical complexity of enterprise BI platforms (Tableau, Power BI, Alteryx) versus the fragmented, code-only experience of pure Python data science workflows.

The platform enables a complete **Ingest → Profile → Clean → Transform → Visualize → Report** analytical lifecycle within a single browser tab, powered by an in-process OLAP engine (DuckDB), a zero-cost multi-provider AI system (Gemini / Groq), and a comprehensive visualization suite (Plotly, Altair, ECharts, PyGWalker).

The system is architected for modularity, testability, and long-term maintainability — following strict separation of concerns between business logic (`modules/`), UI components (`views/`), and the design system (`ui_theme.py`).

---

## 2. Problem Statement & Motivation

The project was motivated by observing the following unresolved problems in the current data tooling market:

| Problem | Market Impact |
|:---|:---|
| Enterprise BI tools (Tableau, Power BI) cost $70–$140/user/month | Prohibitive for individual analysts and small teams |
| No single tool covers Ingest → Clean → Model → Visualize → Export | Analysts switch between 5–7 tools, losing time and lineage |
| AI "Copilots" are locked to single vendors (Microsoft, Salesforce) | API quota exhaustion breaks workflows with no fallback |
| Data cleaning is manual and unreproducible | Black-box sessions impossible to audit or verify |
| Sensitive PII routinely sent to external APIs without masking | Data privacy and GDPR compliance violations |
| AI visualization recommendations are unreliable (hallucinations) | Incorrect chart types erode trust in AI-assisted analytics |

This project addresses each of these problems with a specific, engineered solution documented in Section 4.

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (app.py)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │Ingestion │ │  Clean   │ │ Visualize│ │   AI Assistant   │  │
│  │   Lab    │ │  (Smart) │ │ (Multi   │ │  (Live Surgery)  │  │
│  │          │ │          │ │  Engine) │ │                  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘  │
│       │            │            │                 │             │
│  ┌────▼────────────▼────────────▼─────────────────▼──────────┐ │
│  │                    ui_theme.py (Design System)             │ │
│  └────────────────────────────┬───────────────────────────────┘ │
└───────────────────────────────┼─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                       Business Logic Layer                       │
│   modules/ai_engine.py        Multi-model LLM routing            │
│   modules/cleaning.py         Data profiling & transformation    │
│   modules/analysis.py         Statistical analysis & anomalies   │
│   modules/ingestion.py        File & URL data loading            │
│   modules/export.py           Script & PDF generation            │
│                                │                                 │
└───────────────────────────────┼─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                         Data Layer                               │
│   DuckDB (In-Process OLAP)    ←→    MotherDuck (Cloud)           │
│   Pandas DataFrames (cache)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Inventory

| File / Module | Responsibility |
|:---|:---|
| `app.py` | Entry point, session state management, sidebar, tab routing |
| `ui_theme.py` | Design system — CSS tokens, layout components, helper functions |
| `pyg_viz.py` | PyGWalker BI Explorer with Geo Map Mode (geographic coordinate system) |
| `premium_viz.py` | ECharts premium chart library (Gauge, Radar, Sankey, Candlestick, 3D Bar) |
| `visualization.py` | Legacy Plotly chart builders (retained for compatibility) |
| `modules/ai_engine.py` | Multi-model LLM routing, `safe_ai_call`, SQL validation sandbox |
| `modules/analysis.py` | Z-score anomaly detection, statistical summaries |
| `modules/cleaning.py` | Smart imputation, PII masking, Sweetviz integration |
| `modules/ingestion.py` | File upload, URL fetch, Google Sheets connector |
| `modules/export.py` | Reproducible script generator, PDF export via FPDF2 |
| `views/tab_ingestion.py` | Ingestion Lab UI |
| `views/tab_cleaning.py` | Smart Clean UI — profiling, interventions, deep profiling |
| `views/tab_visualizations.py` | Manual chart builder (13+ chart types) |
| `views/tab_generative_viz.py` | NL-to-chart AI visualization engine |
| `views/tab_assistant.py` | AI chat interface + Perform Live Surgery engine |
| `views/tab_etl.py` | Multi-step SQL ETL pipeline builder |
| `views/tab_forecasting.py` | Scikit-learn regression forecasting lab |
| `views/tab_journalist.py` | FiveThirtyEight-style data journalism studio |
| `views/tab_comparison.py` | Dataset delta comparison engine |
| `views/tab_summary.py` | Executive KPI summary dashboard |
| `views/tab_export.py` | Data export & session reproducibility |
| `connectors/sql_connector.py` | DuckDB / MotherDuck connection abstraction |
| `tests/test_cleaning.py` | Pytest test suite for core cleaning logic |

### 3.3 Technology Stack

| Layer | Technology | Rationale |
|:---|:---|:---|
| **Web Framework** | Streamlit | Rapid development; Python-native reactive UI |
| **Data Engine** | DuckDB 1.4.4 (+ MotherDuck) | In-process OLAP; handles datasets far beyond Pandas memory limits |
| **Primary AI** | Google Gemini (2.5-flash, 2.0-flash) | Free tier; strong reasoning on tabular schemas |
| **Fallback AI** | Groq (Llama 3.3 70B, Mixtral 8x7B) | Zero-cost; fast inference; high-parameter open weights |
| **Charting (Primary)** | Plotly | Interactive, publication-quality charts |
| **Charting (Declarative)** | Vega-Altair | Grammar-of-graphics precision for NVBench-compliant prompts |
| **Charting (Premium)** | Apache ECharts via streamlit-echarts | Animated, canvas-rendered premium visuals |
| **Drag-and-Drop BI** | PyGWalker | Tableau-style canvas; browser-native; free |
| **Data Grid** | Streamlit AG-Grid | Excel-like editing with grouping and advanced sorting |
| **ML / Forecasting** | scikit-learn | Polynomial regression; lightweight; no GPU required |
| **Profiling** | Sweetviz | One-click interactive HTML EDA reports |
| **PDF Export** | FPDF2 | Lightweight; no external LaTeX dependencies |
| **Chart Image Export** | Kaleido | Headless Plotly → PNG/SVG export |
| **Design System** | Custom CSS (DM Sans, Space Grotesk, JetBrains Mono) | Premium dark theme; brand-consistent across all components |

---

## 4. Development History & Problem-Solution Mapping

The platform was developed across four distinct phases. Each phase addressed a specific set of engineering and research challenges.

---

### Phase 1: Foundation (February 2026)

**Objective:** Build a functional AI-powered analysis and cleaning tool that overcomes the limitations of existing single-model, Pandas-only approaches.

#### Problem 1.1 — Pandas Memory Ceiling
- **Problem:** Standard analytics tools load entire datasets into RAM. Datasets exceeding 2–5 GB crash Python processes.
- **Solution:** Replaced Pandas as the primary compute layer with **DuckDB**. All ingested data is registered as DuckDB tables/views. SQL queries execute using DuckDB's columnar vectorized engine, enabling datasets 100× larger than Pandas can handle. MotherDuck support extends this to a serverless cloud context.

#### Problem 1.2 — Manual Cleaning Fatigue
- **Solution:** Built the **Smart Clean** tab with automated imputation (median for numeric, mode for categorical) and one-click duplicate/NA dropping. Analysts no longer write boilerplate Pandas cleaning code.

#### Problem 1.3 — No Natural Language Interface
- **Solution:** Integrated Google Gemini to produce a conversational AI Assistant. Users describe what they need ("show me the top 10 customers by revenue") and the AI generates validated DuckDB SQL.

---

### Phase 2: Resilience & Academic Guardrails (March 2026)

**Objective:** Harden the AI system against failure and implement visualization best practices from the *International Journal of Computing and Digital Systems (IJCDS, March 2024)*.

#### Problem 2.1 — Single-Model AI Fragility (IJCDS §7.B)
- **Problem:** Gemini API rate limits (Error 429) or model deprecations (Error 404) crashed the entire AI layer.
- **Solution:** **Multi-Provider Fallback Architecture** — `safe_ai_call()` cascades through a priority list: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash` → `groq/llama-3.3-70b` → `groq/mixtral-8x7b-32768`. This guarantees 100% AI uptime at zero cost.

#### Problem 2.2 — DuckDB Session State Loss (`CatalogException`)
- **Problem:** Streamlit reruns drop temporary DuckDB views. Queries on dropped views threw `CatalogException` and crashed the app.
- **Solution:** **Dual-Path Schema Detection** — all schema queries use `try-except`. On `CatalogException`, the system falls back to the cached `df_preview` Pandas DataFrame for `dtypes` and `len()`, permanently eliminating the crash.

#### Problem 2.3 — Data Privacy Risk (IJCDS §7.A)
- **Problem:** Sensitive PII (emails, SSNs, phone numbers) was being passed raw to external AI APIs and rendered in charts.
- **Solution:** **Pre-Flight PII Anonymizer** in the Smart Clean tab. A heuristic column-name scanner detects sensitive identifiers and replaces cell contents with `[REDACTED]` before any downstream processing.

#### Problem 2.4 — Browser Freezing on Large Datasets (IJCDS §6.D)
- **Solution:** **Smart Algorithmic Sampling** — datasets exceeding 10,000 rows are automatically downsampled to a statistically representative random subset before being sent to Plotly/Altair/ECharts rendering engines.

#### Problem 2.5 — Data Oversimplification (IJCDS §6.B)
- **Solution:** Expanded chart library to include **Treemaps, Sunbursts, Violin Plots** — capturing hierarchical and distributional nuance that bar/pie charts cannot express.

#### Problem 2.6 — Human Data Entry Errors (IJCDS §6.A)
- **Solution:** **Statistical Anomaly Scanner** — evaluates all numeric columns for values with Z-score > 3, flagging likely typos and impossible values.

#### Problem 2.7 — AI Hallucination in Chart Recommendations (IJCDS §7.B)
- **Solution:** AI chart suggestions are anchored to on-screen **AI Provenance Disclaimers** — un-hideable badges flagging machine-generated visuals and requiring explicit human validation.

#### Problem 2.8 — Executive Cognitive Overload (IJCDS Best Practice 9)
- **Solution:** **Executive Summary** tab isolated from the operational Smart Clean workflow. Displays high-level KPIs with temporal deltas (`% vs. Previous Period`) derived from auto-detected `datetime64` columns.

---

### Phase 3: Enterprise-Grade Security & Semantics (March 2026)

**Objective:** Add security controls, reproducibility, and a semantic layer that meets enterprise data engineering standards.

#### Problem 3.1 — SQL Injection & Destructive AI Output
- **Solution:** **SQL Security Sandbox** — `validate_sql_query()` intercepts every query (user-typed or AI-generated) and blocks execution if destructive keywords (`DROP`, `DELETE`, `TRUNCATE`, `ALTER`, `INSERT`, `UPDATE`) are detected.

#### Problem 3.2 — Logical Data Inconsistencies
- **Solution:** **Data Quality Audit** — performs automated logical checks: negative financial values, malformed email patterns, impossible date ranges, and string-in-numeric columns. Surfaces results as categorized anomaly rows.

#### Problem 3.3 — Poor Business Data Documentation
- **Solution:** **AI Data Dictionary** — generates a structured Markdown document interpreting column names, inferring likely business meaning, and suggesting strategic KPIs for any loaded schema.

#### Problem 3.4 — Keyword-Limited Search (Semantic Gap)
- **Solution:** **Semantic Search (RAG Lite)** — the AI translates natural language concepts ("high-value at-risk customers") into optimized DuckDB SQL filter logic, enabling concept-based rather than keyword-based data retrieval.

#### Problem 3.5 — Analytical Reproducibility Crisis
- **Solution:** **Reproducible Export Pipeline** — users download a standalone `reproducible_analysis.py` script containing the exact DuckDB connection strings, cleaning operations, and session parameters. Also exports a session-matched `requirements.txt`.

---

### Phase 4: Agentic Era & Advanced Features (March–April 2026)

**Objective:** Graduate the AI from a passive advisor to an active data engineering agent, and add the advanced analytical features that differentiate this from commodity BI tools.

#### Problem 4.1 — Static AI Insights (Reactive-Only BI)
- **Problem:** Tableau and Power BI provide insights but cannot *execute* data transformations. Users must manually implement AI suggestions.
- **Solution:** **Agentic Live Surgery Engine** (`views/tab_assistant.py`). AI recommendations embed executable SQL blocks. Clicking "Perform Live Surgery" triggers `clean_sql()` (which strips Markdown fence artifacts from AI output), validates the SQL, executes it against DuckDB, updates the session state `df_preview`, and commits the transformation to the version history — all without user-written code.

#### Problem 4.2 — Analytical Lineage / Reproducibility (Git-Lite)
- **Solution:** **Version History Audit Log** in the governance sidebar. Every action — file upload, cleaning operation, SQL query, AI surgery — is recorded with an ISO 8601 timestamp and the exact code/SQL used. This creates a complete analytical lineage from raw ingest to final export.

#### Problem 4.3 — Text-to-SQL Hallucinations
- **Solution:** **WrenAI-Inspired Semantic Manifest**. Users define calculated metrics (e.g., `Net_Profit = Revenue - COGS`) in a structured UI. This manifest is injected into every AI prompt as a deterministic context anchor, forcing the LLM to follow explicit business logic rather than guessing schema mappings.

#### Problem 4.4 — No Self-Service ETL
- **Solution:** **ETL Pipeline Builder** (`views/tab_etl.py`). A visual multi-step SQL transformation interface backed by DuckDB. Users add steps (SQL transformations), preview the output at each step, and execute the full pipeline sequentially. Replaces the "Coming Soon" placeholder with a production-functional ELT engine.

#### Problem 4.5 — No Geographic Visualization
- **Solution:** **PyGWalker Geo Map Mode** (`pyg_viz.py`). Auto-detects latitude/longitude columns in the dataset, enables a `🌍 Geo Map Mode` toggle that switches the PyGWalker coordinate system to `"geographic"` with `"poi"` (point-of-interest) geometry, and binds detected columns automatically. Leaflet map tiles are styled with a dark-mode CSS filter.

#### Problem 4.6 — CSS Engine Instability (f-string Conflicts)
- **Problem:** `ui_theme.py` used Python f-strings to inject CSS. Literal CSS curly braces (`{` / `}`) were misinterpreted as Python format variables, causing `SyntaxError` and `NameError` on startup.
- **Solution:** All literal CSS blocks use double-curly-brace escaping (`{{ ... }}`). Python template variables (design tokens like `{ACCENT}`, `{SURFACE_2}`) retain single braces. This convention is documented with a comment block inside `ui_theme.py`.

#### Problem 4.7 — Dataframe Canvas Invisible (Dark Overlay)
- **Problem:** CSS rules `background: SURFACE_2 !important` on `[data-testid="stDataFrame"] > div`, `.dvn-scroller`, and `[data-testid="glideDataEditor"]` were painting a dark surface over the Glide Data Grid canvas element, making the Profile Details tab appear blank.
- **Solution:** Removed all `background`, `overflow`, and `padding` custom CSS from the inner dataframe wrapper elements. Streamlit's native dark theme renders the Glide Data Grid correctly without CSS intervention. Only the outer `[data-testid="stDataFrame"]` shell retains `border-radius`.

#### Problem 4.8 — AI-Injected SQL Syntax Artifacts
- **Problem:** Gemini models occasionally wrapped SQL output in Markdown fences (` ```sql ... ``` `) or backtick characters. Passing these raw strings to DuckDB produced `Parser Error: syntax error at or near "\`"`.
- **Solution:** `clean_sql()` helper in `tab_assistant.py` strips Markdown code fences, leading/trailing backticks, `sql` language tags, and normalizes whitespace before any DuckDB execution.

#### Problem 4.9 — Sweetviz Report Disappearing
- **Problem:** The Sweetviz HTML report was re-rendered on every Streamlit rerun, causing it to flicker and disappear when any UI element triggered a state change.
- **Solution:** Report display is bound to `st.session_state.show_sweetviz`. Once set to `True`, the report persists across all reruns until the user explicitly clicks "Close Interactive Report".

---

## 5. Design System

### 5.1 Color Tokens

| Token | Value | Usage |
|:---|:---|:---|
| `ACCENT` | `#6C63FF` | Primary CTA, active tabs, focus rings |
| `SURFACE_1` | `#0D0F14` | Page background |
| `SURFACE_2` | `#13151C` | Card background |
| `SURFACE_3` | `#1A1D28` | Elevated card, input background |
| `SURFACE_4` | `#21253A` | Hover state |
| `BORDER` | `#2A2E45` | Subtle dividers |
| `TEXT_1` | `#F0F2FF` | Primary text |
| `TEXT_2` | `#8B90B4` | Secondary / muted text |
| `TEXT_3` | `#4B5073` | Placeholder / disabled |
| `SUCCESS` | `#00D4A0` | Positive delta, clean data |
| `WARNING` | `#FFB547` | Attention indicators |
| `DANGER` | `#FF5F5F` | Errors, anomalies, destructive actions |
| `INFO` | `#38B6FF` | Informational callouts |

### 5.2 Typography

| Font | Usage |
|:---|:---|
| Space Grotesk | Headings, hero titles, card values |
| DM Sans | Body text, labels, UI elements |
| JetBrains Mono | Code blocks, SQL, model badges |

### 5.3 Key CSS Engineering Decision

All CSS in `ui_theme.py` is delivered via Python f-strings inside `PREMIUM_CSS`. Literal CSS braces are escaped as `{{ }}` while design token substitutions use single braces `{TOKEN_NAME}`. This is the only pattern that prevents Python's f-string parser from treating CSS property blocks as format expressions.

---

## 6. Security Model

| Control | Implementation |
|:---|:---|
| **SQL Injection Prevention** | `validate_sql_query()` blocklist scan on all queries |
| **PII Protection** | Pre-flight heuristic column masking in Smart Clean |
| **Read-Only Enforcement** | DuckDB operates in analytical mode; no DML on source tables |
| **AI Hallucination Mitigation** | Provenance badges, WrenAI semantic manifest, SQL validation |
| **Audit Trail** | Immutable version history with timestamped action log |
| **Secret Management** | API keys stored in `.streamlit/secrets.toml` (gitignored) |

---

## 7. Testing Strategy

The test suite is located in `tests/` and is executed with pytest.

```bash
pytest tests/ -v
```

**Current test coverage areas:**
- `test_cleaning.py`: Smart imputation correctness, PII masking column detection, duplicate detection, missing NA drop counts, Z-score anomaly threshold

**Planned additions:**
- `test_ai_engine.py`: Multi-model fallback chain validation using mocked API responses
- `test_sql_validator.py`: Blocklist enforcement and edge case SQL syntax
- `test_etl_pipeline.py`: DuckDB view creation and sequential step output validation

---

## 8. Competitive Analysis

| Capability | Tableau / Power BI | This Platform |
|:---|:---|:---|
| **Cost** | $70–140/user/month | Free (open-source) |
| **AI Copilot** | Locked to single vendor | Multi-provider, zero-cost fallback |
| **Data Cleaning** | Limited / premium add-on | Full in-app pipeline |
| **PII Masking** | Enterprise license required | Built-in, one click |
| **Reproducibility** | Proprietary binary format | Exportable Python script |
| **Drag-and-Drop BI** | Core feature (paid) | PyGWalker (free) |
| **Geographic Maps** | Built-in | Geo Map Mode via PyGWalker |
| **ETL / Transformation** | External tools required | Built-in SQL pipeline builder |
| **Forecasting** | Premium add-on | Built-in scikit-learn lab |
| **Agentic Execution** | Not available | Live Surgery engine |
| **Audit Log** | Enterprise only | Built-in Version History |
| **Deployment** | Desktop / cloud subscription | Streamlit Community Cloud (free) |

---

## 9. Known Limitations & Future Roadmap

### 9.1 Current Limitations

| Area | Limitation |
|:---|:---|
| **ETL Persistence** | Pipeline steps are lost on session refresh (in-memory only) |
| **Multi-User** | Streamlit session state is per-user; no shared workspace |
| **LLM Context Window** | Very wide schemas (200+ columns) may exceed token limits |
| **Real-Time Streaming** | No live data streaming connector (Kafka, WebSocket) |
| **Authentication** | No built-in login / RBAC layer |

### 9.2 Future Roadmap

| Priority | Feature |
|:---|:---|
| High | **Pipeline Persistence** — save/load ETL templates as JSON files |
| High | **Streamlit Auth** — integrate Streamlit Authenticator for user sessions |
| Medium | **dbt Integration** — native dbt model compilation and run tracking |
| Medium | **Kafka Connector** — real-time streaming data ingestion |
| Medium | **Advanced Forecasting** — ARIMA, Prophet, XGBoost time-series models |
| Low | **Multi-Tenant Sharing** — MotherDuck-backed shared workspaces |
| Low | **Responsive Mobile Layout** — optimize for tablet/mobile viewports |

---

## 10. Conclusion

The **Data Analyzer & Visualizer Tool** demonstrates a comprehensive, production-aligned approach to modern data engineering challenges. By compositing battle-tested open-source components (DuckDB, Streamlit, Plotly, PyGWalker, scikit-learn) under a unified, AI-augmented interface, it delivers — at zero cost — the core analytical capabilities of platforms that charge hundreds of dollars per user per month.

The platform's most significant innovations are:

1. **Zero-Cost AI Resilience** — A cascading multi-provider LLM fallback that guarantees uptime regardless of API quota constraints
2. **Agentic Data Surgery** — Moving AI from passive advisor to active data transformation agent
3. **Academic Visualization Guardrails** — Prompt architecture aligned with NVBench and ChartQA standards, ensuring chart recommendations are semantically grounded rather than statistically random
4. **Full Analytical Lineage** — An immutable audit trail covering every action from raw ingest to final export

The codebase is structured for extensibility: new AI providers, chart engines, or data connectors can be added without modifying existing tabs, data layer interactions, or the design system.

---

*© 2026 Satvik Shrivastava. All rights reserved.*
