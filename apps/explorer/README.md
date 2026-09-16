# 🔬 Corporate AI Disclosure Explorer

Interactive analytics and visual dataset explorer powered by an internal, in-memory **DuckDB** instance and **Gradio**. Designed for SEC filings (10-K, 10-Q) and earnings call transcripts across the S&P 500 panel.

Compatible with local environments and **Hugging Face Spaces** via protocol-agnostic data resolution (`DATA_URI`).

---

## ✨ Features

### 1. 📊 Visual Gold Explorer (Mini-Tableau)
- **No table clutter or raw schema dumps:** Clean, Tableau-style Columns & Rows shelf.
- **Curated Gold Datasets:**
  - `📅 Firm-Year (Annual Panel)`: AI disclosure volume, washing score ($w$), substance ratio, proprietary AI, posture rates.
  - `🏢 Firm Static (Archetypes & Clusters)`: AI posture archetypes, stability scores, positioning.
  - `📊 Firm-Quarter (Quarterly Panel)`: Quarterly disclosure dynamics.
  - `📑 Documents & Filings`: Document-level AI frame counts and investment metrics.
  - `⚡ AI Activities`: Atomic actions, deployment stages, and provider families.
- **Interactive Plotly Charts:** Real-time bar charts, time-series lines, scatter plots, and box plots with hover tooltips and clean typography.
- **Summary Pivot View:** Clean, aggregated data preview beneath the chart.

### 2. 🏢 Company Profile & Filing Inspector
- **Searchable Company Selector:** Instant autocomplete across all S&P 500 tickers.
- **Corporate Card:** Status badge (`Active (S&P 500)` / `Delisted`), AI Posture Archetype (`Governance-Led Disclosers`, `Defensive Disclosers`, etc.), stability score, and posture intensity breakdown.
- **Historical Trajectory:** Dual Plotly charts displaying disclosure volume surge and the trajectory of *Washing Score* ($w$) vs *Substance Ratio*.
- **Filing & AI Paragraph Inspector:**
  - Select any SEC filing or earnings call transcript.
  - Read **the full text** of classified AI disclosure paragraphs formatted as modern cards with tags (`Subject`, `Type`, `Posture`, `Rhetoric`, `Specificity`).
- **Atomic AI Activities:** Catalog of classified AI actions for the selected firm.

### 3. ⚡ Interactive SQL Playground (DuckDB)
- Ad-hoc SQL querying over in-memory Parquet views.
- Curated presets for common thesis questions (top AI disclosers, annual washing trends, archetype distribution, provider families).
- Interactive schema and table catalog browser.

---

## 🎨 Typography & Design
- Primary UI typeface: **Inter** (via Google Fonts).
- Monospace font: **JetBrains Mono** for code, SQL, accession numbers, and metrics.
- Minimalist palette with clean borders, rounded corners, and Tableau-like visual polish.

---

## ⚙️ Environment Variables & Storage Protocols

| Variable | Description | Default |
|---|---|---|
| `DATA_URI` | Path or URI to Parquet datasets (`local`, `s3://...`, `https://...`, or `hf://...`) | Local `data/` folder |
| `DATA_DIR` / `DATA_PATH` | Backward-compatible aliases for `DATA_URI` | — |
| `S3_ENDPOINT` | S3 / Backblaze B2 endpoint (e.g. `s3.us-west-004.backblazeb2.com`) | Optional |
| `S3_ACCESS_KEY_ID` | S3 / B2 Key ID | Optional |
| `S3_SECRET_ACCESS_KEY` | S3 / B2 Application Secret | Optional |
| `S3_REGION` | S3 region | `us-east-1` |
| `HF_TOKEN` | Hugging Face token for private spaces or datasets | Optional |
| `EXPLORER_PORT` | Gradio server port | `7860` |
| `EXPLORER_HOST` | Gradio server host | `0.0.0.0` |

---

## 🚀 Quickstart

### Local Run
```bash
# Using uv (recommended)
uv run python -m apps.explorer.app

# Or via Makefile
make explorer
```

### Remote / Cloud Storage (Backblaze B2 / S3)
```bash
DATA_URI=s3://my-bucket/thesis/data \
S3_ENDPOINT=s3.us-west-004.backblazeb2.com \
S3_ACCESS_KEY_ID=xxx \
S3_SECRET_ACCESS_KEY=yyy \
uv run python -m apps.explorer.app
```

### Hugging Face Spaces Deployment
1. Create a new Space on Hugging Face using the **Gradio** SDK.
2. Push the files in `apps/explorer/` (`app.py`, `config.py`, `db.py`, `views/`, `requirements.txt`).
3. Set `DATA_URI` under **Space Settings > Variables and secrets**.
