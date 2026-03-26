# AStats — Agentic AI for Applied Statistical Workflows
> GSoC 2026 Project #33 | INCF / University of Wisconsin-Madison

AStats is an agentic AI system that automates applied statistical workflows. It combines specialized AI agents with a local LLM (Llama 3.1) to profile datasets, reason about data quality, and build reproducible preprocessing pipelines — all with a full audit trail of every decision.

---

## What's built so far

| Agent | What it does |
|---|---|
| **EDA Agent** | Profiles any dataset — distributions, outliers, normality tests, correlations, missing data, class imbalance |
| **Orchestrator** | Uses Llama 3.1:8b to reason about the EDA report and decide the best statistical approach (A/B/C) with a full preprocessing plan |
| **Visualizer** | Generates 8 interactive Plotly charts — distributions with KDE, per-column box plots, correlation heatmap, skewness chart, quality gauge |

---

## Requirements

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) (for local LLM — free, no API key needed)
- Windows / Mac / Linux

---

## Setup

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/astats.git
cd astats
```

### Step 2 — Create a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

You'll see `(.venv)` in your terminal — this means the environment is active.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This takes 3–5 minutes the first time. It installs pandas, scipy, ydata-profiling, plotly, scikit-learn, xgboost, shap, rich, typer, and more.

### Step 4 — Install Ollama and pull the LLM

1. Download Ollama from [https://ollama.com/download](https://ollama.com/download) and install it
2. Pull the model (one time, ~4.7GB download):

```bash
ollama pull llama3.1:8b
```

3. Test it's working:

```bash
ollama run llama3.1:8b "what is skewness in statistics?"
```

> **No GPU?** AStats works without a GPU — the LLM just runs slower on CPU. The EDA Agent and all visualizations work without Ollama too (rule-based fallback kicks in automatically).

---

## How to run

### Quick demo (no data needed)

Runs on a built-in synthetic messy dataset to show all features:

```bash
python orchestrator.py
```

### Run on your own CSV

```bash
python orchestrator.py run --data path/to/your/data.csv --target your_target_column
```

**Example:**
```bash
python orchestrator.py run --data sales.csv --target churn
```

### Run EDA only (without orchestrator)

```bash
python EDA_Agent.py
```

### CLI commands

```bash
# See all options
python orchestrator.py --help

# Run on a dataset
python orchestrator.py run --data data.csv --target label

# View the full audit trail of a run
python orchestrator.py history astats_output/DATASET/RUN_ID/workflow_state.json

# Check pipeline status
python orchestrator.py status astats_output/DATASET/RUN_ID/workflow_state.json
```

---

## Output files

Every run creates a timestamped folder inside `astats_output/`. Here's what gets generated:

```
astats_output/
└── your_dataset/
    └── 20260325_123456/          ← run ID (timestamp)
        ├── *_profile.html        ← full ydata-profiling deep scan (open in browser)
        ├── *_distributions.html  ← histogram + KDE for each numeric column
        ├── *_boxplots.html       ← box plots, each column on its own scale
        ├── *_correlation.html    ← correlation heatmap (highlights r > 0.85)
        ├── *_missing.html        ← missing data bar chart with thresholds
        ├── *_target_dist.html    ← target variable distribution
        ├── *_quality_gauge.html  ← data quality score gauge + issues table
        ├── *_skewness.html       ← skewness by feature, color coded by severity
        ├── *_eda_report.json     ← structured EDA report (machine readable)
        └── workflow_state.json   ← full audit trail of every decision
```

### How to open the plots

**Windows:**
```bash
start astats_output\your_dataset\RUN_ID\your_dataset_distributions.html
```

**Mac:**
```bash
open astats_output/your_dataset/RUN_ID/your_dataset_distributions.html
```

Or just open the folder in Explorer / Finder and double-click any `.html` file.

---

## Understanding the plots

### Distributions (`*_distributions.html`)
- Each numeric column gets its own subplot with a histogram and KDE curve
- **Red** = has outliers, **Green** = normally distributed, **Orange** = skewed, **Blue** = default
- Skewness value shown in each subplot title

### Box plots (`*_boxplots.html`)
- Each column plotted on **its own scale** — no columns are crushed by others
- Outlier count shown in subplot title
- Red boxes = columns with outliers detected

### Correlation matrix (`*_correlation.html`)
- Red = strong positive correlation, Blue = strong negative
- **Black boxes highlight pairs with |r| > 0.85** — multicollinearity risk

### Missing data (`*_missing.html`)
- Bar chart showing % missing per column
- Orange dashed line at 5%, Red dashed line at 20%

### Quality gauge (`*_quality_gauge.html`)
- Score from 0–100 based on detected issues
- Green = 80+, Yellow = 60–80, Red = below 60
- Issues table shown alongside the gauge

### Skewness chart (`*_skewness.html`)
- Red bars = |skew| > 1 (needs transformation)
- Orange = 0.5–1 (moderate)
- Green = < 0.5 (acceptable)

---

## Approach A / B / C

The Orchestrator recommends one of three approaches based on data quality:

| Approach | When | Preprocessing |
|---|---|---|
| **A** — Standard | Clean data, no major issues | StandardScaler, basic imputation |
| **B** — Robust | 1–2 issues (outliers or skew or missing) | RobustScaler, Winsorize, Yeo-Johnson |
| **C** — Ensemble | 3+ issues | IsolationForest, SMOTE, PCA, drop_vif |

---

## Project structure

```
astats/
├── EDA_Agent.py          # Day 1 — dataset profiling agent
├── orchestrator.py       # Day 2 — LLM orchestrator + CLI
├── visualizer.py         # Improved Plotly visualizations
├── requirements.txt      # All dependencies
└── astats_output/        # Generated reports (gitignored)
```

---

## Roadmap

- [x]  — EDA Agent
- [x]  — LLM Orchestrator + CLI + Audit trail
- [ ]  — Modeling Agent (sklearn + xgboost + lightgbm)
- [ ]  — Inference Agent (statsmodels + pingouin)
- [ ]  — Explanation Agent (SHAP + LIME)
- [ ]  — Full end-to-end pipeline
- [ ]  — Final polish + HTML report

---

## Tech stack

`pandas` · `numpy` · `scipy` · `ydata-profiling` · `scikit-learn` · `xgboost` · `lightgbm` · `statsmodels` · `pingouin` · `shap` · `lime` · `plotly` · `typer` · `rich` · `ollama (llama3.1:8b)`

---

## GSoC 2026

This project is being developed as part of Google Summer of Code 2026 under INCF / University of Wisconsin-Madison.

Mentors: Jonathan Morris, Yohai-Eliel Berreby, Suresh Krishna
