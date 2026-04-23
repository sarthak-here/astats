# AStats — Agentic EDA System Design

## What It Does
An agentic AI system for automated Exploratory Data Analysis (EDA). Feed it any CSV or dataset and it autonomously runs statistical analysis, generates interactive visualizations, detects data quality issues, and produces a full HTML report — without writing a single line of analysis code.

---

## Architecture

```
User (CSV file path + optional target column)
        |
        v
+--------------------------------------------------+
|             orchestrator.py                      |
|  Coordinates the agent pipeline:                 |
|  1. Load & profile dataset                       |
|  2. Dispatch sub-tasks to EDA_Agent              |
|  3. Collect results                              |
|  4. Write output to astats_output/<dataset>/     |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|              EDA_Agent.py                        |
|  The core analysis engine:                       |
|  - Missing value analysis                        |
|  - Distribution analysis per column             |
|  - Outlier detection (IQR + Z-score)            |
|  - Correlation matrix (Pearson + Spearman)      |
|  - Skewness / kurtosis                          |
|  - Target variable distribution (if given)      |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|              visualizer.py                       |
|  Generates interactive HTML charts (Plotly):     |
|  - Distributions histogram per feature           |
|  - Correlation heatmap                          |
|  - Box plots (outlier visualization)            |
|  - Missing value matrix                         |
|  - Quality gauge (data quality score 0-100)     |
|  - Skewness chart                               |
+--------------------------------------------------+
        |
        v
  astats_output/<dataset>/<timestamp>/
    eda_report.json          (machine-readable full stats)
    distributions.html       (interactive histograms)
    correlation.html         (heatmap)
    boxplots.html            (outlier view)
    missing.html             (missingness matrix)
    quality_gauge.html       (overall data quality score)
    profile.html             (ydata-profiling full report)
    workflow_state.json      (orchestrator state log)
```

---

## Input

| Input | Detail |
|---|---|
| CSV file path | Any tabular dataset |
| Target column (optional) | Enables target distribution and correlation ranking |
| Dataset name | Used for output folder naming |

---

## Data Flow

```
my_dataset.csv
        |
  pandas.read_csv()
        |
  EDA_Agent analyzes:
  - Shape: (n_rows, n_cols)
  - Dtypes: numeric / categorical / datetime
  - Missing: count + % per column
  - Numeric stats: mean, median, std, min, max, IQR
  - Outliers: IQR method (< Q1-1.5*IQR or > Q3+1.5*IQR)
  - Skewness: flag columns > 1.0 as highly skewed
  - Correlations: Pearson matrix for numeric columns
  - Data quality score:
      100 - (missing_penalty + outlier_penalty + skew_penalty)
        |
        v
  visualizer.py generates Plotly HTML for each metric
        |
        v
  orchestrator.py writes:
  - eda_report.json  (full stats dictionary)
  - One .html file per visualization
  - workflow_state.json (which steps completed, timestamps)
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Plotly HTML output | Self-contained interactive charts; no server needed to view |
| Timestamped output folders | Multiple runs on same dataset don't overwrite each other |
| workflow_state.json | Enables resuming interrupted analysis runs |
| Separate EDA_Agent and visualizer | Analysis logic stays testable independently of rendering |
| Data quality score (0-100) | Single number for quick dataset assessment before modeling |

---

## Interview Conclusion

AStats solves the "blank notebook" problem that data scientists face at the start of every new dataset: spending hours writing the same profiling boilerplate before any real analysis begins. The orchestrator pattern separates task coordination from the analysis logic — orchestrator.py knows what needs to happen and in what order, while EDA_Agent.py knows how to compute each statistic. The visualizer layer converts raw numbers into Plotly HTML, which opens in any browser without installing anything. The data quality score is a design choice I am particularly proud of: it distills missing values, outliers, and skewness into a single number that immediately tells you whether a dataset needs heavy preprocessing or is ready for modeling. If I were scaling this, I would add an LLM layer that reads the eda_report.json and generates a natural-language narrative interpretation of the findings.
