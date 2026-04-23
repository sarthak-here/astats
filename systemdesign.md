# AStats - Agentic EDA System Design

## What It Does
An agentic AI system for automated Exploratory Data Analysis. Feed it any CSV and it
autonomously runs statistical analysis, generates interactive visualizations, detects
data quality issues, and produces a full HTML report -- without writing any analysis code.

---

## Architecture

```
User (CSV file path + optional target column)
        |
        v
+--------------------------------------------------+
|             orchestrator.py                      |
|  Coordinates the agent pipeline:                 |
|  1. Load and profile dataset                     |
|  2. Dispatch sub-tasks to EDA_Agent              |
|  3. Collect results                              |
|  4. Write to astats_output/<dataset>/<timestamp> |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|              EDA_Agent.py                        |
|  - Missing value analysis                        |
|  - Distribution per column                      |
|  - Outlier detection (IQR + Z-score)            |
|  - Correlation matrix (Pearson + Spearman)      |
|  - Skewness and kurtosis                        |
|  - Target variable distribution (if given)      |
|  - Data quality score (0-100)                   |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|              visualizer.py                       |
|  Generates interactive Plotly HTML charts:       |
|  distributions, correlation heatmap, boxplots,  |
|  missing value matrix, quality gauge, skewness  |
+--------------------------------------------------+
        |
        v
  astats_output/<dataset>/<timestamp>/
    eda_report.json       (full stats, machine-readable)
    distributions.html    (interactive histograms)
    correlation.html      (heatmap)
    boxplots.html
    missing.html
    quality_gauge.html
    workflow_state.json   (orchestrator run log)
```

---

## Data Flow

```
my_data.csv
        |
  pandas.read_csv()
        |
  EDA_Agent computes:
  - Shape, dtypes, missing count + % per column
  - Numeric: mean, median, std, min, max, IQR
  - Outliers: IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
  - Skewness: flag columns > 1.0 as highly skewed
  - Pearson correlation matrix
  - Data quality score:
      100 - (missing_penalty + outlier_penalty + skew_penalty)
        |
  visualizer.py generates Plotly HTML per metric
        |
  orchestrator.py writes:
  - eda_report.json
  - One .html chart per metric
  - workflow_state.json (step completion timestamps)
```

---

## Key Design Decisions

| Decision                        | Reason                                            |
|---------------------------------|---------------------------------------------------|
| Plotly HTML output              | Self-contained interactive charts; no server needed|
| Timestamped output folders      | Multiple runs don't overwrite each other          |
| workflow_state.json             | Enables resuming interrupted analysis             |
| Separate EDA_Agent + visualizer | Analysis logic testable independently of rendering|
| Data quality score (0-100)      | Single number: is this dataset ready for modeling?|

---

## Interview Conclusion

AStats solves the "blank notebook" problem: data scientists spend hours writing the same
profiling boilerplate before any real analysis begins. The orchestrator pattern separates
task coordination from analysis logic -- orchestrator.py knows what to run and in what
order, EDA_Agent.py knows how to compute each statistic, and visualizer.py converts
numbers to Plotly HTML. The data quality score is a design choice I am proud of: it
distills missing values, outliers, and skewness into a single number that immediately
tells you whether a dataset needs heavy preprocessing or is ready for modeling. Scaling:
add an LLM layer that reads eda_report.json and writes a natural-language narrative
of the findings.
