"""
AStats — EDA Agent (Day 1)
GSoC 2026 Project #33
=====================================================
Full-power EDA using ydata-profiling, scipy, plotly, rich.
Outputs:
  - Structured EDAReport (consumed by Orchestrator on Day 2)
  - Interactive HTML report (open in browser)
  - Rich terminal summary
  - Plotly visualizations saved as HTML
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field, asdict
from visualizer import generate_visualizations
from typing import Optional
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from ydata_profiling import ProfileReport

warnings.filterwarnings("ignore")
console = Console()


# ─────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str
    n_missing: int
    pct_missing: float
    n_unique: int
    cardinality_level: str
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    is_normal: Optional[bool] = None
    normality_p_value: Optional[float] = None
    has_outliers: Optional[bool] = None
    outlier_count: Optional[int] = None
    top_values: Optional[dict] = None


@dataclass
class EDAReport:
    dataset_name: str
    n_rows: int
    n_cols: int
    n_missing_cells: int
    pct_missing_cells: float
    duplicate_rows: int
    memory_usage_mb: float
    columns: list = field(default_factory=list)
    correlation_matrix: Optional[dict] = None
    # Flags for Orchestrator
    has_missing_data: bool = False
    has_high_cardinality: bool = False
    has_skewed_features: bool = False
    has_outliers: bool = False
    has_class_imbalance: bool = False
    has_multicollinearity: bool = False
    target_type: Optional[str] = None
    # A / B / C recommendation
    recommended_approach: str = "A"
    approach_rationale: str = ""
    data_quality_score: float = 100.0
    target_column: Optional[str] = None

# ─────────────────────────────────────────────────────
# EDA Agent
# ─────────────────────────────────────────────────────

class EDAAgent:
    """
    Specialized EDA Agent for AStats.
    Combines ydata-profiling (deep scan) with custom
    statistical tests and Plotly visualizations.
    Produces an EDAReport consumed by the Orchestrator.
    """

    SKEW_THRESHOLD        = 1.0
    IQR_FACTOR            = 1.5
    HIGH_CARD_RATIO       = 0.5
    MISSING_THRESHOLD     = 0.05
    IMBALANCE_THRESHOLD   = 0.20
    MULTICOLLINEARITY_THR = 0.85

    def __init__(
        self,
        target_column: Optional[str] = None,
        dataset_name: str = "dataset",
        output_dir: str = "astats_output",
        verbose: bool = True,
    ):
        self.target_column = target_column
        self.dataset_name  = dataset_name
        self.output_dir    = output_dir
        self.verbose       = verbose
        os.makedirs(output_dir, exist_ok=True)

    # ── Main entry point ─────────────────────────────

    def run(self, df: pd.DataFrame) -> EDAReport:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console,
            transient=True,
        ) as progress:

            t1 = progress.add_task("Profiling dataset...", total=None)
            report = self._build_base_report(df)

            progress.update(t1, description="Profiling columns...")
            for col in df.columns:
                report.columns.append(self._profile_column(df[col]))

            progress.update(t1, description="Running statistical tests...")
            self._compute_correlations(report, df)
            self._set_flags(report, df)
            self._score_quality(report)
            self._recommend_approach(report)

            progress.update(t1, description="Generating ydata-profiling report...")
            self._generate_ydata_report(df)

            progress.update(t1, description="Generating Plotly visualizations...")
            self._generate_visualizations(df, report)

        self._print_rich_summary(report)
        return report

    # ── Base report ──────────────────────────────────

    def _build_base_report(self, df: pd.DataFrame) -> EDAReport:
        return EDAReport(
            dataset_name=self.dataset_name,
            n_rows=len(df),
            n_cols=len(df.columns),
            n_missing_cells=int(df.isnull().sum().sum()),
            pct_missing_cells=round(df.isnull().sum().sum() / df.size * 100, 2),
            duplicate_rows=int(df.duplicated().sum()),
            memory_usage_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 3),
        )

    # ── Column profiling ─────────────────────────────

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        n_missing    = int(series.isnull().sum())
        pct_missing  = round(n_missing / len(series) * 100, 2)
        n_unique     = int(series.nunique())
        card_level   = self._cardinality(series, n_unique)
        inferred     = self._infer_type(series, n_unique)

        prof = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            inferred_type=inferred,
            n_missing=n_missing,
            pct_missing=pct_missing,
            n_unique=n_unique,
            cardinality_level=card_level,
        )

        if inferred in ("numeric_continuous", "numeric_discrete"):
            clean = series.dropna().astype(float)
            if len(clean) > 0:
                prof.mean     = round(float(clean.mean()), 4)
                prof.std      = round(float(clean.std()), 4)
                prof.median   = round(float(clean.median()), 4)
                prof.min      = round(float(clean.min()), 4)
                prof.max      = round(float(clean.max()), 4)
                prof.skewness = round(float(clean.skew()), 4)
                prof.kurtosis = round(float(clean.kurt()), 4)
                prof.has_outliers, prof.outlier_count = self._detect_outliers(clean)
                prof.is_normal, prof.normality_p_value = self._test_normality(clean)
        else:
            vc = series.value_counts(normalize=True).head(5)
            prof.top_values = {str(k): round(float(v), 4) for k, v in vc.items()}

        return prof

    # ── Type inference ───────────────────────────────

    def _infer_type(self, series: pd.Series, n_unique: int) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) == 0:
                return "numeric_continuous"
            if n_unique / max(len(series), 1) > 0.95 and clean.min() >= 0 and clean.dtype == int:
                return "id"
            try:
                if (clean == clean.astype(int)).all() and n_unique <= 25:
                    return "numeric_discrete"
            except (ValueError, OverflowError):
                pass
            return "numeric_continuous"
        avg_len = series.dropna().astype(str).str.len().mean() if len(series.dropna()) > 0 else 0
        if avg_len > 60:
            return "text"
        if n_unique <= 2:
            return "boolean"
        return "categorical"

    # ── Statistical helpers ──────────────────────────

    def _cardinality(self, series: pd.Series, n_unique: int) -> str:
        ratio = n_unique / max(len(series), 1)
        if ratio > self.HIGH_CARD_RATIO:
            return "high"
        if n_unique > 15:
            return "medium"
        return "low"

    def _detect_outliers(self, clean: pd.Series):
        if len(clean) < 4:
            return False, 0
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - self.IQR_FACTOR * iqr
        hi  = q3 + self.IQR_FACTOR * iqr
        mask = (clean < lo) | (clean > hi)
        return bool(mask.any()), int(mask.sum())

    def _test_normality(self, clean: pd.Series):
        if len(clean) < 8:
            return None, None
        sample = clean.sample(min(5000, len(clean)), random_state=42)
        try:
            stat, p = stats.shapiro(sample[:500])
            return bool(p > 0.05), round(float(p), 6)
        except Exception:
            pass
        try:
            stat, p = stats.normaltest(sample)
            return bool(p > 0.05), round(float(p), 6)
        except Exception:
            return abs(float(clean.skew())) < 0.5, None

    # ── Correlations ─────────────────────────────────

    def _compute_correlations(self, report: EDAReport, df: pd.DataFrame):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return
        corr = df[num_cols].corr()
        report.correlation_matrix = corr.round(3).to_dict()
        mask = np.abs(corr.values) > self.MULTICOLLINEARITY_THR
        np.fill_diagonal(mask, False)
        report.has_multicollinearity = bool(mask.any())

    # ── Flags ────────────────────────────────────────

    def _set_flags(self, report: EDAReport, df: pd.DataFrame):
        report.has_missing_data     = report.pct_missing_cells > (self.MISSING_THRESHOLD * 100)
        report.has_high_cardinality = any(
            c.cardinality_level == "high" and c.inferred_type == "categorical"
            for c in report.columns
        )
        report.has_skewed_features  = any(
            c.skewness is not None and abs(c.skewness) > self.SKEW_THRESHOLD
            for c in report.columns
        )
        report.has_outliers = any(c.has_outliers for c in report.columns if c.has_outliers)

        if self.target_column and self.target_column in df.columns:
            target = df[self.target_column]
            if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
                report.target_type = "regression"
            else:
                report.target_type = "classification"
                freq = target.value_counts(normalize=True)
                if len(freq) > 0 and freq.min() < self.IMBALANCE_THRESHOLD:
                    report.has_class_imbalance = True

    # ── Data quality score ────────────────────────────

    def _score_quality(self, report: EDAReport):
        score = 100.0
        if report.has_missing_data:       score -= 15
        if report.has_outliers:           score -= 10
        if report.has_skewed_features:    score -= 10
        if report.has_class_imbalance:    score -= 15
        if report.has_high_cardinality:   score -= 10
        if report.has_multicollinearity:  score -= 10
        if report.duplicate_rows > 0:     score -= 5
        report.data_quality_score = max(0.0, round(score, 1))

    # ── A / B / C recommendation ──────────────────────

    def _recommend_approach(self, report: EDAReport):
        issues = sum([
            report.has_missing_data,
            report.has_skewed_features,
            report.has_outliers,
            report.has_class_imbalance,
            report.has_high_cardinality,
            report.has_multicollinearity,
        ])
        if issues == 0:
            report.recommended_approach = "A"
            report.approach_rationale = (
                "Data is clean and well-distributed. "
                "Standard parametric models with default settings are appropriate."
            )
        elif issues <= 2:
            report.recommended_approach = "B"
            report.approach_rationale = (
                f"{issues} data quality issue(s) detected ({self._issue_list(report)}). "
                "Robust preprocessing with regularized models recommended."
            )
        else:
            report.recommended_approach = "C"
            report.approach_rationale = (
                f"{issues} data quality issues detected ({self._issue_list(report)}). "
                "Heavy preprocessing with non-parametric or ensemble approach recommended."
            )

    def _issue_list(self, report: EDAReport) -> str:
        issues = []
        if report.has_missing_data:       issues.append("missing data")
        if report.has_skewed_features:    issues.append("skewed features")
        if report.has_outliers:           issues.append("outliers")
        if report.has_class_imbalance:    issues.append("class imbalance")
        if report.has_high_cardinality:   issues.append("high cardinality")
        if report.has_multicollinearity:  issues.append("multicollinearity")
        return ", ".join(issues)

    # ── ydata-profiling ──────────────────────────────

    def _generate_ydata_report(self, df: pd.DataFrame):
        try:
            profile = ProfileReport(
                df,
                title=f"AStats — {self.dataset_name}",
                explorative=True,
                minimal=False,
            )
            path = os.path.join(self.output_dir, f"{self.dataset_name}_profile.html")
            profile.to_file(path)
            console.print(f"[green]✓[/green] Deep profile report → [cyan]{path}[/cyan]")
        except Exception as e:
            console.print(f"[yellow]⚠ ydata-profiling skipped: {e}[/yellow]")

    # ── Plotly visualizations ─────────────────────────

    def _generate_visualizations(self, df: pd.DataFrame, report: EDAReport):
        report.target_column = self.target_column
        saved = generate_visualizations(df, report, self.output_dir, self.dataset_name)
        for name, path in saved.items():
            console.print(f"[green]✓[/green] Plot → [cyan]{path}[/cyan]")  
    # ── Rich terminal summary ─────────────────────────

    def _print_rich_summary(self, report: EDAReport):
        approach_color = {"A": "green", "B": "yellow", "C": "red"}.get(
            report.recommended_approach, "white"
        )

        console.print()
        console.print(Panel.fit(
            f"[bold]AStats — EDA Report[/bold]\n[dim]{report.dataset_name}[/dim]",
            style="bold blue"
        ))

        overview = Table(show_header=False, box=None, padding=(0, 2))
        overview.add_column(style="dim")
        overview.add_column(style="bold")
        overview.add_row("Rows",          f"{report.n_rows:,}")
        overview.add_row("Columns",       str(report.n_cols))
        overview.add_row("Missing cells", f"{report.pct_missing_cells}%")
        overview.add_row("Duplicates",    f"{report.duplicate_rows:,}")
        overview.add_row("Memory",        f"{report.memory_usage_mb} MB")
        q = report.data_quality_score
        q_color = "green" if q >= 80 else "yellow" if q >= 60 else "red"
        overview.add_row("Quality score", f"[{q_color}]{q}/100[/{q_color}]")
        console.print(overview)
        console.print()

        flags = Table(title="Data quality flags", show_header=True,
                      header_style="bold magenta")
        flags.add_column("Flag",   style="dim", width=28)
        flags.add_column("Status", width=10)

        def flag_row(label, val):
            status = "[red]⚠ YES[/red]" if val else "[green]✓ No[/green]"
            flags.add_row(label, status)

        flag_row("Missing data",      report.has_missing_data)
        flag_row("Skewed features",   report.has_skewed_features)
        flag_row("Outliers",          report.has_outliers)
        flag_row("Class imbalance",   report.has_class_imbalance)
        flag_row("High cardinality",  report.has_high_cardinality)
        flag_row("Multicollinearity", report.has_multicollinearity)
        console.print(flags)
        console.print()

        console.print(Panel(
            f"[bold {approach_color}]Approach {report.recommended_approach}[/bold {approach_color}]\n"
            f"[dim]{report.approach_rationale}[/dim]",
            title="[bold]Orchestrator recommendation[/bold]",
            border_style=approach_color,
        ))

        col_table = Table(title="Column profiles", show_header=True,
                          header_style="bold cyan")
        col_table.add_column("Column",   style="bold", max_width=22)
        col_table.add_column("Type",     style="dim",  max_width=20)
        col_table.add_column("Missing",  justify="right", max_width=10)
        col_table.add_column("Unique",   justify="right", max_width=8)
        col_table.add_column("Skew",     justify="right", max_width=8)
        col_table.add_column("Normal?",  justify="center", max_width=10)
        col_table.add_column("Outliers", justify="right",  max_width=10)

        for c in report.columns:
            skew_str = f"{c.skewness:.2f}" if c.skewness is not None else "—"
            normal_str = (
                "[green]Yes[/green]" if c.is_normal is True
                else "[red]No[/red]" if c.is_normal is False
                else "—"
            )
            outlier_str = (
                f"[red]{c.outlier_count}[/red]" if c.has_outliers
                else "[green]0[/green]"          if c.has_outliers is False
                else "—"
            )
            miss_str = (
                f"[red]{c.pct_missing}%[/red]"    if c.pct_missing > 5
                else f"[yellow]{c.pct_missing}%[/yellow]" if c.pct_missing > 0
                else "[green]0%[/green]"
            )
            col_table.add_row(
                c.name, c.inferred_type, miss_str,
                str(c.n_unique), skew_str, normal_str, outlier_str
            )

        console.print(col_table)
        console.print(f"\n[dim]Outputs saved to:[/dim] [cyan]{self.output_dir}/[/cyan]\n")

    # ── Save JSON ─────────────────────────────────────

    def save_report(self, report: EDAReport) -> str:
        path = os.path.join(self.output_dir, f"{self.dataset_name}_eda_report.json")
        with open(path, "w") as f:
            def convert(obj):
                if isinstance(obj, (bool, int, float, str)) or obj is None:
                    return obj
                if isinstance(obj, np.bool_):    return bool(obj)
                if isinstance(obj, np.integer):  return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, dict):        return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):        return [convert(i) for i in obj]
                return str(obj)

            json.dump(convert(asdict(report)), f, indent=2)

        console.print(f"[green]✓[/green] EDA report (JSON) → [cyan]{path}[/cyan]")
        return path


# ─────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris

    console.rule("[bold blue]Test 1 — Breast Cancer (expect: Approach A)")
    data = load_breast_cancer()
    df1 = pd.DataFrame(data.data, columns=data.feature_names)
    df1["target"] = data.target
    agent1 = EDAAgent(target_column="target", dataset_name="breast_cancer",
                      output_dir="astats_output/test1")
    report1 = agent1.run(df1)
    agent1.save_report(report1)

    console.rule("[bold blue]Test 2 — Iris (expect: Approach A/B)")
    data2 = load_iris()
    df2 = pd.DataFrame(data2.data, columns=data2.feature_names)
    df2["species"] = data2.target
    agent2 = EDAAgent(target_column="species", dataset_name="iris",
                      output_dir="astats_output/test2")
    report2 = agent2.run(df2)
    agent2.save_report(report2)

    console.rule("[bold blue]Test 3 — Messy synthetic (expect: Approach C)")
    np.random.seed(42)
    n = 400
    df3 = pd.DataFrame({
        "age":       np.random.exponential(30, n),
        "income":    np.append(
                         np.random.normal(50000, 12000, n - 15),
                         np.random.normal(900000, 50000, 15)
                     ),
        "score":     np.random.beta(0.4, 0.4, n) * 100,
        "city":      np.random.choice(
                         ["NYC","LA","CHI","HOU","PHX","PHI","SA","SD","DAL","SJ",
                          "AUS","JAX","FW","COL","CHA","IND","SF","SEA","DEN","OKC"], n
                     ),
        "category":  np.random.choice(list("ABCDEFGHIJ"), n),
        "feature_x": np.random.normal(0, 1, n),
        "label":     np.random.choice([0, 1], n, p=[0.91, 0.09]),
    })
    # Introduce multicollinearity
    df3["feature_y"] = df3["feature_x"] * 0.95 + np.random.normal(0, 0.1, n)
    # Introduce missing values
    for col in ["age", "income", "score"]:
        idx = np.random.choice(n, int(n * 0.12), replace=False)
        df3.loc[idx, col] = np.nan

    agent3 = EDAAgent(target_column="label", dataset_name="synthetic_messy",
                      output_dir="astats_output/test3")
    report3 = agent3.run(df3)
    agent3.save_report(report3)

    console.rule("[bold green]All tests complete")
    console.print(
        f"\n[bold]Approach summary:[/bold]\n"
        f"  breast_cancer   → [green]{report1.recommended_approach}[/green]\n"
        f"  iris            → [yellow]{report2.recommended_approach}[/yellow]\n"
        f"  synthetic_messy → [red]{report3.recommended_approach}[/red]\n"
    )
