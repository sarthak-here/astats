"""
AStats — Orchestrator + LLM Core + CLI (Day 2)
GSoC 2026 Project #33
=====================================================
The Orchestrator is the brain of AStats.
- Reads EDAReport JSON from Day 1
- Uses Llama3.1:8b (via Ollama) as the reasoning core
- Routes to Approach A / B / C
- Logs every decision with timestamp into an audit trail
- Exposes a CLI using Typer + Rich
- Produces a WorkflowState consumed by Modeling Agent (Day 3)

Run:
    python orchestrator.py run --data path/to/data.csv --target label
    python orchestrator.py run --eda-report path/to/eda_report.json
    python orchestrator.py history --state path/to/workflow_state.json
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import json
import time
import datetime
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax

# Import Day 1 EDA Agent
import sys
sys.path.append(str(Path(__file__).parent))
from EDA_Agent import EDAAgent, EDAReport

console = Console()
app = typer.Typer(
    name="astats",
    help="AStats — Agentic AI for applied statistical workflows",
    add_completion=False,
)


# ─────────────────────────────────────────────────────
# Enums & Data Structures
# ─────────────────────────────────────────────────────

class Approach(str, Enum):
    A = "A"  # Standard / clean data
    B = "B"  # Robust / handles messiness
    C = "C"  # Ensemble / heavy preprocessing


class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETE  = "complete"
    SKIPPED   = "skipped"
    FAILED    = "failed"


@dataclass
class AuditEntry:
    """Single entry in the decision audit trail."""
    timestamp: str
    step: str
    decision: str
    rationale: str
    llm_used: bool = False
    llm_model: str = ""
    duration_sec: float = 0.0


@dataclass
class PreprocessingPlan:
    """Concrete preprocessing steps decided by the Orchestrator."""
    handle_missing: str = "none"       # none / median / mean / knn / drop
    handle_outliers: str = "none"      # none / winsorize / iqr_clip / isolation_forest
    handle_skewness: str = "none"      # none / log / sqrt / boxcox / yeo_johnson
    handle_imbalance: str = "none"     # none / smote / class_weight / undersample
    handle_multicollinearity: str = "none"  # none / drop_vif / pca
    scaling: str = "standard"         # standard / robust / minmax / none
    encoding: str = "onehot"          # onehot / ordinal / target


@dataclass
class WorkflowState:
    """
    Central state object that flows through the entire AStats pipeline.
    Every agent reads from and writes to this object.
    """
    # Identity
    run_id: str
    dataset_name: str
    created_at: str
    target_column: Optional[str]

    # EDA results (from Day 1)
    eda_report_path: str = ""
    eda_summary: dict = field(default_factory=dict)

    # Orchestrator decisions
    approach: str = "A"
    preprocessing_plan: dict = field(default_factory=dict)
    llm_reasoning: str = ""

    # Pipeline steps
    steps_completed: list = field(default_factory=list)
    steps_pending: list = field(default_factory=list)

    # Audit trail — every decision logged here
    audit_trail: list = field(default_factory=list)

    # Outputs from each agent (filled in Days 3-6)
    modeling_results: dict = field(default_factory=dict)
    inference_results: dict = field(default_factory=dict)
    explanation_results: dict = field(default_factory=dict)
    final_report_path: str = ""

    # Meta
    output_dir: str = "astats_output"
    status: str = "initialised"


# ─────────────────────────────────────────────────────
# LLM Core
# ─────────────────────────────────────────────────────

class LLMCore:
    """
    Thin wrapper around Ollama's local API.
    Falls back gracefully if Ollama is not running.
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    DEFAULT_MODEL = "llama3.1:8b"

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 120):
        self.model   = model
        self.timeout = timeout
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def ask(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the response."""
        if not self.available:
            return "[LLM unavailable — rule-based fallback used]"

        payload = {
            "model":  self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.2,   # Low temp for reproducible decisions
                "num_predict": 512,
            }
        }
        try:
            r = requests.post(self.OLLAMA_URL, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            return f"[LLM error: {e}]"

    def ask_for_json(self, prompt: str, system: str = "") -> dict:
        """Ask LLM to return JSON. Strips markdown fences."""
        raw = self.ask(prompt, system)
        # Strip ```json fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"raw_response": raw}


# ─────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────

class Orchestrator:
    """
    Central decision-making agent.

    Assess → Select → Validate cycle:
    1. Assess: reads EDA flags
    2. Select: chooses Approach A/B/C and preprocessing plan
       - Uses LLM reasoning when available
       - Falls back to rule-based logic otherwise
    3. Validate: confirms plan is coherent
    4. Logs every decision to audit trail
    """

    SYSTEM_PROMPT = """You are AStats, an expert statistical workflow orchestrator.
You analyze data quality reports and decide the best statistical approach.
You are precise, reproducible, and always explain your reasoning.
When asked for JSON, respond ONLY with valid JSON and nothing else."""

    PIPELINE_STEPS = [
        "eda",
        "preprocessing",
        "modeling",
        "inference",
        "explanation",
        "report",
    ]

    def __init__(
        self,
        llm_model: str = "llama3.1:8b",
        output_dir: str = "astats_output",
        interactive: bool = True,
    ):
        self.llm         = LLMCore(model=llm_model)
        self.output_dir  = output_dir
        self.interactive = interactive
        os.makedirs(output_dir, exist_ok=True)

        if self.llm.available:
            console.print(f"[green]✓[/green] LLM online — [cyan]{llm_model}[/cyan]")
        else:
            console.print("[yellow]⚠ Ollama not running — rule-based fallback active[/yellow]")

    # ── Public entry point ────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        target_column: Optional[str] = None,
        eda_report: Optional[EDAReport] = None,
    ) -> WorkflowState:

        run_id   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir  = os.path.join(self.output_dir, dataset_name, run_id)
        os.makedirs(out_dir, exist_ok=True)

        state = WorkflowState(
            run_id=run_id,
            dataset_name=dataset_name,
            created_at=datetime.datetime.now().isoformat(),
            target_column=target_column,
            output_dir=out_dir,
            steps_pending=list(self.PIPELINE_STEPS),
            status="running",
        )

        console.print()
        console.print(Rule(f"[bold blue]AStats Orchestrator — {dataset_name}[/bold blue]"))

        # Step 1: EDA (run or load)
        eda_report = self._step_eda(state, df, dataset_name, target_column, eda_report, out_dir)

        # Step 2: LLM Reasoning
        self._step_llm_reasoning(state, eda_report)

        # Step 3: Approach selection
        self._step_select_approach(state, eda_report)

        # Step 4: Preprocessing plan
        self._step_preprocessing_plan(state, eda_report)

        # Step 5: Validate
        self._step_validate(state)

        # Save state
        state.status = "orchestrated"
        state.steps_completed.append("eda")
        state.steps_completed.append("preprocessing")
        state.steps_pending = [s for s in self.PIPELINE_STEPS
                                if s not in state.steps_completed]

        self._save_state(state)
        self._print_summary(state)

        return state

    # ── Step: EDA ────────────────────────────────────

    def _step_eda(
        self, state, df, dataset_name, target_column, eda_report, out_dir
    ) -> EDAReport:
        t0 = time.time()

        if eda_report is None:
            console.print("\n[bold]Step 1 — Running EDA Agent...[/bold]")
            agent = EDAAgent(
                target_column=target_column,
                dataset_name=dataset_name,
                output_dir=out_dir,
                verbose=False,
            )
            eda_report = agent.run(df)
            agent.save_report(eda_report)
            report_path = os.path.join(out_dir, f"{dataset_name}_eda_report.json")
            state.eda_report_path = report_path
        else:
            console.print("\n[bold]Step 1 — Using provided EDA report[/bold]")

        # Store key flags in state
        state.eda_summary = {
            "n_rows":               eda_report.n_rows,
            "n_cols":               eda_report.n_cols,
            "quality_score":        eda_report.data_quality_score,
            "target_type":          eda_report.target_type,
            "has_missing":          eda_report.has_missing_data,
            "has_outliers":         eda_report.has_outliers,
            "has_skew":             eda_report.has_skewed_features,
            "has_imbalance":        eda_report.has_class_imbalance,
            "has_multicollinearity":eda_report.has_multicollinearity,
            "eda_approach":         eda_report.recommended_approach,
        }

        self._log(state, "eda", "EDA complete",
                  f"Quality score: {eda_report.data_quality_score}/100. "
                  f"EDA recommended approach {eda_report.recommended_approach}.",
                  duration=time.time() - t0)

        console.print(f"  [green]✓[/green] EDA complete — quality score: "
                      f"[bold]{eda_report.data_quality_score}/100[/bold]")
        return eda_report

    # ── Step: LLM Reasoning ───────────────────────────

    def _step_llm_reasoning(self, state: WorkflowState, eda: EDAReport):
        if not self.llm.available:
            state.llm_reasoning = "LLM unavailable — rule-based decisions used."
            return

        console.print("\n[bold]Step 2 — LLM reasoning...[/bold]")
        t0 = time.time()

        prompt = f"""Analyze this dataset profile and reason about the best statistical workflow:

Dataset: {state.dataset_name}
Rows: {eda.n_rows} | Columns: {eda.n_cols}
Target type: {eda.target_type}
Data quality score: {eda.data_quality_score}/100

Issues detected:
- Missing data: {eda.has_missing_data} ({eda.pct_missing_cells}%)
- Skewed features: {eda.has_skewed_features}
- Outliers: {eda.has_outliers}
- Class imbalance: {eda.has_class_imbalance}
- High cardinality: {eda.has_high_cardinality}
- Multicollinearity: {eda.has_multicollinearity}

Column summary:
{self._format_columns(eda)}

Based on this profile:
1. What preprocessing steps are most critical?
2. What modeling approach (A=standard parametric, B=robust/regularized, C=ensemble/non-parametric) fits best?
3. What are the key statistical risks to watch for?

Be concise and specific."""

        reasoning = self.llm.ask(prompt, system=self.SYSTEM_PROMPT)
        state.llm_reasoning = reasoning

        self._log(state, "llm_reasoning",
                  f"LLM analysis complete",
                  reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
                  llm_used=True, llm_model=self.llm.model,
                  duration=time.time() - t0)

        console.print(f"  [green]✓[/green] LLM reasoning complete "
                      f"[dim]({round(time.time()-t0, 1)}s)[/dim]")

    # ── Step: Approach Selection ──────────────────────

    def _step_select_approach(self, state: WorkflowState, eda: EDAReport):
        console.print("\n[bold]Step 3 — Selecting approach...[/bold]")
        t0 = time.time()

        # If LLM available, ask it to confirm/override the EDA recommendation
        if self.llm.available:
            prompt = f"""Given this analysis:
{state.llm_reasoning}

And the EDA recommended approach: {eda.recommended_approach}

Choose exactly one approach:
A = Standard parametric (clean data, normal distributions, no major issues)
B = Robust/regularized (1-2 issues: outliers OR missing OR skew)
C = Ensemble/non-parametric (3+ issues, heavy preprocessing needed)

Respond with ONLY valid JSON like this:
{{"approach": "B", "confidence": "high", "rationale": "one sentence reason"}}"""

            result = self.llm.ask_for_json(prompt, system=self.SYSTEM_PROMPT)
            approach = result.get("approach", eda.recommended_approach)
            rationale = result.get("rationale", eda.approach_rationale)
            confidence = result.get("confidence", "medium")

            # Validate LLM didn't hallucinate
            if approach not in ("A", "B", "C"):
                approach = eda.recommended_approach
                rationale = f"LLM returned invalid approach, fell back to EDA recommendation."

        else:
            approach  = eda.recommended_approach
            rationale = eda.approach_rationale
            confidence = "high"

        state.approach = approach
        approach_color = {"A": "green", "B": "yellow", "C": "red"}.get(approach, "white")

        self._log(state, "approach_selection",
                  f"Approach {approach} selected (confidence: {confidence})",
                  rationale, llm_used=self.llm.available,
                  llm_model=self.llm.model if self.llm.available else "",
                  duration=time.time() - t0)

        console.print(
            f"  [green]✓[/green] Approach [{approach_color}][bold]{approach}[/bold][/{approach_color}] "
            f"selected [dim](confidence: {confidence})[/dim]"
        )

    # ── Step: Preprocessing Plan ──────────────────────

    def _step_preprocessing_plan(self, state: WorkflowState, eda: EDAReport):
        console.print("\n[bold]Step 4 — Building preprocessing plan...[/bold]")
        t0 = time.time()

        if self.llm.available:
            prompt = f"""For Approach {state.approach} on this dataset:
- Missing data: {eda.has_missing_data} ({eda.pct_missing_cells}%)
- Outliers: {eda.has_outliers}
- Skewed features: {eda.has_skewed_features}
- Class imbalance: {eda.has_class_imbalance}
- Multicollinearity: {eda.has_multicollinearity}
- Target type: {eda.target_type}
- Rows: {eda.n_rows}

Choose the best preprocessing strategy. Respond ONLY with valid JSON:
{{
  "handle_missing": "median",
  "handle_outliers": "winsorize",
  "handle_skewness": "yeo_johnson",
  "handle_imbalance": "smote",
  "handle_multicollinearity": "drop_vif",
  "scaling": "robust",
  "encoding": "onehot"
}}

Options:
- handle_missing: none / median / mean / knn / drop
- handle_outliers: none / winsorize / iqr_clip / isolation_forest
- handle_skewness: none / log / sqrt / boxcox / yeo_johnson
- handle_imbalance: none / smote / class_weight / undersample
- handle_multicollinearity: none / drop_vif / pca
- scaling: standard / robust / minmax / none
- encoding: onehot / ordinal / target"""

            plan_dict = self.llm.ask_for_json(prompt, system=self.SYSTEM_PROMPT)

            # Validate keys — fall back to rule-based for any missing
            valid_plan = self._rule_based_plan(eda, approach=state.approach)
            for key in asdict(valid_plan).keys():
                if key in plan_dict and isinstance(plan_dict[key], str):
                    setattr(valid_plan, key, plan_dict[key])
            plan = valid_plan
        else:
            plan = self._rule_based_plan(eda, approach=state.approach)

        state.preprocessing_plan = asdict(plan)

        self._log(state, "preprocessing_plan",
                  "Preprocessing plan finalised",
                  json.dumps(asdict(plan), indent=2),
                  llm_used=self.llm.available,
                  llm_model=self.llm.model if self.llm.available else "",
                  duration=time.time() - t0)

        console.print(f"  [green]✓[/green] Preprocessing plan ready")

    def _rule_based_plan(self, eda: EDAReport, approach: str = "C") -> PreprocessingPlan:
        """Fallback rule-based preprocessing plan."""
        plan = PreprocessingPlan()
        if eda.has_missing_data:
            plan.handle_missing = "knn" if eda.n_rows > 500 else "median"
        if eda.has_outliers:
            plan.handle_outliers = "winsorize" if approach == "B" else "isolation_forest"
        if eda.has_skewed_features:
            plan.handle_skewness = "yeo_johnson"
        if eda.has_class_imbalance:
            plan.handle_imbalance = "smote" if eda.n_rows > 200 else "class_weight"
        if eda.has_multicollinearity:
            plan.handle_multicollinearity = "drop_vif"
        plan.scaling = "robust" if eda.has_outliers else "standard"
        return plan

    # ── Step: Validate ────────────────────────────────

    def _step_validate(self, state: WorkflowState):
        console.print("\n[bold]Step 5 — Validating plan...[/bold]")
        t0 = time.time()

        issues = []
        plan = state.preprocessing_plan

        # Coherence checks
        if state.eda_summary.get("has_missing") and plan.get("handle_missing") == "none":
            issues.append("Missing data detected but handle_missing is 'none'")
        if state.eda_summary.get("has_imbalance") and plan.get("handle_imbalance") == "none":
            issues.append("Class imbalance detected but handle_imbalance is 'none'")

        if issues:
            console.print(f"  [yellow]⚠ {len(issues)} validation warning(s):[/yellow]")
            for issue in issues:
                console.print(f"    [yellow]• {issue}[/yellow]")
            self._log(state, "validation",
                      f"Validation passed with {len(issues)} warnings",
                      "; ".join(issues), duration=time.time() - t0)
        else:
            console.print(f"  [green]✓[/green] Plan validated — no issues")
            self._log(state, "validation", "Validation passed",
                      "All checks passed.", duration=time.time() - t0)

    # ── Helpers ───────────────────────────────────────

    def _format_columns(self, eda: EDAReport) -> str:
        lines = []
        for c in eda.columns[:10]:  # First 10 cols only to keep prompt short
            if c.skewness is not None:
                lines.append(
                    f"  {c.name}: {c.inferred_type}, skew={c.skewness}, "
                    f"outliers={c.outlier_count}, missing={c.pct_missing}%"
                )
        return "\n".join(lines)

    def _log(
        self, state: WorkflowState, step: str,
        decision: str, rationale: str,
        llm_used: bool = False, llm_model: str = "",
        duration: float = 0.0
    ):
        entry = AuditEntry(
            timestamp=datetime.datetime.now().isoformat(),
            step=step,
            decision=decision,
            rationale=rationale,
            llm_used=llm_used,
            llm_model=llm_model,
            duration_sec=round(duration, 3),
        )
        state.audit_trail.append(asdict(entry))

    def _save_state(self, state: WorkflowState):
        path = os.path.join(state.output_dir, "workflow_state.json")

        def convert(obj):
            if isinstance(obj, (bool, int, float, str)) or obj is None: return obj
            if isinstance(obj, np.bool_):    return bool(obj)
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, dict):        return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):        return [convert(i) for i in obj]
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(convert(asdict(state)), f, indent=2)
        console.print(f"\n[green]✓[/green] Workflow state saved → [cyan]{path}[/cyan]")

    def _print_summary(self, state: WorkflowState):
        approach_color = {"A": "green", "B": "yellow", "C": "red"}.get(state.approach, "white")

        console.print()
        console.print(Rule("[bold]Orchestrator Summary[/bold]"))

        # Decision table
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Step",     style="dim",  width=28)
        table.add_column("Decision", style="bold", width=40)
        table.add_column("LLM",      justify="center", width=6)
        table.add_column("Time",     justify="right",  width=8)

        for entry in state.audit_trail:
            llm_str  = "[green]Y[/green]" if entry["llm_used"] else "[dim]N[/dim]"
            time_str = f"{entry['duration_sec']}s"
            table.add_row(
                entry["step"],
                entry["decision"][:38] + ".." if len(entry["decision"]) > 40
                else entry["decision"],
                llm_str, time_str
            )
        console.print(table)

        # Final recommendation
        console.print()
        console.print(Panel(
            f"[bold {approach_color}]Approach {state.approach}[/bold {approach_color}]\n\n"
            f"[bold]Preprocessing plan:[/bold]\n" +
            "\n".join(
                f"  [dim]{k.replace('_', ' ').title()}:[/dim] {v}"
                for k, v in state.preprocessing_plan.items()
                if v != "none"
            ) +
            f"\n\n[dim]Run ID: {state.run_id}[/dim]\n"
            f"[dim]Output: {state.output_dir}[/dim]",
            title="[bold]Final workflow plan[/bold]",
            border_style=approach_color,
        ))


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

@app.command("run")
def cli_run(
    data: Optional[Path] = typer.Option(None, "--data", "-d",
        help="Path to CSV dataset"),
    target: Optional[str] = typer.Option(None, "--target", "-t",
        help="Target column name"),
    eda_report: Optional[Path] = typer.Option(None, "--eda-report",
        help="Path to existing EDA report JSON (skips re-running EDA)"),
    dataset_name: Optional[str] = typer.Option(None, "--name", "-n",
        help="Dataset name (defaults to filename)"),
    llm_model: str = typer.Option("llama3.1:8b", "--model",
        help="Ollama model to use"),
    output_dir: str = typer.Option("astats_output", "--output", "-o",
        help="Output directory"),
):
    """Run the full AStats orchestration pipeline on a dataset."""

    console.print(Panel.fit(
        "[bold blue]AStats[/bold blue] — Agentic Statistical Workflow\n"
        "[dim]GSoC 2026 Project #33[/dim]",
    ))

    # Load data
    if data is not None:
        if not data.exists():
            console.print(f"[red]Error: file not found: {data}[/red]")
            raise typer.Exit(1)
        df = pd.read_csv(data)
        name = dataset_name or data.stem
        console.print(f"[green]✓[/green] Loaded [cyan]{data}[/cyan] "
                      f"— {len(df):,} rows × {len(df.columns)} cols")

    elif eda_report is not None:
        # Load from existing EDA JSON — recreate minimal df
        if not eda_report.exists():
            console.print(f"[red]Error: EDA report not found: {eda_report}[/red]")
            raise typer.Exit(1)
        with open(eda_report) as f:
            report_data = json.load(f)
        name = dataset_name or report_data.get("dataset_name", "dataset")
        console.print(f"[green]✓[/green] Loaded EDA report: [cyan]{eda_report}[/cyan]")
        # We still need a df for the Orchestrator — warn user
        console.print("[yellow]⚠ No CSV provided — EDA step will be skipped, "
                      "using cached report.[/yellow]")
        df = pd.DataFrame()  # Empty — EDA already done

    else:
        # Demo mode — use sklearn breast cancer
        console.print("[dim]No data provided — running demo on breast cancer dataset[/dim]")
        from sklearn.datasets import load_breast_cancer
        data_sk = load_breast_cancer()
        df = pd.DataFrame(data_sk.data, columns=data_sk.feature_names)
        df["target"] = data_sk.target
        target = target or "target"
        name = dataset_name or "breast_cancer_demo"

    orchestrator = Orchestrator(
        llm_model=llm_model,
        output_dir=output_dir,
    )

    state = orchestrator.run(
        df=df,
        dataset_name=name,
        target_column=target,
    )

    console.print(f"\n[bold green]Done![/bold green] "
                  f"Workflow state saved to [cyan]{state.output_dir}[/cyan]")


@app.command("history")
def cli_history(
    state_path: Path = typer.Argument(..., help="Path to workflow_state.json"),
):
    """Display the full audit trail for a completed run."""
    if not state_path.exists():
        console.print(f"[red]File not found: {state_path}[/red]")
        raise typer.Exit(1)

    with open(state_path) as f:
        state = json.load(f)

    console.print(Panel.fit(
        f"[bold]Audit Trail[/bold]\n"
        f"[dim]Run ID: {state['run_id']}  |  "
        f"Dataset: {state['dataset_name']}  |  "
        f"Approach: {state['approach']}[/dim]"
    ))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Time",     style="dim",  width=12)
    table.add_column("Step",     width=22)
    table.add_column("Decision", width=36)
    table.add_column("LLM",      justify="center", width=5)
    table.add_column("Secs",     justify="right",  width=6)

    for entry in state.get("audit_trail", []):
        ts   = entry["timestamp"][11:19]  # HH:MM:SS
        llm  = "[green]Y[/green]" if entry["llm_used"] else "[dim]N[/dim]"
        table.add_row(
            ts, entry["step"],
            entry["decision"][:34] + ".." if len(entry["decision"]) > 36
            else entry["decision"],
            llm, str(entry["duration_sec"])
        )

    console.print(table)

    # Show LLM reasoning if present
    if state.get("llm_reasoning"):
        console.print()
        console.print(Panel(
            state["llm_reasoning"],
            title="[bold]LLM Reasoning[/bold]",
            border_style="blue",
        ))


@app.command("status")
def cli_status(
    state_path: Path = typer.Argument(..., help="Path to workflow_state.json"),
):
    """Show current pipeline status for a run."""
    if not state_path.exists():
        console.print(f"[red]File not found: {state_path}[/red]")
        raise typer.Exit(1)

    with open(state_path) as f:
        state = json.load(f)

    console.print(Panel.fit(f"[bold]Pipeline Status — {state['dataset_name']}[/bold]"))

    all_steps = ["eda", "preprocessing", "modeling", "inference", "explanation", "report"]
    completed = state.get("steps_completed", [])
    pending   = state.get("steps_pending", [])

    for step in all_steps:
        if step in completed:
            console.print(f"  [green]✓[/green] {step}")
        elif step in pending:
            console.print(f"  [dim]○ {step} (pending)[/dim]")
        else:
            console.print(f"  [yellow]◑ {step} (running)[/yellow]")


# ─────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # If run directly (not via CLI), run the demo
    if len(sys.argv) == 1:
        console.print("[dim]Running in demo mode — "
                      "use 'python orchestrator.py --help' for CLI usage[/dim]\n")

        from sklearn.datasets import load_breast_cancer
        import numpy as np

        # Test with messy synthetic data
        np.random.seed(42)
        n = 400
        df = pd.DataFrame({
            "age":       np.random.exponential(30, n),
            "income":    np.append(np.random.normal(50000, 12000, n-15),
                                   np.random.normal(900000, 50000, 15)),
            "score":     np.random.beta(0.4, 0.4, n) * 100,
            "city":      np.random.choice(["NYC","LA","CHI","HOU","PHX"], n),
            "feature_x": np.random.normal(0, 1, n),
            "label":     np.random.choice([0, 1], n, p=[0.91, 0.09]),
        })
        df["feature_y"] = df["feature_x"] * 0.95 + np.random.normal(0, 0.1, n)
        for col in ["age", "income"]:
            idx = np.random.choice(n, int(n * 0.12), replace=False)
            df.loc[idx, col] = np.nan

        orch = Orchestrator(output_dir="astats_output/orchestrator_demo")
        state = orch.run(df, dataset_name="demo_messy", target_column="label")
    else:
        app()
