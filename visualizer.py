"""
AStats — Visualizer (improved)
GSoC 2026 Project #33
=====================================================
Drop-in replacement for the visualization section
in eda_agent.py. Call generate_visualizations(df, report, output_dir, dataset_name)

Improvements over v1:
- Each column gets its own subplot and scale (no crushed axes)
- Color-coded by data quality flag (outlier = red, normal = green, etc.)
- Distributions show KDE curve on top of histogram
- Box plots one per column, properly scaled
- Correlation heatmap annotated and centered
- Missing data bar chart (cleaner than heatmap)
- Target distribution with imbalance annotation
- Quality gauge with colored zones
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats


def generate_visualizations(df, report, output_dir, dataset_name):
    """
    Generate all improved visualizations.
    Returns dict of {name: filepath}.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    num_cols = [c.name for c in report.columns
                if c.inferred_type in ("numeric_continuous", "numeric_discrete")
                and c.name != report.target_type]
    cat_cols = [c.name for c in report.columns if c.inferred_type == "categorical"]

    # 1. Distribution grid (histogram + KDE per column)
    path = _distribution_grid(df, report, num_cols, output_dir, dataset_name)
    if path: saved["distributions"] = path

    # 2. Box plots — one per column, own scale
    path = _box_plots(df, report, num_cols, output_dir, dataset_name)
    if path: saved["boxplots"] = path

    # 3. Correlation heatmap
    path = _correlation_heatmap(df, num_cols, output_dir, dataset_name)
    if path: saved["correlation"] = path

    # 4. Missing data bar chart
    path = _missing_bar(df, report, output_dir, dataset_name)
    if path: saved["missing"] = path

    # 5. Target distribution
    if report.target_column and report.target_column in df.columns:
        path = _target_distribution(df, report, output_dir, dataset_name)
        if path: saved["target_dist"] = path

    # 6. Quality gauge
    path = _quality_gauge(report, output_dir, dataset_name)
    if path: saved["quality_gauge"] = path

    # 7. Skewness bar chart
    path = _skewness_chart(report, num_cols, output_dir, dataset_name)
    if path: saved["skewness"] = path

    return saved


# ─────────────────────────────────────────────────────
# 1. Distribution grid
# ─────────────────────────────────────────────────────

def _distribution_grid(df, report, num_cols, output_dir, dataset_name):
    if not num_cols:
        return None

    cols_to_plot = num_cols[:9]
    n = len(cols_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=cols_to_plot,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Color each column based on its flags
    col_profiles = {c.name: c for c in report.columns}

    for i, col in enumerate(cols_to_plot):
        row, col_idx = divmod(i, ncols)
        row += 1
        col_idx += 1

        clean = df[col].dropna()
        profile = col_profiles.get(col)

        # Color based on issues
        if profile and profile.has_outliers:
            color = "#e74c3c"   # red — has outliers
        elif profile and profile.is_normal is True:
            color = "#2ecc71"   # green — normal
        elif profile and profile.skewness and abs(profile.skewness) > 1:
            color = "#f39c12"   # orange — skewed
        else:
            color = "#5B8FF9"   # blue — default

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=clean,
                name=col,
                showlegend=False,
                marker_color=color,
                opacity=0.75,
                nbinsx=30,
            ),
            row=row, col=col_idx
        )

        # KDE overlay
        try:
            kde = scipy_stats.gaussian_kde(clean)
            x_range = np.linspace(clean.min(), clean.max(), 200)
            kde_vals = kde(x_range)
            # Scale KDE to histogram height
            hist_counts, hist_bins = np.histogram(clean, bins=30)
            scale = hist_counts.max() / kde_vals.max() if kde_vals.max() > 0 else 1
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_vals * scale,
                    mode="lines",
                    line=dict(color=color, width=2, dash="solid"),
                    opacity=0.9,
                    showlegend=False,
                    name=f"{col} KDE",
                ),
                row=row, col=col_idx
            )
        except Exception:
            pass

        # Show skew in subplot title
        if profile and profile.skewness is not None:
            skew_val = profile.skewness
            skew_str = f"{col} (skew: {skew_val:.2f})"
            fig.layout.annotations[i].text = skew_str

    fig.update_layout(
        title=dict(
            text="<b>Feature distributions</b><br>"
                 "<sup><span style='color:#e74c3c'>■</span> has outliers  "
                 "<span style='color:#2ecc71'>■</span> normal  "
                 "<span style='color:#f39c12'>■</span> skewed  "
                 "<span style='color:#5B8FF9'>■</span> other</sup>",
            x=0.01,
        ),
        height=280 * nrows + 80,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", zeroline=False)

    path = os.path.join(output_dir, f"{dataset_name}_distributions.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────────────
# 2. Box plots — one per column, own scale
# ─────────────────────────────────────────────────────

def _box_plots(df, report, num_cols, output_dir, dataset_name):
    if not num_cols:
        return None

    cols_to_plot = num_cols[:9]
    n = len(cols_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    col_profiles = {c.name: c for c in report.columns}

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=cols_to_plot,
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    for i, col in enumerate(cols_to_plot):
        row, col_idx = divmod(i, ncols)
        row += 1
        col_idx += 1

        clean = df[col].dropna()
        profile = col_profiles.get(col)
        has_outliers = profile.has_outliers if profile else False
        n_outliers = profile.outlier_count if profile else 0

        box_color   = "#e74c3c" if has_outliers else "#5B8FF9"
        point_color = "#c0392b" if has_outliers else "#2980b9"

        fig.add_trace(
            go.Box(
                y=clean,
                name=col,
                showlegend=False,
                boxpoints="outliers",
                marker=dict(
                    color=point_color,
                    size=5,
                    opacity=0.7,
                ),
                line=dict(color=box_color, width=1.5),
                fillcolor=box_color,
                opacity=0.4,
                # Show stats on hover
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "Q1: %{q1:.2f}<br>"
                    "Median: %{median:.2f}<br>"
                    "Q3: %{q3:.2f}<br>"
                    "Mean: %{mean:.2f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row, col=col_idx
        )

        # Show outlier count in subplot title
        if has_outliers and n_outliers:
            fig.layout.annotations[i].text = f"{col} (⚠ {n_outliers} outliers)"

    fig.update_layout(
        title=dict(
            text="<b>Box plots — each column on its own scale</b><br>"
                 "<sup><span style='color:#e74c3c'>■</span> has outliers  "
                 "<span style='color:#5B8FF9'>■</span> clean</sup>",
            x=0.01,
        ),
        height=280 * nrows + 80,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", zeroline=False)

    path = os.path.join(output_dir, f"{dataset_name}_boxplots.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────────────
# 3. Correlation heatmap
# ─────────────────────────────────────────────────────

def _correlation_heatmap(df, num_cols, output_dir, dataset_name):
    if len(num_cols) < 2:
        return None

    corr = df[num_cols].corr().round(2)

    # Mask diagonal for cleaner look
    mask = corr.copy()
    np.fill_diagonal(mask.values, np.nan)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
    )

    fig.update_traces(
        textfont=dict(size=12, family="Arial"),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    )

    # Highlight strong correlations with border
    strong_pairs = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if i != j and abs(corr.loc[c1, c2]) > 0.85:
                strong_pairs.append((i, j, corr.loc[c1, c2]))

    for i, j, val in strong_pairs:
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5,
            y0=j - 0.5, y1=j + 0.5,
            line=dict(color="#2c3e50", width=2),
        )

    fig.update_layout(
        title=dict(
            text="<b>Correlation matrix</b><br>"
                 "<sup>Boxes highlight |r| > 0.85 (multicollinearity risk)</sup>",
            x=0.01,
        ),
        coloraxis_colorbar=dict(
            title="r",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1<br>(negative)", "-0.5", "0", "0.5", "1<br>(positive)"],
            len=0.8,
        ),
        height=max(400, len(num_cols) * 50 + 120),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        xaxis=dict(tickangle=-45),
    )

    path = os.path.join(output_dir, f"{dataset_name}_correlation.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────────────
# 4. Missing data bar chart
# ─────────────────────────────────────────────────────

def _missing_bar(df, report, output_dir, dataset_name):
    missing_pcts = {
        c.name: c.pct_missing
        for c in report.columns
        if c.pct_missing > 0
    }

    if not missing_pcts:
        # All complete — show a "no missing data" chart
        fig = go.Figure(go.Bar(
            x=list(df.columns),
            y=[0] * len(df.columns),
            marker_color="#2ecc71",
            text=["100% complete"] * len(df.columns),
            textposition="outside",
        ))
        fig.update_layout(
            title="<b>Missing data</b> — No missing values detected",
            yaxis=dict(range=[0, 5], title="% missing"),
            height=300,
        )
    else:
        cols   = list(missing_pcts.keys())
        values = list(missing_pcts.values())
        colors = ["#e74c3c" if v > 20 else "#f39c12" if v > 5 else "#5B8FF9"
                  for v in values]

        fig = go.Figure(go.Bar(
            x=cols,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>",
        ))

        # Threshold lines
        fig.add_hline(y=5,  line_dash="dash", line_color="#f39c12",
                      annotation_text="5% threshold", annotation_position="right")
        fig.add_hline(y=20, line_dash="dash", line_color="#e74c3c",
                      annotation_text="20% threshold", annotation_position="right")

        fig.update_layout(
            title=dict(
                text="<b>Missing data by column</b><br>"
                     "<sup><span style='color:#e74c3c'>■</span> >20%  "
                     "<span style='color:#f39c12'>■</span> 5-20%  "
                     "<span style='color:#5B8FF9'>■</span> <5%</sup>",
                x=0.01,
            ),
            yaxis=dict(title="% missing", range=[0, max(values) * 1.3]),
            xaxis=dict(title="Column"),
            height=380,
        )

    fig.update_layout(
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )

    path = os.path.join(output_dir, f"{dataset_name}_missing.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────────────
# 5. Target distribution
# ─────────────────────────────────────────────────────

def _target_distribution(df, report, output_dir, dataset_name):
    target_col = report.target_column
    if not target_col or target_col not in df.columns:
        return None

    target = df[target_col]

    if report.target_type == "classification":
        vc = target.value_counts().reset_index()
        vc.columns = ["class", "count"]
        vc["pct"] = (vc["count"] / vc["count"].sum() * 100).round(1)

        # Color imbalanced classes red
        min_pct = vc["pct"].min()
        colors = ["#e74c3c" if p == min_pct and min_pct < 20 else "#5B8FF9"
                  for p in vc["pct"]]

        fig = go.Figure(go.Bar(
            x=vc["class"].astype(str),
            y=vc["count"],
            marker_color=colors,
            text=[f"{p:.1f}%" for p in vc["pct"]],
            textposition="outside",
            hovertemplate="Class: %{x}<br>Count: %{y}<br>%{text}<extra></extra>",
        ))

        imbalance_note = ""
        if report.has_class_imbalance:
            imbalance_note = f"<br><sup style='color:#e74c3c'>⚠ Class imbalance detected — minority class: {min_pct:.1f}%</sup>"

        fig.update_layout(
            title=dict(
                text=f"<b>Target distribution — {target_col}</b>{imbalance_note}",
                x=0.01,
            ),
            xaxis=dict(title="Class"),
            yaxis=dict(title="Count"),
            height=380,
        )

    else:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=target.dropna(),
            nbinsx=40,
            marker_color="#5B8FF9",
            opacity=0.8,
            name=target_col,
        ))
        # KDE
        try:
            clean = target.dropna()
            kde = scipy_stats.gaussian_kde(clean)
            x_range = np.linspace(clean.min(), clean.max(), 300)
            kde_vals = kde(x_range)
            counts, _ = np.histogram(clean, bins=40)
            scale = counts.max() / kde_vals.max()
            fig.add_trace(go.Scatter(
                x=x_range, y=kde_vals * scale,
                mode="lines",
                line=dict(color="#e74c3c", width=2),
                name="KDE",
            ))
        except Exception:
            pass

        fig.update_layout(
            title=dict(
                text=f"<b>Target distribution — {target_col}</b>",
                x=0.01,
            ),
            xaxis=dict(title=target_col),
            yaxis=dict(title="Count"),
            height=380,
        )

    fig.update_layout(
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )

    path = os.path.join(output_dir, f"{dataset_name}_target_dist.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────────────
# 6. Quality gauge
# ─────────────────────────────────────────────────────

def _quality_gauge(report, output_dir, dataset_name):
    score = report.data_quality_score
    bar_color = "#2ecc71" if score >= 80 else "#f39c12" if score >= 60 else "#e74c3c"
    label = "Good" if score >= 80 else "Fair" if score >= 60 else "Poor"

    # Build issues list for annotation
    issues = []
    if report.has_missing_data:        issues.append("Missing data")
    if report.has_outliers:            issues.append("Outliers")
    if report.has_skewed_features:     issues.append("Skewed features")
    if report.has_class_imbalance:     issues.append("Class imbalance")
    if report.has_high_cardinality:    issues.append("High cardinality")
    if report.has_multicollinearity:   issues.append("Multicollinearity")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "table"}]],
        column_widths=[0.5, 0.5],
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(text=f"<b>Data quality</b><br><span style='font-size:0.9em;color:gray'>{label}</span>"),
        delta=dict(reference=80, valueformat=".0f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor="white",
            borderwidth=1,
            bordercolor="#ccc",
            steps=[
                dict(range=[0,  60], color="#fdecea"),
                dict(range=[60, 80], color="#fef3cd"),
                dict(range=[80, 100], color="#eafaf1"),
            ],
            threshold=dict(
                line=dict(color="#2c3e50", width=2),
                thickness=0.75,
                value=score,
            ),
        ),
    ), row=1, col=1)

    # Issues table
    if issues:
        header_text = ["Issue", "Status"]
        cells_col1  = issues
        cells_col2  = ["⚠ Detected"] * len(issues)
    else:
        cells_col1 = ["No issues found"]
        cells_col2 = ["✓ Clean"]

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Issue</b>", "<b>Status</b>"],
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="left",
            height=30,
        ),
        cells=dict(
            values=[cells_col1, cells_col2],
            fill_color=[
                ["#fff5f5" if i % 2 == 0 else "white" for i in range(len(cells_col1))],
                ["#fff5f5" if i % 2 == 0 else "white" for i in range(len(cells_col1))],
            ],
            font=dict(size=11),
            align="left",
            height=26,
        ),
    ), row=1, col=2)

    fig.update_layout(
        height=380,
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(
            text=f"<b>Data quality dashboard — {dataset_name}</b>",
            x=0.01,
        ),
    )

    path = os.path.join(output_dir, f"{dataset_name}_quality_gauge.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────────────
# 7. Skewness bar chart
# ─────────────────────────────────────────────────────

def _skewness_chart(report, num_cols, output_dir, dataset_name):
    col_profiles = {c.name: c for c in report.columns if c.name in num_cols}
    names  = [n for n in num_cols if col_profiles.get(n) and col_profiles[n].skewness is not None]
    skews  = [col_profiles[n].skewness for n in names]

    if not names:
        return None

    colors = ["#e74c3c" if abs(s) > 1 else "#f39c12" if abs(s) > 0.5 else "#2ecc71"
              for s in skews]

    fig = go.Figure(go.Bar(
        x=names,
        y=skews,
        marker_color=colors,
        text=[f"{s:.2f}" for s in skews],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Skewness: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=1,  line_dash="dash", line_color="#e74c3c",
                  annotation_text="+1 threshold")
    fig.add_hline(y=-1, line_dash="dash", line_color="#e74c3c",
                  annotation_text="-1 threshold")
    fig.add_hline(y=0,  line_dash="solid", line_color="#95a5a6",
                  line_width=0.5)

    fig.update_layout(
        title=dict(
            text="<b>Skewness by feature</b><br>"
                 "<sup><span style='color:#e74c3c'>■</span> |skew| > 1 (high)  "
                 "<span style='color:#f39c12'>■</span> 0.5–1 (moderate)  "
                 "<span style='color:#2ecc71'>■</span> < 0.5 (low)</sup>",
            x=0.01,
        ),
        yaxis=dict(title="Skewness", zeroline=True, zerolinecolor="#ccc"),
        xaxis=dict(title="Feature", tickangle=-30),
        height=400,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )

    path = os.path.join(output_dir, f"{dataset_name}_skewness.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path
