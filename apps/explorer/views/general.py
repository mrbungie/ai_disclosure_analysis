"""
apps/explorer/views/general.py — View 1: Mini-Tableau Visual Gold Explorer with Thesis Sector Taxonomy.
Supports multi-measures (+ Add Metric), aggregation formulas, and thesis-defined economic sectors (11 sectors, 56 SIC-2 groups, 7 divisions).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

from apps.explorer.db import (
    get_sic_filter_options,
    get_tickers_list,
    run_query,
)

# Curated Gold Datasets with thesis.qmd economic sector dimensions
DATASETS_CONFIG = {
    "📅 Firm-Year (Annual Panel)": {
        "table": "gold.datasets__firm_year__firm_year",
        "dimensions": [
            ("Year", "t.year"),
            ("Aggregated Economic Sector (Thesis)", "s.agg_sector"),
            ("2-Digit Major SIC Group", "s.sic2"),
            ("Broad SIC Division", "s.division"),
            ("Ticker", "t.ticker"),
        ],
        "measures": [
            ("AI Frames (Paragraphs)", "n_frames"),
            ("AI Activities Count", "activities__n_activities"),
            ("Washing Score (w)", "w"),
            ("Substance Ratio (pct_substance)", "pct_substance"),
            ("Proprietary AI Disclosures", "proprietary_ai"),
            ("Third-Party Named Providers", "third_party_named_provider"),
            ("Promotional Posture Rate", "promotional_posture"),
            ("Hedging Posture Rate", "hedging_posture"),
            ("Temporal Posture Rate", "temporal_posture"),
        ],
        "default_x": "t.year",
        "default_y": "n_frames",
        "default_agg": "AVG",
        "default_chart": "Bar Chart",
    },
    "🏢 Firm Static (Archetypes & Clusters)": {
        "table": "gold.datasets__firm__firm",
        "dimensions": [
            ("AI Posture Archetype", "t.archetype"),
            ("Aggregated Economic Sector (Thesis)", "s.agg_sector"),
            ("2-Digit Major SIC Group", "s.sic2"),
            ("Broad SIC Division", "s.division"),
            ("Cluster ID", "t.cluster"),
            ("Ticker", "t.ticker"),
        ],
        "measures": [
            ("Archetype Stability", "archetype_stability"),
            ("Total AI Frames", "n_frames"),
            ("Disclosure Intensity", "disclosure_intensity"),
            ("Promotional Posture", "promotional_posture"),
            ("Risk Orientation", "risk_orientation"),
            ("Governance Orientation", "governance_orientation"),
            ("Specificity Score", "specificity"),
            ("AI Positioning", "ai_positioning"),
        ],
        "default_x": "t.archetype",
        "default_y": "archetype_stability",
        "default_agg": "AVG",
        "default_chart": "Bar Chart",
    },
    "📊 Firm-Quarter (Quarterly Panel)": {
        "table": "gold.datasets__firm_quarter__firm_quarter",
        "dimensions": [
            ("Quarter", "t.quarter"),
            ("Aggregated Economic Sector (Thesis)", "s.agg_sector"),
            ("2-Digit Major SIC Group", "s.sic2"),
            ("Broad SIC Division", "s.division"),
            ("Ticker", "t.ticker"),
        ],
        "measures": [
            ("AI Frames Count", "n_frames"),
            ("Activities Count", "n_activities"),
            ("Washing Score", "w"),
            ("Substance Ratio", "pct_substance"),
            ("Promotional Rate", "promotional_posture"),
            ("Hedging Rate", "hedging_posture"),
        ],
        "default_x": "t.quarter",
        "default_y": "n_frames",
        "default_agg": "AVG",
        "default_chart": "Line Trend",
    },
    "📑 Documents & Filings (10-K, 10-Q, Calls)": {
        "table": "gold.datasets__document__document",
        "dimensions": [
            ("Filing Form Type", "t.form"),
            ("Channel (SEC / Call)", "t.channel"),
            ("Fiscal Year", "t.fy"),
            ("Aggregated Economic Sector (Thesis)", "s.agg_sector"),
            ("2-Digit Major SIC Group", "s.sic2"),
            ("Broad SIC Division", "s.division"),
            ("Ticker", "t.ticker"),
        ],
        "measures": [
            ("AI Frames in Document", "n_frames"),
            ("AI Activities in Document", "n_activities"),
            ("Proprietary AI Mentions", "proprietary_ai"),
            ("Named Provider Mentions", "third_party_named_provider"),
            ("Infrastructure Investment Mentions", "infrastructure_investment"),
            ("Quantified Outcome Mentions", "quantified_outcome"),
            ("Governance / Restriction Mentions", "governance_or_restriction"),
        ],
        "default_x": "t.form",
        "default_y": "n_frames",
        "default_agg": "AVG",
        "default_chart": "Bar Chart",
    },
    "⚡ AI Activities (Atomic Actions & Tech)": {
        "table": "gold.datasets__activity__activity",
        "dimensions": [
            ("Deployment Stage", "t.stage"),
            ("Provider Family", "t.provider_family"),
            ("Evidence Type", "t.evidence_type"),
            ("Form", "t.form"),
            ("Aggregated Economic Sector (Thesis)", "s.agg_sector"),
            ("2-Digit Major SIC Group", "s.sic2"),
            ("Broad SIC Division", "s.division"),
            ("Ticker", "t.ticker"),
        ],
        "measures": [
            ("Activity Records Count", "COUNT_STAR"),
            ("Firm Promotional Posture", "firm__promotional_posture"),
            ("Firm Risk Orientation", "firm__risk_orientation"),
        ],
        "default_x": "t.provider_family",
        "default_y": "COUNT_STAR",
        "default_agg": "COUNT",
        "default_chart": "Bar Chart",
    },
}

AGG_CHOICES = ["AVG", "SUM", "MEDIAN", "COUNT", "MIN", "MAX"]
CHART_TYPES = ["Bar Chart", "Line Trend", "Scatter Plot", "Box Plot"]

METRIC_FORMULA_DOCS = {
    "n_frames": "**AI Frames (Paragraphs)**: Total count of extracted paragraphs containing classified AI mentions.",
    "activities__n_activities": "**AI Activities**: Count of atomic, actionable AI initiatives and technology deployments.",
    "w": "**Washing Score (w)**: Empirical decoupling metric ($w = \\text{volume\\_percentile} - \\text{substance\\_percentile}$). Positive values indicate narrative disclosure exceeding realized operational implementation.",
    "pct_substance": "**Substance Ratio (pct_substance)**: Proportion of concrete operational disclosures (internal proprietary models, named tools, infrastructure investments) relative to total AI claims.",
    "proprietary_ai": "**Proprietary AI**: Mentions of in-house developed models, internal neural architectures, and custom algorithms.",
    "third_party_named_provider": "**Third-Party Providers**: Disclosed partnerships or integrations with named vendors (OpenAI, AWS Bedrock, Microsoft Copilot, Anthropic, etc.).",
    "promotional_posture": "**Promotional Posture**: Proportion of sentences framing AI through market leadership, revenue upside, or competitive superiority.",
    "hedging_posture": "**Hedging Posture**: Proportion of sentences containing legal/operational caveats, safety disclaimers, or uncertainty disclaimers.",
    "temporal_posture": "**Temporal Posture**: Ratio of realized existing AI deployments versus hypothetical or aspirational future claims.",
    "archetype_stability": "**Archetype Stability**: Posterior assignment probability and silhouette stability of the firm's AI posture cluster.",
    "COUNT_STAR": "**Record Count**: Raw frequency count of matching rows or filing events.",
}


def get_dataset_choices():
    return list(DATASETS_CONFIG.keys())


def on_dataset_change(dataset_name: str):
    """Update X and all Y measure choices when dataset changes."""
    cfg = DATASETS_CONFIG.get(dataset_name, DATASETS_CONFIG[list(DATASETS_CONFIG.keys())[0]])
    x_choices = [(label, col) for label, col in cfg["dimensions"]]
    y_choices = [("None", "")] + [(label, col) for label, col in cfg["measures"]]
    color_choices = [("None", "")] + [(label, col) for label, col in cfg["dimensions"] if col != cfg["default_x"]]

    return (
        gr.update(choices=x_choices, value=cfg["default_x"]),
        gr.update(choices=y_choices[1:], value=cfg["default_y"]),
        gr.update(value=cfg["default_agg"]),
        gr.update(choices=y_choices, value=""),
        gr.update(choices=y_choices, value=""),
        gr.update(choices=y_choices, value=""),
        gr.update(choices=y_choices, value=""),
        gr.update(choices=color_choices, value=""),
        gr.update(value=cfg["default_chart"]),
    )


def generate_tableau_chart(
    dataset_name: str,
    chart_type: str,
    x_col: str,
    color_col: str,
    selected_tickers: list[str] | None,
    selected_divisions: list[str] | None,
    selected_sic2: list[str] | None,
    selected_sectors: list[str] | None,
    m1_col: str,
    m1_agg: str,
    m2_col: str,
    m2_agg: str,
    m3_col: str,
    m3_agg: str,
    m4_col: str,
    m4_agg: str,
    m5_col: str,
    m5_agg: str,
):
    """Build and render an interactive Plotly chart supporting multi-measures and thesis economic sectors."""
    cfg = DATASETS_CONFIG.get(dataset_name, DATASETS_CONFIG[list(DATASETS_CONFIG.keys())[0]])
    table = cfg["table"]

    # Gather active measures
    raw_measures = [
        (m1_col, m1_agg or "AVG"),
        (m2_col, m2_agg or "AVG"),
        (m3_col, m3_agg or "AVG"),
        (m4_col, m4_agg or "AVG"),
        (m5_col, m5_agg or "AVG"),
    ]
    active_measures: list[tuple[str, str, str]] = []
    seen_aliases = set()

    for col, agg in raw_measures:
        if col and col.strip():
            clean_col = col.strip()
            clean_agg = agg.strip().upper() if agg else "AVG"
            if clean_col == "COUNT_STAR":
                alias = "COUNT(*)"
            else:
                alias = f"{clean_agg}({clean_col})"
            if alias not in seen_aliases:
                seen_aliases.add(alias)
                active_measures.append((clean_col, clean_agg, alias))

    if not active_measures:
        def_col = cfg["default_y"]
        def_agg = cfg.get("default_agg", "AVG")
        alias = "COUNT(*)" if def_col == "COUNT_STAR" else f"{def_agg}({def_col})"
        active_measures = [(def_col, def_agg, alias)]

    # Sector table join check
    needs_sector_join = (
        "s." in (x_col or "")
        or "s." in (color_col or "")
        or bool(selected_divisions)
        or bool(selected_sic2)
        or bool(selected_sectors)
    )

    from_clause = f"{table} t"
    if needs_sector_join:
        from_clause += " LEFT JOIN internal_firm_sectors s ON t.ticker = s.ticker"

    # Build WHERE clauses
    where_parts = []
    if selected_tickers and len(selected_tickers) > 0:
        escaped = [f"'{t}'" for t in selected_tickers if t]
        if escaped:
            where_parts.append(f"t.ticker IN ({', '.join(escaped)})")

    if selected_divisions and len(selected_divisions) > 0:
        escaped_div = [f"'{d.replace("'", "''")}'" for d in selected_divisions if d]
        if escaped_div:
            where_parts.append(f"s.division IN ({', '.join(escaped_div)})")

    if selected_sic2 and len(selected_sic2) > 0:
        escaped_s2 = [f"'{c}'" for c in selected_sic2 if c]
        if escaped_s2:
            where_parts.append(f"s.sic2 IN ({', '.join(escaped_s2)})")

    if selected_sectors and len(selected_sectors) > 0:
        escaped_sec = [f"'{sec.replace("'", "''")}'" for sec in selected_sectors if sec]
        if escaped_sec:
            where_parts.append(f"s.agg_sector IN ({', '.join(escaped_sec)})")

    if x_col:
        where_parts.append(f"{x_col} IS NOT NULL")

    where_str = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

    # Clean dimension column names for grouping and plotting
    x_alias = "x_axis"
    x_expr = f"{x_col} AS {x_alias}"

    # Grouped / Aggregated Query for Bar and Line charts
    group_expressions = [x_col]
    select_dimensions = [x_expr]
    group_plot_cols = [x_alias]

    if color_col and color_col != x_col:
        color_alias = "color_group"
        group_expressions.append(color_col)
        select_dimensions.append(f"{color_col} AS {color_alias}")
        group_plot_cols.append(color_alias)

    # Box Plot or Scatter Plot without aggregation
    if chart_type in ("Box Plot", "Scatter Plot"):
        first_metric = active_measures[0][0]
        select_cols = [f"{x_col} AS {x_alias}"]
        if first_metric != "COUNT_STAR":
            select_cols.append(f't."{first_metric}"')
        if len(active_measures) > 1 and active_measures[1][0] != "COUNT_STAR":
            select_cols.append(f't."{active_measures[1][0]}"')
        if color_col:
            select_cols.append(f"{color_col} AS color_group")
        select_cols.append('t.ticker AS "ticker"')

        sql = f"""
        SELECT {', '.join(set(select_cols))}
        FROM {from_clause}
        {where_str}
        LIMIT 2000
        """
        df, err, elapsed, count = run_query(sql, limit=2000)
        if err:
            return None, None, f"⚠️ Query Error: {err}"

        if df is None or df.empty:
            return None, None, "No data matches your selection."

        readable_x = x_col.replace("s.", "").replace("t.", "").replace("agg_sector", "Economic Sector").replace("sic2", "SIC-2").replace("division", "Broad Division")

        if chart_type == "Box Plot":
            metric_col = first_metric if first_metric != "COUNT_STAR" else df.columns[1]
            fig = px.box(
                df,
                x=x_alias,
                y=metric_col,
                color="color_group" if color_col else None,
                template="plotly_white",
                title=f"{metric_col} Distribution across {readable_x}",
            )
        else:
            # Scatter Plot
            if len(active_measures) > 1 and active_measures[1][0] != "COUNT_STAR":
                m1, m2 = active_measures[0][0], active_measures[1][0]
                fig = px.scatter(
                    df,
                    x=m1,
                    y=m2,
                    color="color_group" if color_col else x_alias,
                    hover_data=["ticker"],
                    template="plotly_white",
                    title=f"{m2} vs {m1}",
                )
            else:
                m1 = first_metric if first_metric != "COUNT_STAR" else df.columns[1]
                fig = px.scatter(
                    df,
                    x=x_alias,
                    y=m1,
                    color="color_group" if color_col else None,
                    hover_data=["ticker"],
                    template="plotly_white",
                    title=f"{m1} vs {readable_x}",
                )

        fig.update_layout(
            font_family="Inter, -apple-system, sans-serif",
            title_font_size=15,
            xaxis_title=readable_x,
            margin=dict(l=40, r=30, t=50, b=40),
        )
        kpi_text = f"**Plotly Visualization:** `{count:,}` records sampled in `{elapsed}s`."
        return fig, df.head(50), kpi_text

    # Standard Aggregated Group Query (Bar & Line)
    measure_selects = []
    measure_aliases = []
    for col, agg, alias in active_measures:
        if col == "COUNT_STAR":
            measure_selects.append(f'COUNT(*) AS "{alias}"')
        else:
            measure_selects.append(f'ROUND({agg}(t."{col}"), 4) AS "{alias}"')
        measure_aliases.append(alias)

    sql = f"""
    SELECT 
        {', '.join(select_dimensions)},
        COUNT(*) AS row_count,
        {", ".join(measure_selects)}
    FROM {from_clause}
    {where_str}
    GROUP BY {', '.join(group_expressions)}
    ORDER BY {x_alias} ASC
    LIMIT 500
    """
    df, err, elapsed, count = run_query(sql, limit=500)
    if err:
        return None, None, f"⚠️ Aggregation Error: {err}"

    if df is None or df.empty:
        return None, None, "No records found matching these criteria."

    # Format chart title and layout
    measures_str = ", ".join(measure_aliases)
    readable_x = x_col.replace("s.", "").replace("t.", "").replace("agg_sector", "Economic Sector").replace("sic2", "SIC-2").replace("division", "Broad Division")
    chart_title = f"{measures_str} by {readable_x}"
    if color_col:
        chart_title += f" (Color: {color_col.replace('s.', '').replace('t.', '')})"

    if chart_type == "Line Trend":
        fig = px.line(
            df,
            x=x_alias,
            y=measure_aliases if len(measure_aliases) > 1 else measure_aliases[0],
            color="color_group" if color_col else None,
            markers=True,
            template="plotly_white",
            title=chart_title,
        )
    else:
        # Bar Chart
        fig = px.bar(
            df,
            x=x_alias,
            y=measure_aliases if len(measure_aliases) > 1 else measure_aliases[0],
            color="color_group" if color_col else None,
            barmode="group",
            template="plotly_white",
            title=chart_title,
        )

    fig.update_layout(
        font_family="Inter, -apple-system, sans-serif",
        title_font_size=15,
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis_title=readable_x,
        yaxis_title="Metric Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
    )

    # Active formula summary text
    formula_items = [f"`{alias}`" for alias in measure_aliases]
    kpi_text = (
        f"**Active Formulas:** {', '.join(formula_items)} &nbsp;|&nbsp; "
        f"`{count}` groups calculated in `{elapsed}s`"
    )
    return fig, df, kpi_text


def create_general_view():
    """Build Gradio UI components for the Multi-Metric Mini-Tableau Visual Explorer."""
    tickers_list = [t[1] for t in get_tickers_list()]
    divisions_list, sic2_choices, sectors_list = get_sic_filter_options()

    default_dataset = "📅 Firm-Year (Annual Panel)"
    cfg = DATASETS_CONFIG[default_dataset]
    all_measures_choices = [("None", "")] + [(label, col) for label, col in cfg["measures"]]

    visible_measures_count = gr.State(value=1)

    with gr.Column():
        gr.Markdown(
            """
            ### 📊 Visual Gold Explorer (Mini-Tableau)
            Select a Gold panel, pick dimensions on the X-axis, and **add multiple measures** with custom summarizations (`AVG`, `SUM`, `MEDIAN`, etc.). Filter by **Aggregated Economic Sectors (Thesis)**, **SIC-2**, and **Broad Divisions**.
            """
        )

        with gr.Group():
            with gr.Row():
                dataset_selector = gr.Dropdown(
                    choices=get_dataset_choices(),
                    value=default_dataset,
                    label="1. Dataset / Grain",
                    interactive=True,
                    scale=2,
                )
                x_axis_selector = gr.Dropdown(
                    choices=[(label, col) for label, col in cfg["dimensions"]],
                    value=cfg["default_x"],
                    label="2. Columns Shelf (X-Axis Dimension)",
                    interactive=True,
                    scale=2,
                )
                chart_type_selector = gr.Dropdown(
                    choices=CHART_TYPES,
                    value=cfg["default_chart"],
                    label="3. Chart Type",
                    interactive=True,
                    scale=1,
                )
                color_selector = gr.Dropdown(
                    choices=[("None", "")] + [(label, col) for label, col in cfg["dimensions"] if col != cfg["default_x"]],
                    value="",
                    label="Color Breakdown (Optional)",
                    interactive=True,
                    scale=1,
                )

            gr.Markdown("#### 📈 Rows Shelf (Measures / Metrics & Summarization)")

            # Measure Slot 1 (always visible)
            with gr.Row():
                m1_metric = gr.Dropdown(
                    choices=[(label, col) for label, col in cfg["measures"]],
                    value=cfg["default_y"],
                    label="Measure 1 (Metric)",
                    interactive=True,
                    scale=3,
                )
                m1_agg = gr.Dropdown(
                    choices=AGG_CHOICES,
                    value="AVG",
                    label="Summarize By",
                    interactive=True,
                    scale=1,
                )

            # Measure Slot 2
            with gr.Row(visible=False) as m2_row:
                m2_metric = gr.Dropdown(
                    choices=all_measures_choices,
                    value="pct_substance" if any(c[1] == "pct_substance" for c in cfg["measures"]) else "",
                    label="Measure 2 (Metric)",
                    interactive=True,
                    scale=3,
                )
                m2_agg = gr.Dropdown(
                    choices=AGG_CHOICES,
                    value="AVG",
                    label="Summarize By",
                    interactive=True,
                    scale=1,
                )
                m2_clear_btn = gr.Button("✕", size="sm", variant="secondary", scale=0)

            # Measure Slot 3
            with gr.Row(visible=False) as m3_row:
                m3_metric = gr.Dropdown(
                    choices=all_measures_choices,
                    value="",
                    label="Measure 3 (Metric)",
                    interactive=True,
                    scale=3,
                )
                m3_agg = gr.Dropdown(
                    choices=AGG_CHOICES,
                    value="AVG",
                    label="Summarize By",
                    interactive=True,
                    scale=1,
                )
                m3_clear_btn = gr.Button("✕", size="sm", variant="secondary", scale=0)

            # Measure Slot 4
            with gr.Row(visible=False) as m4_row:
                m4_metric = gr.Dropdown(
                    choices=all_measures_choices,
                    value="",
                    label="Measure 4 (Metric)",
                    interactive=True,
                    scale=3,
                )
                m4_agg = gr.Dropdown(
                    choices=AGG_CHOICES,
                    value="AVG",
                    label="Summarize By",
                    interactive=True,
                    scale=1,
                )
                m4_clear_btn = gr.Button("✕", size="sm", variant="secondary", scale=0)

            # Measure Slot 5
            with gr.Row(visible=False) as m5_row:
                m5_metric = gr.Dropdown(
                    choices=all_measures_choices,
                    value="",
                    label="Measure 5 (Metric)",
                    interactive=True,
                    scale=3,
                )
                m5_agg = gr.Dropdown(
                    choices=AGG_CHOICES,
                    value="AVG",
                    label="Summarize By",
                    interactive=True,
                    scale=1,
                )
                m5_clear_btn = gr.Button("✕", size="sm", variant="secondary", scale=0)

            with gr.Row():
                add_metric_btn = gr.Button("➕ Add Metric", variant="secondary", size="sm", scale=1)
                update_chart_btn = gr.Button("⚡ Update Chart", variant="primary", size="lg", scale=3)

            # Sector & Ticker Filters (Thesis Taxonomy: 11 Sectors, SIC-2, Divisions, Tickers)
            with gr.Accordion("🔍 Sector & Ticker Filters (Thesis 11 Economic Sectors, SIC-2, Divisions)", open=False):
                with gr.Row():
                    sector_filter = gr.Dropdown(
                        choices=sectors_list,
                        value=[],
                        multiselect=True,
                        label="Aggregated Economic Sector (11 Thesis Sectors)",
                        info="e.g. Technology, Financial Services, Healthcare & Pharma, Industrials & Mfg",
                        scale=2,
                    )
                    division_filter = gr.Dropdown(
                        choices=divisions_list,
                        value=[],
                        multiselect=True,
                        label="Broad SIC Division (7 Divisions)",
                        info="e.g. Manufacturing, Services, Transport & Utilities",
                        scale=1,
                    )

                with gr.Row():
                    sic2_filter = gr.Dropdown(
                        choices=sic2_choices,
                        value=[],
                        multiselect=True,
                        label="2-Digit Major SIC Group (56 Groups)",
                        info="e.g. 73: Business Services / Software, 35: Industrial Machinery & Computers",
                        scale=2,
                    )
                    ticker_filter = gr.Dropdown(
                        choices=tickers_list,
                        value=[],
                        multiselect=True,
                        label="Tickers Filter (Optional)",
                        info="Filter specific companies (e.g. AAPL, NVDA, MSFT)",
                        scale=1,
                    )

            # Collapsible Metric Calculation & Formula Reference
            with gr.Accordion("📖 Metric Dictionary & Calculation Formulas", open=False):
                metric_docs_md = "\n\n".join([f"- {desc}" for desc in METRIC_FORMULA_DOCS.values()])
                gr.Markdown(metric_docs_md)

        kpi_display = gr.Markdown("")

        # Interactive Plotly Chart Area
        plot_output = gr.Plot(label="Interactive Visualization")

        # Summary Pivot View Table
        with gr.Accordion("📋 Aggregated Data Summary (Pivot View)", open=False):
            summary_table = gr.DataFrame(
                datatype="auto",
                interactive=False,
                wrap=True,
            )

        # Event: Add metric slot
        def on_add_slot(current_count):
            new_count = min(current_count + 1, 5)
            return (
                new_count,
                gr.update(visible=new_count >= 2),
                gr.update(visible=new_count >= 3),
                gr.update(visible=new_count >= 4),
                gr.update(visible=new_count >= 5),
            )

        add_metric_btn.click(
            on_add_slot,
            inputs=[visible_measures_count],
            outputs=[visible_measures_count, m2_row, m3_row, m4_row, m5_row],
        )

        # Clear buttons
        m2_clear_btn.click(lambda: ("", gr.update(visible=False)), outputs=[m2_metric, m2_row])
        m3_clear_btn.click(lambda: ("", gr.update(visible=False)), outputs=[m3_metric, m3_row])
        m4_clear_btn.click(lambda: ("", gr.update(visible=False)), outputs=[m4_metric, m4_row])
        m5_clear_btn.click(lambda: ("", gr.update(visible=False)), outputs=[m5_metric, m5_row])

        # Event: Dataset change
        dataset_selector.change(
            on_dataset_change,
            inputs=[dataset_selector],
            outputs=[
                x_axis_selector,
                m1_metric,
                m1_agg,
                m2_metric,
                m3_metric,
                m4_metric,
                m5_metric,
                color_selector,
                chart_type_selector,
            ],
        )

        # Event: Update chart
        def execute_viz(
            dset, ctype, xcol, color, tickers, divs, s2, secs,
            m1_c, m1_a, m2_c, m2_a, m3_c, m3_a, m4_c, m4_a, m5_c, m5_a
        ):
            fig, df, kpi = generate_tableau_chart(
                dset, ctype, xcol, color, tickers, divs, s2, secs,
                m1_c, m1_a, m2_c, m2_a, m3_c, m3_a, m4_c, m4_a, m5_c, m5_a
            )
            return fig, df, kpi

        update_chart_btn.click(
            execute_viz,
            inputs=[
                dataset_selector,
                chart_type_selector,
                x_axis_selector,
                color_selector,
                ticker_filter,
                division_filter,
                sic2_filter,
                sector_filter,
                m1_metric, m1_agg,
                m2_metric, m2_agg,
                m3_metric, m3_agg,
                m4_metric, m4_agg,
                m5_metric, m5_agg,
            ],
            outputs=[plot_output, summary_table, kpi_display],
        )

        # Preload initial visualization on start
        init_fig, init_df, init_kpi = generate_tableau_chart(
            default_dataset,
            cfg["default_chart"],
            cfg["default_x"],
            "",
            None, None, None, None,
            cfg["default_y"], "AVG",
            "", "AVG",
            "", "AVG",
            "", "AVG",
            "", "AVG",
        )
        plot_output.value = init_fig
        summary_table.value = init_df
        kpi_display.value = init_kpi

    return update_chart_btn
