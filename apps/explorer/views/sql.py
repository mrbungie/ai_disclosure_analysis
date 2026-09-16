"""
apps/explorer/views/sql.py — View 3: Interactive DuckDB SQL Explorer.
Ad-hoc SQL querying over in-memory Parquet views with English presets and catalog inspector.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from apps.explorer.db import (
    list_tables_by_schema,
    run_query,
)

PRESET_QUERIES = {
    "🏆 Top 15 Tickers by AI Frames Volume (2024)": """SELECT 
    ticker,
    year,
    n_frames,
    activities__n_activities AS n_activities,
    ROUND(w, 4) AS washing_score,
    ROUND(pct_substance, 4) AS pct_substance
FROM gold.datasets__firm_year__firm_year
WHERE year = 2024
ORDER BY n_frames DESC
LIMIT 15;""",

    "📈 Average Washing Score & Substance Ratio by Year": """SELECT 
    year,
    COUNT(DISTINCT ticker) AS total_firms,
    ROUND(AVG(w), 4) AS avg_washing_score,
    ROUND(AVG(pct_substance), 4) AS avg_pct_substance,
    ROUND(AVG(n_frames), 2) AS avg_n_frames
FROM gold.datasets__firm_year__firm_year
GROUP BY year
ORDER BY year ASC;""",

    "🎭 Distribution of AI Posture Archetypes": """SELECT 
    archetype,
    COUNT(*) AS firm_count,
    ROUND(AVG(archetype_stability), 3) AS avg_stability,
    ROUND(AVG(n_frames), 1) AS avg_frames,
    ROUND(AVG(promotional_posture), 3) AS avg_promo,
    ROUND(AVG(risk_orientation), 3) AS avg_risk
FROM gold.datasets__firm__firm
WHERE archetype IS NOT NULL
GROUP BY archetype
ORDER BY firm_count DESC;""",

    "🤖 Proprietary AI vs Third-Party Named Providers by Year": """SELECT 
    year,
    ROUND(SUM(proprietary_ai), 0) AS total_proprietary_ai,
    ROUND(SUM(third_party_named_provider), 0) AS total_third_party_named,
    ROUND(AVG(pct_substance), 3) AS avg_substance
FROM gold.datasets__firm_year__firm_year
GROUP BY year
ORDER BY year ASC;""",

    "📑 Filings with Highest Volume of Classified AI Paragraphs": """SELECT 
    ticker,
    form,
    accession_number,
    fecha AS filing_date,
    n_frames,
    n_activities,
    proprietary_ai,
    named_product_or_process
FROM gold.datasets__document__document
ORDER BY n_frames DESC
LIMIT 20;""",

    "🔬 Atomic Activities: Most Mentioned Provider Families": """SELECT 
    provider_family,
    COUNT(*) AS total_mentions,
    COUNT(DISTINCT ticker) AS unique_firms
FROM gold.datasets__activity__activity
WHERE provider_family IS NOT NULL AND provider_family != ''
GROUP BY provider_family
ORDER BY total_mentions DESC;""",
}


def on_preset_select(query_title: str):
    return PRESET_QUERIES.get(query_title, "")


def execute_custom_sql(sql: str, limit: int):
    if not sql or not sql.strip():
        return (
            None,
            gr.update(value="⚠️ Please enter a valid SQL query.", visible=True),
            "",
        )

    df, err, elapsed, row_count = run_query(sql, limit=int(limit))
    if err:
        return (
            None,
            gr.update(value=f"❌ **Query Error:**\n```\n{err}\n```", visible=True),
            "",
        )

    status_md = f"✅ **Query succeeded:** `{row_count:,}` rows returned in `{elapsed}s`."
    return df, gr.update(visible=False), status_md


def get_catalog_markdown() -> str:
    """Generate Markdown guide of all available schemas and tables in DuckDB."""
    schemas = list_tables_by_schema()
    lines = ["#### 📚 Registered Database Schemas & Tables\n"]
    for s in sorted(schemas.keys()):
        tbls = schemas[s]
        lines.append(f"<details><summary><b>Schema <code>{s}</code></b> ({len(tbls)} tables/views)</summary>\n")
        lines.append("<ul>")
        for t in sorted(tbls)[:25]:
            lines.append(f"<li><code>{s}.{t}</code></li>")
        if len(tbls) > 25:
            lines.append(f"<li><i>... and {len(tbls) - 25} more tables</i></li>")
        lines.append("</ul>\n</details>\n")
    return "\n".join(lines)


def create_sql_view():
    """Build Gradio UI components for View 3."""
    with gr.Column():
        gr.Markdown(
            """
            ### ⚡ Interactive SQL Playground (DuckDB)
            Run ad-hoc SQL queries directly over in-memory Parquet views.
            Tables are mapped by schemas: `gold.*`, `silver.*`, `bronze.*`, etc.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESET_QUERIES.keys()),
                        value=list(PRESET_QUERIES.keys())[0],
                        label="Example Query Presets",
                        interactive=True,
                        scale=3,
                    )
                    load_preset_btn = gr.Button("📋 Load Query", variant="secondary", scale=1)

                sql_editor = gr.Code(
                    value=PRESET_QUERIES[list(PRESET_QUERIES.keys())[0]],
                    language="sql",
                    lines=10,
                    label="SQL Editor (DuckDB dialect)",
                )

                with gr.Row():
                    limit_selector = gr.Dropdown(
                        choices=[50, 100, 250, 500, 1000, 5000],
                        value=100,
                        label="Row Limit (LIMIT)",
                        scale=1,
                    )
                    run_btn = gr.Button("▶️ Execute Query", variant="primary", scale=2, size="lg")

            with gr.Column(scale=1):
                with gr.Accordion("📖 Schema Catalog Browser", open=True):
                    catalog_display = gr.Markdown(value=get_catalog_markdown)

        status_box = gr.Markdown("")
        error_box = gr.Markdown("", visible=False)
        result_df = gr.DataFrame(
            headers=None,
            datatype="auto",
            wrap=True,
            interactive=False,
            label="Query Results",
        )

        # Events
        preset_dropdown.change(on_preset_select, inputs=[preset_dropdown], outputs=[sql_editor])
        load_preset_btn.click(on_preset_select, inputs=[preset_dropdown], outputs=[sql_editor])

        run_btn.click(
            execute_custom_sql,
            inputs=[sql_editor, limit_selector],
            outputs=[result_df, error_box, status_box],
        )

        return run_btn
