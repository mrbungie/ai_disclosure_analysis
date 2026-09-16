"""
apps/explorer/app.py — Gradio Dataset Explorer via internal in-memory DuckDB.

Run locally:
    uv run python -m apps.explorer.app
    # or via Makefile
    make explorer

Environment variables:
    DATA_URI / DATA_DIR     Path or URI to Parquet datasets (s3://, https://, hf://, or local)
    EXPLORER_HOST           Server host (default: 0.0.0.0)
    EXPLORER_PORT           Server port (default: 7860)
    EXPLORER_SHARE          Set to true to generate a public Gradio share link
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr

from apps.explorer.config import (
    DATA_URI,
    DEBUG,
    SERVER_NAME,
    SERVER_PORT,
    SHARE,
)
from apps.explorer.db import (
    get_init_stats,
    init_db,
)
from apps.explorer.views.card import create_card_view
from apps.explorer.views.general import create_general_view
from apps.explorer.views.sql import create_sql_view

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

code, pre, .cm-editor, .cm-scroller, .cm-content, .dataframe table {
    font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: auto !important;
}

.status-badge {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 12px;
}

blockquote {
    border-left: 3px solid #3b82f6 !important;
    background: #f8fafc;
    border-radius: 0 6px 6px 0;
    padding: 10px 16px !important;
    margin: 8px 0 !important;
    color: #1e293b;
    font-size: 0.94rem;
    line-height: 1.5;
}

.tab-nav button {
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}
"""


def get_status_banner_text() -> str:
    """Return formatted status text for the active DuckDB connection."""
    stats = get_init_stats()
    status = stats.get("status", "unknown")
    views = stats.get("views_count", 0)
    elapsed = stats.get("elapsed_s", 0)
    uri = stats.get("data_uri", str(DATA_URI))
    proto = stats.get("protocol", "local").upper()
    schemas = ", ".join(f"`{s}`" for s in stats.get("schemas", []))

    if status == "ready":
        return (
            f"🦆 **DuckDB:** In-Memory (Zero Copy) &nbsp;|&nbsp; "
            f"**Protocol:** `{proto}` &nbsp;|&nbsp; "
            f"**Views Registered:** `{views}` in `{elapsed}s` &nbsp;|&nbsp; "
            f"**Schemas:** {schemas} &nbsp;|&nbsp; "
            f"**Data Source:** `{uri}`"
        )
    return f"⚠️ DuckDB Status: `{status}` &nbsp;|&nbsp; Source: `{uri}`"


def reload_database():
    """Force re-scan and reload of all Parquet datasets into internal DuckDB."""
    init_db(force_refresh=True)
    return get_status_banner_text()


def build_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    init_db()

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#fafafa",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#e2e8f0",
        block_radius="8px",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title="Corporate AI Disclosure Explorer (DuckDB)") as demo:
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown(
                    """
                    # 🔬 SEC Corporate AI Disclosure Dataset Explorer
                    **Interactive visual exploration and ad-hoc analytics over thesis Parquet datasets via in-memory DuckDB.**  
                    Designed for local development and Hugging Face Spaces via `DATA_URI` (supports local, S3, B2, and HTTP/HF).
                    """
                )
            with gr.Column(scale=1):
                reload_btn = gr.Button("🔄 Reload Data Views", variant="secondary", size="sm")

        with gr.Group(elem_classes=["status-badge"]):
            status_banner = gr.Markdown(value=get_status_banner_text())

        reload_btn.click(reload_database, outputs=[status_banner])

        with gr.Tabs():
            with gr.TabItem("📊 1. Visual Gold Explorer (Mini-Tableau)"):
                create_general_view()

            with gr.TabItem("🏢 2. Company Profile & Filing Inspector"):
                create_card_view()

            with gr.TabItem("⚡ 3. SQL Playground"):
                create_sql_view()

    return demo


demo = build_app()

if __name__ == "__main__":
    print(f"🚀 Starting Corporate AI Disclosure Explorer at http://{SERVER_NAME}:{SERVER_PORT}")
    print(f"📁 Data Source: {DATA_URI}")
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )
    demo.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=SHARE,
        show_error=True,
        theme=custom_theme,
        css=CUSTOM_CSS,
    )
