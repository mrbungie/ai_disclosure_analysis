"""
apps/explorer/views/card.py — View 2: Company Card, Historical Trajectory, and Filing Inspector.
English interface with interactive Plotly trends and clean paragraph card inspection.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

from apps.explorer.db import (
    get_firm_overview,
    get_tickers_list,
    run_query,
)


def fetch_card_details(ticker: str):
    """Fetch profile summary, timeline metrics, and filings list for selected ticker."""
    if not ticker:
        return (
            "",
            None,
            gr.update(choices=[], value=None),
            None,
            None,
        )

    # 1. Company Overview
    overview = get_firm_overview(ticker)
    status_badge = "🔴 Delisted" if overview.get("delisted") else "🟢 Active (S&P 500)"
    archetype = overview.get("archetype", "Unclassified")
    stability = overview.get("archetype_stability")
    stab_str = f" (Stability: `{stability:.1%}`)" if stability is not None else ""
    intensity = overview.get("disclosure_intensity")
    int_str = f"{intensity:.2f}" if intensity is not None else "N/A"

    overview_md = f"""
    ### 🏢 {overview.get('company_name', ticker)} (`{ticker}`)
    **CIK:** `{overview.get('cik', 'N/A')}` | **SIC / Industry:** `{overview.get('sic', 'N/A')}` — {overview.get('industry_group', 'N/A')} | **Status:** {status_badge}
    
    > **AI Posture Archetype:** **`{archetype}`**{stab_str}  
    > **Disclosure Intensity:** `{int_str}` | **Total AI Frames:** `{overview.get('n_frames_total', 0)}`  
    > **Promotional Posture:** `{overview.get('promotional_posture', 0.0) or 0:.2f}` | **Risk Orientation:** `{overview.get('risk_orientation', 0.0) or 0:.2f}` | **Governance:** `{overview.get('governance_orientation', 0.0) or 0:.2f}`
    """

    # 2. Historical Trajectory (firm_year)
    sql_history = f"""
    SELECT 
        year,
        n_frames,
        activities__n_activities AS n_activities,
        ROUND(w, 4) AS washing_score,
        ROUND(pct_substance, 4) AS pct_substance,
        ROUND(promotional_posture, 4) AS promotional_posture,
        ROUND(proprietary_ai, 2) AS proprietary_ai,
        ROUND(third_party_named_provider, 2) AS third_party_named_provider
    FROM gold.datasets__firm_year__firm_year
    WHERE ticker = '{ticker}'
    ORDER BY year ASC
    """
    df_history, _, _, _ = run_query(sql_history)

    fig = None
    if df_history is not None and not df_history.empty and "year" in df_history.columns:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("AI Disclosure Volume (Frames & Activities)", "Disclosure Quality (Washing vs Substance)"),
            horizontal_spacing=0.12,
        )

        years = df_history["year"].tolist()

        # Left chart: Volume
        fig.add_trace(
            go.Bar(
                x=years,
                y=df_history["n_frames"].fillna(0).tolist(),
                name="AI Frames",
                marker_color="#3b82f6",
                opacity=0.85,
            ),
            row=1, col=1,
        )
        if "n_activities" in df_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=df_history["n_activities"].fillna(0).tolist(),
                    name="AI Activities",
                    mode="lines+markers",
                    line=dict(color="#ef4444", width=2.5),
                    marker=dict(size=7),
                ),
                row=1, col=1,
            )

        # Right chart: Quality
        if "washing_score" in df_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=df_history["washing_score"].fillna(0).tolist(),
                    name="Washing Score (w)",
                    mode="lines+markers",
                    line=dict(color="#8b5cf6", width=2.5),
                    marker=dict(size=7, symbol="square"),
                ),
                row=1, col=2,
            )
        if "pct_substance" in df_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=df_history["pct_substance"].fillna(0).tolist(),
                    name="Substance Ratio",
                    mode="lines+markers",
                    line=dict(color="#10b981", width=2.2, dash="dash"),
                    marker=dict(size=7, symbol="triangle-up"),
                ),
                row=1, col=2,
            )

        fig.update_layout(
            template="plotly_white",
            font_family="Inter, -apple-system, sans-serif",
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        )
        fig.update_xaxes(type="category")

    # 3. Filings List
    sql_filings = f"""
    SELECT 
        m.form_type, 
        m.filing_date, 
        m.period_end_date, 
        m.accession_number,
        COALESCE(d.n_frames, 0) AS n_frames,
        COALESCE(d.n_activities, 0) AS n_activities
    FROM (
        SELECT form_type, filing_date, period_end_date, accession_number, ticker FROM silver.filing_manifest
        UNION ALL
        SELECT form_type, filing_date, period_end_date, accession_number, ticker FROM silver.filing_manifest_10q
    ) m
    LEFT JOIN gold.datasets__document__document d ON m.accession_number = d.accession_number
    WHERE m.ticker = '{ticker}'
    ORDER BY m.filing_date DESC
    LIMIT 200
    """
    df_filings, _, _, _ = run_query(sql_filings)

    filing_choices = []
    default_acc = None
    if df_filings is not None and not df_filings.empty:
        sorted_choices = df_filings.sort_values(by=["n_frames", "filing_date"], ascending=[False, False])
        for _, row in sorted_choices.iterrows():
            acc = row["accession_number"]
            fdate = str(row["filing_date"])[:10]
            ftype = row["form_type"]
            nframes = int(row["n_frames"])
            label = f"{fdate} | {ftype} | Acc: {acc} ({nframes} AI frames)"
            filing_choices.append((label, acc))
        if filing_choices:
            default_acc = filing_choices[0][1]

    # 4. Activities
    sql_activities = f"""
    SELECT 
        fecha AS date,
        form,
        action,
        object,
        function,
        stage,
        provider_family,
        evidence_type
    FROM gold.datasets__activity__activity
    WHERE ticker = '{ticker}'
    ORDER BY fecha DESC
    LIMIT 100
    """
    df_activities, _, _, _ = run_query(sql_activities)

    return (
        overview_md,
        fig,
        gr.update(choices=filing_choices, value=default_acc),
        df_filings,
        df_activities,
    )


def fetch_filing_frames_cards(accession_number: str):
    """Fetch AI-classified paragraphs and format them into readable cards."""
    if not accession_number:
        return "Select a filing from the list to inspect its classified AI paragraphs."

    sql_frames = f"""
    SELECT 
        f.paragraph_index,
        f.subject,
        f.ai_type,
        f.temporal,
        array_to_string(f.rhetoric, ', ') AS rhetoric,
        array_to_string(f.specificity, ', ') AS specificity,
        p.paragraph_text
    FROM silver.ai_frames f
    LEFT JOIN bronze.unique_paragraphs p ON f.text_hash = p.text_hash
    WHERE f.accession_number = '{accession_number}'
    ORDER BY f.paragraph_index ASC
    LIMIT 50
    """
    df_frames, err, _, count = run_query(sql_frames)
    if err:
        return f"⚠️ Error retrieving paragraphs: {err}"

    if df_frames is None or df_frames.empty:
        return f"ℹ️ No AI disclosure paragraphs detected in filing `{accession_number}`."

    cards = [f"#### 📑 AI Disclosure Paragraphs in Filing `{accession_number}` ({count} frames)\n"]
    for idx, row in df_frames.iterrows():
        p_idx = row.get("paragraph_index", idx)
        subj = row.get("subject", "N/A")
        aitype = row.get("ai_type", "N/A")
        temporal = row.get("temporal", "N/A")
        rhetoric = row.get("rhetoric", "")
        text = row.get("paragraph_text", "") or "*(No text extracted)*"

        badges = [
            f"`Subject: {subj}`",
            f"`Type: {aitype}`",
            f"`Posture: {temporal}`",
        ]
        if rhetoric:
            badges.append(f"`Rhetoric: {rhetoric}`")

        card = f"""
---
##### Paragraph #{p_idx}
{" &nbsp;|&nbsp; ".join(badges)}

> {text}
"""
        cards.append(card)

    return "\n".join(cards)


def create_card_view():
    """Build Gradio UI components for View 2."""
    tickers_list = get_tickers_list()
    default_ticker = "AAPL" if any(t[1] == "AAPL" for t in tickers_list) else (tickers_list[0][1] if tickers_list else "")
    init_overview, init_fig, init_filings_upd, init_filings_df, init_act_df = fetch_card_details(default_ticker)

    with gr.Column():
        gr.Markdown(
            """
            ### 🏢 Company Profile & Filing Inspector
            Search any company in the S&P 500 panel to examine corporate posture, disclosure trajectory, and read **the full text** of AI paragraphs.
            """
        )

        with gr.Row():
            ticker_dropdown = gr.Dropdown(
                choices=tickers_list,
                value=default_ticker,
                label="Select Company / Ticker",
                info="Search by ticker symbol or company name",
                interactive=True,
                scale=3,
            )
            refresh_btn = gr.Button("🔄 Load Company Card", variant="primary", scale=1)

        overview_box = gr.Markdown(value=init_overview)

        with gr.Tabs():
            with gr.TabItem("📈 Historical Trajectory"):
                chart_history = gr.Plot(value=init_fig, label="AI Disclosure & Quality Trajectory")

            with gr.TabItem("📑 Filings & Paragraph Inspector"):
                gr.Markdown("#### Step 1: Select a Filing or Earnings Call to Inspect")
                with gr.Row():
                    filing_selector = gr.Dropdown(
                        choices=init_filings_upd.get("choices", []),
                        value=init_filings_upd.get("value"),
                        label="Filing (Accession Number)",
                        info="Ranked by number of detected AI frames",
                        interactive=True,
                        scale=3,
                    )
                    inspect_btn = gr.Button("🔎 Inspect AI Text", variant="secondary", scale=1)

                frames_card_box = gr.Markdown("Click 'Inspect AI Text' to view classified paragraphs...")

                with gr.Accordion("📋 All Filings Manifest for Ticker", open=False):
                    table_filings = gr.DataFrame(
                        value=init_filings_df,
                        datatype="auto",
                        wrap=True,
                        interactive=False,
                    )

            with gr.TabItem("⚡ Detected AI Activities"):
                gr.Markdown("#### Atomic AI Activities Classified for this Firm")
                table_activities = gr.DataFrame(
                    value=init_act_df,
                    datatype="auto",
                    wrap=True,
                    interactive=False,
                )

        # Event Handlers
        def update_card(ticker):
            ov, fig, fil_upd, fil_df, act_df = fetch_card_details(ticker)
            first_acc = fil_upd.get("value")
            first_cards = fetch_filing_frames_cards(first_acc) if first_acc else ""
            return ov, fig, fil_upd, fil_df, act_df, first_cards

        ticker_dropdown.change(
            update_card,
            inputs=[ticker_dropdown],
            outputs=[
                overview_box,
                chart_history,
                filing_selector,
                table_filings,
                table_activities,
                frames_card_box,
            ],
        )

        refresh_btn.click(
            update_card,
            inputs=[ticker_dropdown],
            outputs=[
                overview_box,
                chart_history,
                filing_selector,
                table_filings,
                table_activities,
                frames_card_box,
            ],
        )

        filing_selector.change(
            fetch_filing_frames_cards,
            inputs=[filing_selector],
            outputs=[frames_card_box],
        )

        inspect_btn.click(
            fetch_filing_frames_cards,
            inputs=[filing_selector],
            outputs=[frames_card_box],
        )

        # Preload initial filing text for default ticker
        if init_filings_upd.get("value"):
            frames_card_box.value = fetch_filing_frames_cards(init_filings_upd.get("value"))

    return ticker_dropdown
