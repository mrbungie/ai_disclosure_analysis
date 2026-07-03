"""
12_event_study_panel.py — Event study / DiD panel builder

Produces `data/processed/panels/event_study_panel.parquet` with:
- Firm-year observations enriched with event indicators and treatment flags
- Pre/post windows for ChatGPT (2022), SEC guidance (2024), DeepSeek (2025)
- Treatment group indicators based on 2023 baseline disclosure style

Also runs a power check (`reports/did_power_check.txt`): with 115 firms across
54 SIC industry groups, industry x year x treatment cells used in a DiD design
get small fast. Single-flag treatment counts can look adequate while the actual
regression cells (e.g. treatment x tech-sector x pre/post) are underpowered.
The check reports firm counts for each treatment flag alone AND for the
flag combinations actually used in event-specific DiD specs, flagging any
cell below MIN_CELL_FIRMS so that shows up before the DiD is run, not after.

Usage:
    uv run python scripts/12_event_study_panel.py
"""
import json
import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path

try:
    import pipeline_logger
except ImportError:
    from scripts import pipeline_logger


def load_config():
    with open("configs/config.json") as f:
        return json.load(f)


EVENTS = {
    "chatgpt":     {"year": 2022, "pre_years": [2021],          "post_years": [2023, 2024]},
    "sec_2024":    {"year": 2024, "pre_years": [2021, 2022, 2023], "post_years": [2024, 2025]},
    "deepseek":    {"year": 2025, "pre_years": [2022, 2023, 2024], "post_years": [2025, 2026]},
}

# Treatment group definitions (based on pre-event baseline characteristics)
# Each entry: column to use, threshold for "high" group, description
TREATMENT_DEFS = {
    "high_promotional_pre2024":  ("share_promotional",  "high",  "High promotional tone in 2023 (SEC treatment)"),
    "high_technical_pre2024":    ("D3 Technical",        "high",  "High technical specificity in 2023"),
    "high_defensive_pre2025":    ("D8 Defensive",        "high",  "High defensive framing in 2024 (DeepSeek treatment)"),
    "tech_sector":               ("industry_group",      "tech",  "Tech & Software sector"),
}

# Minimum number of distinct firms required in a DiD regression cell before we
# trust the resulting standard errors. Below this, prefer collapsing/simplifying
# the treatment definition (e.g. drop the tech_sector interaction) or merging
# event windows rather than running the DiD as specified.
MIN_CELL_FIRMS = 15

# Flag combinations actually intended for use in event-specific DiD specs
# (not just single flags) — these are the cells that matter for power.
DID_CELL_SPECS = {
    "sec_2024 DiD":     ["high_promotional_pre2024"],
    "sec_2024 DiD (tech-interacted)": ["high_promotional_pre2024", "tech_sector"],
    "deepseek DiD":     ["high_defensive_pre2025"],
    "deepseek DiD (tech-interacted)": ["high_defensive_pre2025", "tech_sector"],
    "technical x tech-sector": ["high_technical_pre2024", "tech_sector"],
}


def build_panel(features_path, clusters_path, output_path):
    features = pd.read_parquet(features_path)
    clusters = pd.read_parquet(clusters_path)

    # Merge cluster assignments
    df = features.merge(
        clusters[["ticker", "year", "cluster", "cluster_name"]],
        on=["ticker", "year"], how="left"
    )

    # ── Event indicators ──────────────────────────────────────────────────────
    for event, cfg in EVENTS.items():
        df[f"post_{event}"]    = (df["year"] >= cfg["year"]).astype(int)
        df[f"window_{event}"]  = df["year"].isin(cfg["pre_years"] + cfg["post_years"]).astype(int)
        df[f"pre_{event}"]     = df["year"].isin(cfg["pre_years"]).astype(int)

    # ── Treatment group flags ─────────────────────────────────────────────────
    # Assign based on 2023 (pre-SEC) baseline to avoid contamination
    base_year = 2023
    base = df[df["year"] == base_year].copy()

    dim_cols = [c for c in clusters.columns if c.startswith("D") and c[1].isdigit()]

    for flag, (col, rule, _) in TREATMENT_DEFS.items():
        if rule == "high":
            if col in base.columns:
                threshold = base[col].median()
                high_tickers = base.loc[base[col] >= threshold, "ticker"].unique()
                df[flag] = df["ticker"].isin(high_tickers).astype(int)
            else:
                df[flag] = np.nan
        elif rule == "tech":
            tech_sic = [g for g in df["industry_group"].unique()
                        if any(k in g for k in ["Software", "Semiconductor", "Computer", "Electronic"])]
            df[flag] = df["industry_group"].isin(tech_sic).astype(int)

    # ── Firm fixed effects (firm ID encoding) ─────────────────────────────────
    firm_ids = {t: i for i, t in enumerate(sorted(df["ticker"].unique()))}
    df["firm_id"] = df["ticker"].map(firm_ids)

    # ── Relative time variables (event-relative year) ─────────────────────────
    for event, cfg in EVENTS.items():
        df[f"rel_year_{event}"] = df["year"] - cfg["year"]

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    pipeline_logger.log_event(
        pipeline_step="event_study_panel",
        level="SUCCESS",
        message=f"Event study panel saved: {len(df)} rows, {len(df.columns)} cols → {output_path}"
    )
    print(f"Saved {len(df)} firm-year observations to {output_path}")
    print(f"Columns: {len(df.columns)}")
    print("\nEvent window sizes:")
    for event in EVENTS:
        w = df[f"window_{event}"].sum()
        pre = df[f"pre_{event}"].sum()
        post = df[df[f"window_{event}"] == 1][f"post_{event}"].sum()
        print(f"  {event}: {w} obs total  ({pre} pre / {post} post)")

    print("\nTreatment group sizes (full panel):")
    for flag in TREATMENT_DEFS:
        if flag in df.columns and df[flag].notna().any():
            n = int(df[flag].sum())
            n_firms = df[df[flag] == 1]["ticker"].nunique()
            print(f"  {flag}: {n} firm-years  ({n_firms} firms)")

    run_power_check(df)

    return df


def run_power_check(df: pd.DataFrame, out_path: Path = Path("reports/did_power_check.txt")) -> None:
    """
    Report distinct-firm counts for each DiD treatment cell before the DiD is run.

    Single-flag counts (printed above) can look adequate while the actual
    regression cell used in a given DiD spec — e.g. high_promotional_pre2024
    interacted with tech_sector — has very few firms, producing unreliable
    standard errors. This flags any such cell below MIN_CELL_FIRMS so the
    spec can be simplified (drop the interaction, merge event windows, or
    widen the "high" threshold) before results are reported.
    """
    lines = [
        "DiD Power Check",
        "=" * 50,
        f"Panel: {df['ticker'].nunique()} firms, {len(df)} firm-years",
        f"Minimum firms per cell (MIN_CELL_FIRMS): {MIN_CELL_FIRMS}",
        "",
    ]
    underpowered = []

    for spec_name, flags in DID_CELL_SPECS.items():
        missing = [f for f in flags if f not in df.columns or df[f].isna().all()]
        if missing:
            lines.append(f"{spec_name}: SKIPPED (missing flags: {', '.join(missing)})")
            continue

        mask = np.ones(len(df), dtype=bool)
        for f in flags:
            mask &= (df[f] == 1).values
        n_firms = df.loc[mask, "ticker"].nunique()
        n_firm_years = int(mask.sum())
        flagged = n_firms < MIN_CELL_FIRMS
        status = "UNDERPOWERED" if flagged else "ok"
        lines.append(
            f"{spec_name} [{' x '.join(flags)}]: {n_firms} firms, "
            f"{n_firm_years} firm-years — {status}"
        )
        if flagged:
            underpowered.append(spec_name)

    lines.append("")
    if underpowered:
        lines.append(
            f"WARNING: {len(underpowered)} cell(s) below {MIN_CELL_FIRMS} firms: "
            f"{', '.join(underpowered)}. Consider simplifying the treatment "
            f"definition (drop the interaction term) or merging event windows "
            f"before running the DiD on these specs."
        )
    else:
        lines.append("All checked DiD cells meet the minimum firm-count threshold.")

    summary = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary)
    print(f"\n{summary}\n")
    print(f"Power check written to {out_path}")

    pipeline_logger.log_event(
        pipeline_step="event_study_panel",
        level="WARNING" if underpowered else "INFO",
        message=(
            f"DiD power check: {len(underpowered)} underpowered cell(s) "
            f"out of {len(DID_CELL_SPECS)} specs checked."
        ),
    )


def main():
    config = load_config()
    features_path = Path("data/processed/features/firm_year_features.parquet")
    clusters_path = Path("data/processed/clusters/firm_year_clusters.parquet")
    output_path   = Path("data/processed/panels/event_study_panel.parquet")

    for p in [features_path, clusters_path]:
        if not p.exists():
            print(f"Error: missing {p}. Run pipeline scripts first.")
            return

    build_panel(features_path, clusters_path, output_path)


if __name__ == "__main__":
    main()
