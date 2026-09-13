"""La capa de actividades sobre qué actividades cuenta la misma empresa en la
call y en el filing del mismo ejercicio (`06`).

Insumos: `channel_activity_cells` (`activity_profiles.py`). Determinístico,
sin LLM.

  Brecha de actividades entre canales: misma empresa, mismo ejercicio fiscal,
  proporción de actividades de cada familia en la call menos la del filing,
  sobre celdas con ≥3 actividades en cada canal; por ejercicio. Figura
  `fig_brecha_actividades.png`.

Salida: `activity_grounding.json` (clave `channel_gap`, la única que lee
`thesis.qmd`).

2026-09-13 (docs/migration_v1_to_v2_analytics.md): eliminadas
`grid_grounding` (cruzaba contra `firm_voice_behavior_grid.parquet`, la
grilla voz×comportamiento de `build_voice_behavior_grid.py`) y
`washing_grounding` (leía `firm_washing_score_all.parquet`, un score de
washing beta-binomial superseded por `washing_score.py`'s índice W). Ninguna
de las dos escribía nada que `thesis.qmd` leyera (solo `["channel_gap"]`
se usa) -- caracterizar empresas por voz×comportamiento ya no es el método
vigente; eso lo hace el archetype de `build_strategy_dimensions.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
GAP_FAMILIES = ["customer_facing_deployment", "internal_deployment", "quantified_outcome", "infrastructure_investment",
                "third_party_named_provider", "proprietary_ai", "talent_or_training", "piloting_or_exploring",
                "governance_or_restriction", "named_product_or_process", "named_function"]
GAP_LABELS = {"customer_facing_deployment": "customer-facing deployment", "internal_deployment": "internal deployment",
              "quantified_outcome": "quantified outcome", "infrastructure_investment": "infrastructure investment",
              "third_party_named_provider": "named third-party provider", "proprietary_ai": "proprietary AI",
              "talent_or_training": "talent or training", "piloting_or_exploring": "piloting or exploring",
              "governance_or_restriction": "governance or restriction", "named_product_or_process": "named product or process",
              "named_function": "named business function"}


def channel_activity_gap() -> dict:
    ch = pd.read_parquet(OUT_DIR / "channel_activity_cells.parquet")
    ch = ch[ch["fy"].between(2021, 2026)]
    wide = ch.pivot_table(index=["ticker", "fy"], columns="channel", values=GAP_FAMILIES + ["n_activities"]).dropna()
    wide = wide[(wide[("n_activities", "call")] >= 3) & (wide[("n_activities", "filing")] >= 3)]
    gaps = pd.DataFrame(index=wide.index)
    for f in GAP_FAMILIES:
        gaps[f] = wide[(f, "call")] / wide[("n_activities", "call")] - wide[(f, "filing")] / wide[("n_activities", "filing")]
        gaps[f"{f}__call"] = wide[(f, "call")] / wide[("n_activities", "call")]
        gaps[f"{f}__filing"] = wide[(f, "filing")] / wide[("n_activities", "filing")]
    gaps = gaps.reset_index()
    n_cells, n_firms = len(gaps), gaps["ticker"].nunique()
    rows = []
    for f in GAP_FAMILIES:
        t, p = stats.ttest_1samp(gaps[f], 0.0)
        rows.append({"family": f, "call": 100 * gaps[f"{f}__call"].mean(), "filing": 100 * gaps[f"{f}__filing"].mean(),
                     "gap_pp": 100 * gaps[f].mean(), "t": float(t), "p": float(p)})
    tbl = pd.DataFrame(rows).set_index("family")
    by_year = (gaps.groupby("fy")[GAP_FAMILIES].mean() * 100).T
    print(f"\n3. BRECHA DE ACTIVIDADES ENTRE CANALES — {n_cells} celdas empresa × ejercicio con ≥3 actividades en cada canal, {n_firms} empresas")
    print("   % de las actividades del canal en cada familia; brecha = call − filing, p.p.")
    print(tbl.round(2).to_string())
    print("\n   brecha por ejercicio (p.p.):"); print(by_year.round(1).to_string())
    # figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = by_year.copy(); data["all"] = tbl["gap_pp"]
    data = data.loc[tbl["gap_pp"].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = float(np.abs(data.values).max())
    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(data.shape[1])); ax.set_xticklabels([str(c) if c != "all" else "2021–25" for c in data.columns])
    ax.set_yticks(range(data.shape[0])); ax.set_yticklabels([GAP_LABELS[f] for f in data.index], fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]; ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > vmax * 0.55 else "black")
    ax.set_xlabel("fiscal year"); ax.set_title("What the same firm says it does with AI: share of activities on the call minus in filings (pp)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="call − filing, percentage points of the channel's activities")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_brecha_actividades.png", dpi=150); plt.close(fig)
    return {"n_cells": int(n_cells), "n_firms": int(n_firms), "pooled": json.loads(tbl.to_json(orient="index")),
            "by_year": json.loads(by_year.round(2).to_json(orient="index"))}


def main() -> None:
    out = {"channel_gap": channel_activity_gap()}
    (OUT_DIR / "activity_grounding.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\n-> {OUT_DIR}/activity_grounding.json, fig_brecha_actividades.png")


if __name__ == "__main__":
    main()
