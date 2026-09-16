"""La capa de actividades sobre qué actividades cuenta la misma empresa en la
call y en el filing del mismo ejercicio (`06`).

Insumos: `data/gold/covariates/document/activities.parquet` (conteos de
actividades por documento), sumados por (ticker, ejercicio fiscal, canal) del
spine de documentos. Determinístico, sin LLM.

  Brecha de actividades entre canales: misma empresa, mismo ejercicio fiscal,
  proporción de actividades de cada familia en la call menos la del filing,
  sobre celdas con ≥3 actividades en cada canal; por ejercicio. Figura
  `fig_brecha_actividades.png`.

Salida: `data/results/washing/activity_grounding.json` (clave `channel_gap`,
la única que lee `thesis.qmd`). Dentro de `channel_gap`, `pooled` trae la
brecha de todo el panel y `by_year` la misma brecha por ejercicio, cada una
con `gap_pp`, `n`, `t`, `p` y el IC del 95% (`ci_low`, `ci_high`) calculado
sobre las celdas de ese año.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

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
    doc = L.read_gold("document", ("covariates", "activities", GAP_FAMILIES + ["n_activities"]))
    ch = doc.groupby(["ticker", "fy", "channel"])[GAP_FAMILIES + ["n_activities"]].sum().reset_index()
    ch = ch[ch["n_activities"] > 0]
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
    # Mismo test de una muestra que las filas agrupadas, pero dentro de cada
    # ejercicio: la celda empresa × ejercicio es la unidad, así que el IC del
    # año sale de las celdas de ese año y no de un reescalado del agrupado.
    fiscal_years = [int(v) for v in sorted(gaps["fy"].unique())]
    by_year_stats: dict[str, dict[str, dict]] = {}
    for f in GAP_FAMILIES:
        per_year: dict[str, dict] = {}
        for fy in fiscal_years:
            x = gaps.loc[gaps["fy"] == fy, f].dropna()
            n = len(x)
            mean = 100 * float(x.mean()) if n else float("nan")
            if n < 2 or float(x.std(ddof=1)) == 0.0:
                per_year[str(fy)] = {"gap_pp": mean, "n": n, "t": None, "p": None,
                                     "ci_low": None, "ci_high": None}
                continue
            t, p = stats.ttest_1samp(x, 0.0)
            se = 100 * float(x.std(ddof=1)) / np.sqrt(n)
            crit = float(stats.t.ppf(0.975, n - 1))
            per_year[str(fy)] = {"gap_pp": mean, "n": n, "t": float(t), "p": float(p),
                                 "ci_low": mean - crit * se, "ci_high": mean + crit * se}
        by_year_stats[f] = per_year
    print(f"\n3. BRECHA DE ACTIVIDADES ENTRE CANALES — {n_cells} celdas empresa × ejercicio con ≥3 actividades en cada canal, {n_firms} empresas")
    print("   % de las actividades del canal en cada familia; brecha = call − filing, p.p.")
    print(tbl.round(2).to_string())
    print("\n   brecha por ejercicio (p.p.):"); print(by_year.round(1).to_string())
    print("\n   celdas por ejercicio:"); print(gaps.groupby("fy").size().to_string())
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
    fig.tight_layout(); fig.savefig(L.results_path("washing", "fig_brecha_actividades.png"), dpi=150); plt.close(fig)
    return {"n_cells": int(n_cells), "n_firms": int(n_firms), "pooled": json.loads(tbl.to_json(orient="index")),
            "by_year": by_year_stats}


def main() -> None:
    out = {"channel_gap": channel_activity_gap()}
    destination = L.results_path("washing", "activity_grounding.json")
    destination.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\n-> {destination}, fig_brecha_actividades.png")


if __name__ == "__main__":
    main()
