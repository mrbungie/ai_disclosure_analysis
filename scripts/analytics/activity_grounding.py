"""La capa de actividades sobre tres análisis existentes: qué acciones hay
detrás del eje de conducta de la grilla (`03`), qué actividad identificable
respalda el exceso promocional (`08`) y qué actividades cuenta la misma
empresa en la call y en el filing del mismo ejercicio (`06`).

Insumos: `firm_activities`, `firm_activity_profiles`, `channel_activity_cells`
(`activity_profiles.py`), `firm_voice_behavior_grid`, `firm_washing_score`,
`firm_year_master_v2`. Determinístico, sin LLM.

  1. Concreción conductual por celda de la grilla: media del score de
     concreción (función declarada, desplegada o escalada, producto con
     nombre, resultado cuantificado, proveedor nombrado; `activity_profiles.
     concreteness`) y % de empresas con al menos una actividad desplegada con
     producto o proceso nombrado. Entre las empresas de voz alta, las que
     describen actividad identificable contra las que sólo hacen afirmaciones
     estratégicas.
  2. Exceso promocional y respaldo conductual: para las colas del test de
     `08`, cuántas actividades identificables hay detrás; correlación entre el
     exceso y la concreción sobre las 449.
  3. Brecha de actividades entre canales: misma empresa, mismo ejercicio
     fiscal, proporción de actividades de cada familia en la call menos la del
     filing, sobre celdas con ≥3 actividades en cada canal; por ejercicio.
     Figura `fig_brecha_actividades.png`.

Salida: `activity_grounding.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity_profiles import ACTIVITY_FAMILIES, flags  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
CORNERS = {"washing (voz alta, conducta baja)": "desacople absoluto (voz alta, conducta baja)",
           "sustancia callada (voz baja, conducta alta)": "sustancia callada",
           "vocales sustantivos": "vocales sustantivos", "silenciosos": "silenciosos"}
GAP_FAMILIES = ["customer_facing_deployment", "internal_deployment", "quantified_outcome", "infrastructure_investment",
                "third_party_named_provider", "proprietary_ai", "talent_or_training", "piloting_or_exploring",
                "governance_or_restriction", "named_product_or_process", "named_function"]
GAP_LABELS = {"customer_facing_deployment": "customer-facing deployment", "internal_deployment": "internal deployment",
              "quantified_outcome": "quantified outcome", "infrastructure_investment": "infrastructure investment",
              "third_party_named_provider": "named third-party provider", "proprietary_ai": "proprietary AI",
              "talent_or_training": "talent or training", "piloting_or_exploring": "piloting or exploring",
              "governance_or_restriction": "governance or restriction", "named_product_or_process": "named product or process",
              "named_function": "named business function"}


def grounded_firms(a: pd.DataFrame) -> pd.Series:
    """Empresas con al menos una actividad desplegada o escalada CON producto o proceso nombrado."""
    f = flags(a)
    g = a["stage"].isin(["deployed", "scaled"]) & (a["evidence_strength"] == "named_product_or_process")
    return g.groupby(a["ticker"]).any()


def grid_grounding(prof: pd.DataFrame, a: pd.DataFrame) -> dict:
    grid = pd.read_parquet(OUT_DIR / "firm_voice_behavior_grid.parquet")[["ticker", "voz_shrunk", "comportamiento_shrunk", "nivel_voz", "nivel_conducta", "celda"]]
    m = grid.merge(prof[["ticker", "n_activities", "concrecion_conductual"]], on="ticker", how="left")
    m["n_activities"] = m["n_activities"].fillna(0).astype(int)
    m["grounded"] = m["ticker"].map(grounded_firms(a)).fillna(False).astype(bool)
    m["corner"] = m["celda"].map(CORNERS).fillna("resto de la grilla")
    m.loc[m["celda"] == "sin IA", "corner"] = "sin IA"
    rows = m.groupby("corner").agg(firms=("ticker", "size"), median_activities=("n_activities", "median"),
                                   mean_concreteness=("concrecion_conductual", "mean"), share_grounded=("grounded", "mean"),
                                   share_no_activity=("n_activities", lambda s: float((s == 0).mean())))
    order = ["desacople absoluto (voz alta, conducta baja)", "sustancia callada", "vocales sustantivos", "silenciosos", "resto de la grilla", "sin IA"]
    rows = rows.reindex(order)
    ok = m["concrecion_conductual"].notna()
    rho = stats.spearmanr(m.loc[ok, "comportamiento_shrunk"], m.loc[ok, "concrecion_conductual"])
    # voz alta: identificable contra estratégico
    hv = m[m["nivel_voz"] == "alta"].copy()
    med = hv["concrecion_conductual"].median()
    hv["tipo"] = np.where(hv["grounded"], "con actividad desplegada identificable", "sin actividad desplegada identificable")
    master = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")
    fin = master.groupby("ticker")[["rd_intensity", "beta", "ps_ratio", "frames_per_1k"]].median()
    hv = hv.merge(fin, left_on="ticker", right_index=True, how="left")
    hv_tbl = hv.groupby("tipo").agg(firms=("ticker", "size"), median_activities=("n_activities", "median"),
                                    mean_concreteness=("concrecion_conductual", "mean"), rd=("rd_intensity", "median"),
                                    beta=("beta", "median"), ps=("ps_ratio", "median"), frames_per_1k=("frames_per_1k", "median"),
                                    examples=("ticker", lambda s: ", ".join(hv.loc[s.index].sort_values("n_activities", ascending=False)["ticker"].head(8))))
    corner_hv = hv[hv["corner"] == order[0]]
    print("1. GRILLA — concreción conductual por esquina")
    print(rows.round(3).to_string())
    print(f"   Spearman(eje de conducta, concreción) = {rho.statistic:+.3f} (p={rho.pvalue:.1e}, n={int(ok.sum())})")
    print(f"\n   Voz alta ({len(hv)} empresas): mediana de concreción {med:.2f}")
    print(hv_tbl.round(3).to_string())
    print(f"   En la esquina de desacople absoluto: {int(corner_hv['grounded'].sum())} de {len(corner_hv)} tienen alguna actividad desplegada con producto nombrado; "
          f"{int((corner_hv['n_activities'] == 0).sum())} no tienen ninguna actividad")
    print("   desacople sin actividad identificable:", ", ".join(corner_hv[~corner_hv.grounded].sort_values("n_activities")["ticker"].tolist()))
    print("   desacople con actividad identificable:", ", ".join(corner_hv[corner_hv.grounded].sort_values("n_activities", ascending=False)["ticker"].tolist()))
    return {"by_corner": json.loads(rows.to_json(orient="index")), "spearman_conduct_concreteness": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
            "high_voice": json.loads(hv_tbl.to_json(orient="index")),
            "decoupled_corner": {"grounded": corner_hv[corner_hv.grounded]["ticker"].tolist(), "not_grounded": corner_hv[~corner_hv.grounded]["ticker"].tolist()}}


def washing_grounding(prof: pd.DataFrame, a: pd.DataFrame) -> dict:
    w = pd.read_parquet(OUT_DIR / "firm_washing_score.parquet")[["ticker", "n_frames", "exceso", "tasa_obs", "p_esperada", "washing", "callada"]]
    m = w.merge(prof[["ticker", "n_activities", "concrecion_conductual", "share_named_product_or_process", "share_quantified_outcome", "share_deployed_or_scaled"]], on="ticker", how="left")
    m["n_activities"] = m["n_activities"].fillna(0).astype(int)
    m["grounded"] = m["ticker"].map(grounded_firms(a)).fillna(False).astype(bool)
    g = a[a["stage"].isin(["deployed", "scaled"]) & (a["evidence_strength"] == "named_product_or_process")].groupby("ticker").size()
    m["n_grounded_activities"] = m["ticker"].map(g).fillna(0).astype(int)
    ok = m["concrecion_conductual"].notna()
    rho = stats.spearmanr(m.loc[ok, "exceso"], m.loc[ok, "concrecion_conductual"])
    rho_n = stats.spearmanr(m.loc[ok, "exceso"], np.log1p(m.loc[ok, "n_grounded_activities"]))
    tails = m[m["washing"] | m["callada"]].copy()
    tails["cola"] = np.where(tails["washing"], "washing", "sustancia callada")
    tails = tails.sort_values(["cola", "exceso"], ascending=[False, False])
    cols = ["ticker", "cola", "n_frames", "tasa_obs", "p_esperada", "exceso", "n_activities", "n_grounded_activities", "concrecion_conductual"]
    print("\n2. EXCESO PROMOCIONAL Y RESPALDO CONDUCTUAL")
    print(tails[cols].round(3).to_string(index=False))
    print(f"   Spearman(exceso, concreción) = {rho.statistic:+.3f} (p={rho.pvalue:.3f}); Spearman(exceso, log actividades desplegadas con nombre) = {rho_n.statistic:+.3f} (p={rho_n.pvalue:.1e}); n={int(ok.sum())}")
    # entre las 449: promoción alta con y sin respaldo
    m["promo_alto"] = m["tasa_obs"] >= m["tasa_obs"].quantile(0.8)
    hp = m[m["promo_alto"]]
    split = hp.groupby("grounded").agg(firms=("ticker", "size"), median_activities=("n_activities", "median"), mean_concreteness=("concrecion_conductual", "mean"),
                                       examples=("ticker", lambda s: ", ".join(hp.loc[s.index].sort_values("n_activities", ascending=False)["ticker"].head(8))))
    print(f"   Quintil superior de tasa promocional ({len(hp)} empresas): con y sin actividad desplegada con producto nombrado")
    print(split.round(3).to_string())
    return {"tails": json.loads(tails[cols].to_json(orient="records")), "spearman_excess_concreteness": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
            "spearman_excess_grounded_n": {"rho": float(rho_n.statistic), "p": float(rho_n.pvalue)},
            "top_promo_quintile": json.loads(split.to_json(orient="index"))}


def channel_activity_gap() -> dict:
    ch = pd.read_parquet(OUT_DIR / "channel_activity_cells.parquet")
    ch = ch[ch["fy"].between(2021, 2025)]
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
    a = pd.read_parquet(OUT_DIR / "firm_activities.parquet")
    prof = pd.read_parquet(OUT_DIR / "firm_activity_profiles.parquet")
    out = {"grid": grid_grounding(prof, a), "washing": washing_grounding(prof, a), "channel_gap": channel_activity_gap()}
    (OUT_DIR / "activity_grounding.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\n-> {OUT_DIR}/activity_grounding.json, fig_brecha_actividades.png")


if __name__ == "__main__":
    main()
