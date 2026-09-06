"""Figuras finales del descriptivo temporal (bloque 1) y del event study de la
SEC (bloque 7).

  fig_evolucion_intensidad.png   frames de IA por 1.000 párrafos y % de empresas
                                 con algún frame, por año, todas las empresas
                                 con filings (cero si no hablan)
  fig_evolucion_composicion.png  promocional, riesgo, gobernanza, despliegue y
                                 realizado, por 1.000 párrafos, por año
  fig_sec_event_study.png        coeficientes del event study de shock_analysis
                                 (exposición × trimestre), promocionales por
                                 1.000 párrafos, con IC 95%

Ejercicios 2021-2025 en el cuerpo; 2026 se dibuja punteado como "YTD".
Salida en `data/processed/clusters/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"


def evolution() -> None:
    m = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")
    y = m.groupby("year").agg(any_ai=("any_ai", "mean"), frames=("frames_per_1k", "mean"), promo=("promo_per_1k", "mean"),
                              risk=("risk_per_1k", "mean"), gov=("gov_per_1k", "mean"), deployed=("deployed_per_1k", "mean"),
                              realized=("realized_per_1k", "mean"), n=("ticker", "nunique"))
    main, ytd = y.loc[y.index <= 2025], y.loc[y.index >= 2025]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(main.index, main.frames, "o-", color="#1f77b4", label="frames de IA por 1.000 párrafos")
    ax.plot(ytd.index, ytd.frames, "o:", color="#1f77b4")
    ax.set_ylabel("frames de IA por 1.000 párrafos"); ax.set_xlabel("año de presentación")
    ax2 = ax.twinx()
    ax2.plot(main.index, main.any_ai * 100, "s-", color="#d62728", label="% empresas con algún frame")
    ax2.plot(ytd.index, ytd.any_ai * 100, "s:", color="#d62728"); ax2.set_ylabel("% de empresas con algún frame de IA"); ax2.set_ylim(0, 100)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.set_title("Intensidad de la divulgación de IA en filings SEC, todas las empresas (2026 = YTD, punteado)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_evolucion_intensidad.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, label, color in (("promo", "promocional", "#ff7f0e"), ("risk", "riesgo", "#d62728"), ("gov", "gobernanza", "#2ca02c"),
                              ("deployed", "despliegue", "#1f77b4"), ("realized", "realizado", "#9467bd")):
        ax.plot(main.index, main[col], "o-", color=color, label=label); ax.plot(ytd.index, ytd[col], "o:", color=color)
    ax.set_ylabel("frames por 1.000 párrafos"); ax.set_xlabel("año de presentación"); ax.legend(fontsize=9)
    ax.set_title("Composición: la cantidad creció, la mezcla cambió más (riesgo y gobernanza > promoción)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_evolucion_composicion.png", dpi=150); plt.close(fig)
    print(y.round(3).to_string())


def sec_event_study() -> None:
    d = json.load(open(OUT_DIR / "shock_analysis.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, (outcome, title) in zip(axes, (("promo_per_1k", "promocionales por 1.000 párrafos"), ("risk_per_1k", "riesgo por 1.000 párrafos"))):
        o = d["sec"]["outcomes"][outcome]; t = pd.DataFrame(o["coefficients"]).sort_values("event_time")
        pre = t.event_time < 0
        ax.errorbar(t.event_time[pre], t.coef[pre], yerr=1.96 * t.se[pre], fmt="o", color="#4c72b0", capsize=3, label="antes")
        ax.errorbar(t.event_time[~pre], t.coef[~pre], yerr=1.96 * t.se[~pre], fmt="s", color="#c44e52", capsize=3, label="después")
        ax.axhline(0, color="grey", lw=1); ax.axvline(-0.5, color="black", ls=":", lw=1)
        ax.set_title(f"{title}\ntendencias previas: p = {o['pretrend_p']:.3f}", fontsize=10)
        ax.set_xlabel("trimestres desde 2024Q1 (t−1 = referencia)"); ax.set_ylabel("exposición × trimestre")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Event study SEC: las empresas más expuestas ya divergían antes del aviso", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_sec_event_study.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    evolution(); sec_event_study()
    print(f"-> {OUT_DIR}/fig_evolucion_intensidad.png, fig_evolucion_composicion.png, fig_sec_event_study.png")
