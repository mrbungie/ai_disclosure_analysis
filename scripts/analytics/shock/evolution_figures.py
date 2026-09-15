"""Figuras finales del descriptivo temporal (bloque 1).

  fig_evolucion_intensidad.png   frames de IA por 1.000 palabras y % de empresas
                                 con algún frame, por año, todas las empresas
                                 con filings (cero si no hablan)
  fig_evolucion_composicion.png  promocional, riesgo, gobernanza, despliegue y
                                 realizado, por 1.000 palabras, por año

Ejercicios 2021-2025 en el cuerpo; 2026 se dibuja punteado como "YTD".
Fuente: spine firm_year + covariates/firm_year/disclosure_volume.
Salida en `data/results/shock/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402


def evolution() -> None:
    m = L.read_gold("firm_year", ("covariates", "disclosure_volume"))
    y = m.groupby("year").agg(any_ai=("any_ai", "mean"), frames=("frames_per_1k", "mean"), promo=("promo_per_1k", "mean"),
                              risk=("risk_per_1k", "mean"), gov=("gov_per_1k", "mean"), deployed=("deployed_per_1k", "mean"),
                              realized=("realized_per_1k", "mean"), n=("ticker", "nunique"))
    main, ytd = y.loc[y.index <= 2025], y.loc[y.index >= 2025]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(main.index, main.frames, "o-", color="#1f77b4", label="AI frames per 1,000 words")
    ax.plot(ytd.index, ytd.frames, "o:", color="#1f77b4")
    ax.set_ylabel("AI frames per 1,000 words"); ax.set_xlabel("filing year")
    ax2 = ax.twinx()
    ax2.plot(main.index, main.any_ai * 100, "s-", color="#d62728", label="% of firms with any AI frame")
    ax2.plot(ytd.index, ytd.any_ai * 100, "s:", color="#d62728"); ax2.set_ylabel("% of firms with any AI frame"); ax2.set_ylim(0, 100)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.set_title("AI disclosure intensity in SEC filings, all firms (2026 = YTD, dotted)", fontsize=10)
    fig.tight_layout(); fig.savefig(L.results_path("shock", "fig_evolucion_intensidad.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, label, color in (("promo", "promotional", "#ff7f0e"), ("risk", "risk", "#d62728"), ("gov", "governance", "#2ca02c"),
                              ("deployed", "deployment", "#1f77b4"), ("realized", "realized", "#9467bd")):
        ax.plot(main.index, main[col], "o-", color=color, label=label); ax.plot(ytd.index, ytd[col], "o:", color=color)
    ax.set_ylabel("frames per 1,000 words"); ax.set_xlabel("filing year"); ax.legend(fontsize=9)
    ax.set_title("Composition: volume grew, the mix changed more (risk and governance > promotion)", fontsize=10)
    fig.tight_layout(); fig.savefig(L.results_path("shock", "fig_evolucion_composicion.png"), dpi=150); plt.close(fig)
    print(y.round(3).to_string())


if __name__ == "__main__":
    evolution()
    out_dir = L.RESULTS / "shock"
    print(f"-> {out_dir}/fig_evolucion_intensidad.png, fig_evolucion_composicion.png")
