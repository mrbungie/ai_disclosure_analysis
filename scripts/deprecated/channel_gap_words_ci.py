"""CI firm-clustered para la brecha call-vs-filing en palabras (Cap. 6, Fig. 12).

Complementa `channel_gap_words_robustness.py`: ese script reporta medias y
ratio, este agrega el 95% CI (bootstrap, clusterizado por firma) de la
diferencia dentro de celda (call - filing) para promo y quant per 1,000
palabras, que la figura necesita para mostrar un coefficient plot en vez de
solo dos puntos.

Determinístico salvo la semilla del bootstrap (fija). Requiere duckdb/thesis.duckdb.

Salida: `data/processed/clusters/channel_gap_words_ci.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_gap_analysis import load_frames, DB  # noqa: E402
from channel_gap_words_robustness import load_documents_words  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
N_BOOT = 2000
SEED = 0


def firm_clustered_ci(diff: pd.Series, firm: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    firms = firm.unique()
    by_firm = diff.groupby(firm).apply(list)
    point = diff.mean()
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample_firms = rng.choice(firms, size=len(firms), replace=True)
        vals = np.concatenate([by_firm[f] for f in sample_firms])
        boot_means[b] = vals.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    frames = load_frames(con)
    cell = load_documents_words(con, frames)

    outcomes = ["promo_per_1k_words", "quant_per_1k_words"]
    wide = cell.pivot(index=["ticker", "fy"], columns="channel", values=outcomes)
    wide.columns = [f"{v}_{ch}" for v, ch in wide.columns]
    wide = wide.dropna(subset=[f"{outcomes[0]}_call", f"{outcomes[0]}_filing"]).reset_index()

    result = {"n_cells": int(len(wide)), "n_firms": int(wide["ticker"].nunique())}
    for col in outcomes:
        diff = wide[f"{col}_call"] - wide[f"{col}_filing"]
        point, lo, hi = firm_clustered_ci(diff, wide["ticker"])
        result[col] = {"diff_mean": point, "ci_lo": lo, "ci_hi": hi}
        label = "promotional" if col.startswith("promo") else "quantified"
        print(f"{label:12s} call-filing diff per 1,000 words: {point:.4f} [{lo:.4f}, {hi:.4f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "channel_gap_words_ci.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
