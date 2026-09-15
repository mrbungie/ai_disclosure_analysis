"""Diffusión de la divulgación de IA por ejercicio (fig-ai-diffusion):
participación (`any_ai`) e intensidad narrativa (`frames_per_1k`), media
sobre todas las empresas-año con filings, 2021-2026.

Fuente: `data/gold/covariates/firm_year/disclosure_volume.parquet`.

Salida: `data/results/washing/ai_diffusion_by_year.parquet`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402


def main() -> None:
    m = L.read_gold("firm_year", ("covariates", "disclosure_volume", ["any_ai", "frames_per_1k"]))
    y = m.groupby("year").agg(any_ai=("any_ai", "mean"), frames=("frames_per_1k", "mean")).reset_index()
    print(y.round(4).to_string(index=False))
    destination = L.results_path("washing", "ai_diffusion_by_year.parquet")
    y.to_parquet(destination, index=False)
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
