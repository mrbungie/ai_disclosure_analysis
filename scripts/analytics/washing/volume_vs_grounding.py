"""Volumen de actividades divulgadas vs. fundamentación, por ejercicio
(fig-volume-vs-grounding): el boom de IA generativa multiplicó el volumen de
actividades declaradas mientras diluía su fundamentación promedio.

Una fila por actividad (spine `activity` de gold con
`covariates/activity/{extraction, taxonomy}`), año fiscal-de-filing para
el canal `filing` (`silver.filing_manifest`/`silver.filing_manifest_10q`) y
año fiscal de la call (del `accession_number` sintético `TICKER_YYYYQnQ`
para el canal `call`), agregado 2021-2026 a:

  n_activities            volumen total de actividades divulgadas
  deployed_share_pct      % en producción o escaladas (`stage`)
  generic_evidence_pct    % con evidencia genérica, sin nombrar (`evidence_type`)
  named_evidence_pct      % con evidencia nombrada (producto/proveedor)
  identified_function_pct % con función de negocio identificada (`function_family`)

La proyección de año completo 2026 (factor estacional same-window) se deja
en el chunk de la tesis, que ya calcula ese factor (`hd_act_factor`) para
otras figuras -- este script sólo persiste los agregados observados.

Salida: `data/results/washing/volume_vs_grounding.parquet`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

YEARS = (2021, 2022, 2023, 2024, 2025, 2026)


def load_filing_dates() -> pd.DataFrame:
    main = L.scan("silver.filing_manifest").filter(pl.col("form_type") != "Earnings call transcript")
    tenq = L.scan("silver.filing_manifest_10q")
    cols = ["country_code", "accession_number", "filing_date"]
    return pl.concat([main.select(cols), tenq.select(cols)]).collect().to_pandas()


def main() -> None:
    act = L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))
    filings = load_filing_dates()

    act_f = act[act.channel == "filing"].merge(filings, on=["country_code", "accession_number"], how="left")
    act_f["year"] = pd.to_datetime(act_f["filing_date"]).dt.year
    act_c = act[act.channel == "call"].copy()
    act_c["year"] = act_c["accession_number"].str.extract(r"_(\d{4})Q")[0].astype(float)
    both = pd.concat([act_f, act_c], ignore_index=True)
    both = both[both["year"].between(YEARS[0], YEARS[-1])]
    both["year"] = both["year"].astype(int)

    out = pd.DataFrame({
        "n_activities": both.groupby("year").size(),
        "deployed_share_pct": both.groupby("year")["stage"].apply(lambda s: s.isin(["deployed", "scaled"]).mean() * 100),
        "generic_evidence_pct": both.groupby("year")["evidence_type"].apply(lambda s: (s == "generic").mean() * 100),
        "named_evidence_pct": both.groupby("year")["evidence_type"].apply(lambda s: (s == "named").mean() * 100),
        "identified_function_pct": both.groupby("year")["function_family"].apply(lambda s: (s != "unspecified").mean() * 100),
    }).reset_index().rename(columns={"index": "year"})

    print(out.round(2).to_string(index=False))
    destination = L.results_path("washing", "volume_vs_grounding.parquet")
    out.to_parquet(destination, index=False)
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
