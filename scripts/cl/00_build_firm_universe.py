"""
scripts/cl/00_build_firm_universe.py — builds the Chile firm universe from
the public empresas-cmf-chile catalog (github.com/JoaquinMulet/empresas-cmf-chile),
the same catalog mcp-cmf-chile itself uses to resolve tickers to RUTs.

Universe = "all CMF equity issuers" (per docs/international_expansion_plan.md
— small enough that narrowing it loses data without saving meaningful
download/compute time, same reasoning as NOT restricting Chile to IPSA).
TIPO_ENTIDAD == EMISOR or BANCO only — ETF (34 in the raw catalog) and
VALOR_EXTRANJERO (7, foreign securities cross-listed, not CMF-regulated
Chilean filers) are excluded: neither writes the Memoria Anual / Análisis
Razonado narrative disclosures this project extracts.

Writes configs/cl/universe.csv + configs/cl/universe_membership.csv (same
shape as configs/us/, see scripts/us/docs/universe_expansion_plan.md) and
data/interim/manifests_cl/firm_universe.parquet.

Usage:
    uv run python scripts/cl/00_build_firm_universe.py
"""

import io
from pathlib import Path

import pandas as pd
import requests
import yaml

CATALOG_URL = "https://raw.githubusercontent.com/JoaquinMulet/empresas-cmf-chile/master/empresas_chile.csv"
INCLUDED_TIPO_ENTIDAD = {"EMISOR", "BANCO"}


def _rut_digits(rut: str) -> str:
    """'93007000-9' -> '93007000' — the digit part alone is what every
    mcf-mcp-chile tool call needs (its rutSchema strips the DV itself,
    but normalizing once here keeps our own manifest/universe keys clean)."""
    return rut.split("-")[0].replace(".", "").strip()


def main():
    with open("configs/cl/config.yaml") as f:
        config = yaml.safe_load(f)

    resp = requests.get(CATALOG_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep=";")
    df.columns = [c.strip().upper() for c in df.columns]

    df = df[df["TIPO_ENTIDAD"].isin(INCLUDED_TIPO_ENTIDAD)].copy()
    df["rut"] = df["RUT"].apply(_rut_digits)
    df = df.drop_duplicates(subset=["rut"])

    universe = pd.DataFrame({
        "nemo": df["NEMO"],
        "rut": df["rut"],
        "rut_dv": df["RUT"],
        "razon_social": df["RAZON_SOCIAL"],
        "isin": df["ISIN"],
        "tipo_entidad": df["TIPO_ENTIDAD"],
        "norma": df["NORMA"],
        "active_status": "active",
    }).sort_values("nemo")

    membership = pd.DataFrame({
        "rut": universe["rut"],
        "group": universe["tipo_entidad"].map({"EMISOR": "cmf_emisor", "BANCO": "cmf_banco"}),
    })

    universe_path = Path("configs/cl/universe.csv")
    membership_path = Path("configs/cl/universe_membership.csv")
    universe.to_csv(universe_path, index=False)
    membership.to_csv(membership_path, index=False)

    manifest_dir = Path(config["storage"]["interim_manifests"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(manifest_dir / "firm_universe.parquet", index=False)

    print(f"Wrote {len(universe)} firms ({(universe['tipo_entidad'] == 'EMISOR').sum()} EMISOR, "
          f"{(universe['tipo_entidad'] == 'BANCO').sum()} BANCO) -> {universe_path}, {membership_path}, "
          f"{manifest_dir / 'firm_universe.parquet'}")


if __name__ == "__main__":
    main()
