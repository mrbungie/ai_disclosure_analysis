"""Rebuilds the firm-level AI-disclosure clusters and the firm-year panel
from `gold_ai_frames` / `gold_ai_entity_mentions`.

This script did not exist. The parquets under `data/processed/clusters/`
were produced ad hoc in an earlier session and only the OUTPUTS survived —
`docs/analytics/01_ai_disclosure_analytics.md` and `06_voice_vs_behavior_
clustering.md` describe the method in prose, with abbreviated Python, and
that prose plus the stored column schemas is what this reconstructs. It is
therefore NOT guaranteed to reproduce the original numbers bit for bit:
K-means on a changed population re-fits its centroids, so cluster
membership and the archetype letters can legitimately move.

Three artifacts, in dependency order:

1. `firm_behavior_clusters.parquet` — K-means over 15 BEHAVIOR features
   (what firms say they DO), one row per firm.
2. `voice_x_behavior.parquet` — the voice archetype crossed against that
   behavior cluster. Two independent clusterings, deliberately: voice
   metrics are all rhetoric, so clustering them and calling the result
   "behavior" was the conceptual bug 06_... was written to correct.
3. `firm_year_archetype_behaviors.parquet` — the same voice metrics
   recomputed per (ticker, year) and PROJECTED onto the pooled centroids,
   never re-clustered per year, so "this firm's 2023 archetype" is
   comparable across years.

Cluster ids are arbitrary in K-means — a re-fit renumbers them. Both
labelings here are therefore assigned from cluster PROFILES (see
`_label_voice_archetypes` / `_label_behavior_clusters`), not from the
integer sklearn happened to hand out, so "D" keeps meaning "vocal leader"
across re-runs instead of silently becoming a different group.

Population: firms with >=5 frames pooled (>=3 per year for the panel), US
only. Forms follow `filing_manifest` — 10-K, DEF 14A and 8-K. The 10-Q is
NOT pooled in: it is a separate instrument with its own manifest by an
explicit project-scope decision (see README), not an oversight.

Deterministic and free: no LLM call, no API. It reads artifacts that DID
cost LLM calls (`gold_ai_frames`) and never writes to them.

Usage:
    uv run python scripts/analytics/build_firm_clusters.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"

POOLED_MIN_FRAMES = 5   # docs/analytics/01, "Método"
PANEL_MIN_FRAMES = 3    # docs/analytics/01, "Construcción del panel"
SEED = 42

SPECIFICITY_FLAGS = [
    "specificity_business_process", "specificity_product_or_system",
    "specificity_vendor_or_partner", "specificity_quantified_metric",
    "specificity_date_or_timeline",
]
VOICE_FEATURES = [
    "specificity_index", "quantified_rate", "promotional_rate", "strategic_rate",
    "realized_share", "hypothetical_share", "risk_share", "gov_share",
    "firm_subject_share",
]
# `use_stage_unspecified` is deliberately absent: it is the residual bucket,
# so including it would let "we didn't say" act as a behavior of its own.
BEHAVIOR_CONCEPTS = [
    "deployed", "pilot_or_testing", "exploring", "ai_investment",
    "ai_infrastructure", "ai_talent", "proprietary_ai", "third_party_ai",
    "expansion_or_scaling", "productivity_outcome", "revenue_outcome",
    "cost_outcome", "customer_outcome",
]
BEHAVIOR_FEATURES = BEHAVIOR_CONCEPTS + ["domain_customer_facing", "domain_internal"]
# The panel carries the residual too — there it is descriptive, not an input.
PANEL_BEHAVIOR_CONCEPTS = BEHAVIOR_CONCEPTS + ["use_stage_unspecified"]
DOMAINS = ["customer_facing", "internal", "unspecified"]


def load_frames(con) -> pd.DataFrame:
    """One row per frame, with the filing's ticker/cik/year attached.

    The JOIN to `filing_manifest` is what scopes this to 10-K + DEF 14A +
    8-K: the 10-Q lives in `filing_manifest_10q` and is a separate
    instrument by project scope, so it drops out here by construction
    rather than by a filter someone has to remember to write."""
    return con.execute("""
        SELECT fm.ticker, fm.cik, fm.form_type, fm.accession_number,
               extract(year from fm.filing_date)::INT AS year,
               f.text_hash, f.frame_index, f.subject, f.temporal, f.domain, f.concepts,
               f.rhetoric_promotional, f.rhetoric_strategic_importance,
               f.specificity_business_process, f.specificity_product_or_system,
               f.specificity_vendor_or_partner, f.specificity_quantified_metric,
               f.specificity_date_or_timeline
        FROM gold_ai_frames f
        JOIN filing_manifest fm USING (country_code, accession_number)
        WHERE f.country_code = 'us' AND f.has_frame AND fm.ticker IS NOT NULL
        -- ORDER BY is load-bearing, not cosmetic. Without it DuckDB returns
        -- rows in whatever order its parallel scan produces, which varies
        -- run to run; the per-firm means below then sum the same floats in a
        -- different order and land on different last digits. That was enough
        -- to move the K-means centroids and reshuffle cluster labels between
        -- two runs of an otherwise seeded, deterministic script.
        ORDER BY fm.ticker, fm.accession_number, f.text_hash, f.frame_index
    """).fetchdf()


def load_entity_mentions(con) -> pd.DataFrame:
    """Named-entity mentions per (ticker, year), restricted to texts the
    judge actually granted an AI frame.

    That restriction is not cosmetic. `gold_ai_entity_mentions` is literal
    regex with no semantic check, and DEF 14A broke it: 191 proxy texts
    match `claude` because directors are NAMED Claude, and only 5 of them
    carry an AI frame. Counting raw matches here would put a director's
    first name into a firm's AI profile."""
    return con.execute("""
        SELECT fm.ticker, extract(year from fm.filing_date)::INT AS year,
               m.text_hash, m.term
        FROM gold_ai_entity_mentions m
        JOIN filing_manifest fm USING (country_code, accession_number)
        WHERE m.country_code = 'us' AND fm.ticker IS NOT NULL
          AND EXISTS (SELECT 1 FROM gold_ai_frames f
                      WHERE f.text_hash = m.text_hash AND f.has_frame)
        ORDER BY fm.ticker, year, m.text_hash, m.term   -- ver load_frames()
    """).fetchdf()


def voice_metrics(frames: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The 9 rhetoric metrics of docs/analytics/01 §Método, averaged over
    each group's frames. Every one is a rate in [0, 1], so they are
    comparable before standardization and interpretable after it."""
    df = frames.copy()
    df["specificity_index"] = df[SPECIFICITY_FLAGS].astype(float).mean(axis=1)
    df["quantified_rate"] = df["specificity_quantified_metric"].astype(float)
    df["promotional_rate"] = df["rhetoric_promotional"].astype(float)
    df["strategic_rate"] = df["rhetoric_strategic_importance"].astype(float)
    df["realized_share"] = (df["temporal"] == "realized").astype(float)
    df["hypothetical_share"] = (df["temporal"] == "hypothetical").astype(float)
    concepts = df["concepts"].apply(lambda c: list(c) if c is not None else [])
    df["risk_share"] = concepts.apply(lambda cs: any(c.startswith("risk_") for c in cs)).astype(float)
    df["gov_share"] = concepts.apply(lambda cs: any(c.startswith("gov_") for c in cs)).astype(float)
    df["firm_subject_share"] = (df["subject"] == "firm").astype(float)
    out = df.groupby(keys)[VOICE_FEATURES].mean()
    out["n_frames"] = df.groupby(keys).size()
    return out.reset_index()



def shrink_rates(rates: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """Encogimiento empírico-Bayes de cada tasa hacia la media global, en
    proporción a cuántos frames la sostienen.

    Sin esto una tasa estimada con 5 frames pesa igual que una estimada con
    500, y `cluster_diagnostics.py` midió lo que eso produce: la partición de
    4 grupos no es reproducible (Jaccard bootstrap 0,52) y
    `specificity_index` tiene confiabilidad 0,000 a nivel empresa — su varianza
    observada entre empresas es MENOR que la varianza de muestreo esperada, o
    sea que es ruido puro. Prior beta ajustado por momentos sobre las tasas
    observadas."""
    out = {}
    for column in rates.columns:
        p = rates[column]
        mean, var = float(p.mean()), float(p.var(ddof=1))
        if var <= 0 or not 0 < mean < 1:
            out[column] = p
            continue
        strength = max(mean * (1 - mean) / var - 1, 1e-6)
        alpha, beta = mean * strength, (1 - mean) * strength
        out[column] = (p * counts + alpha) / (counts + alpha + beta)
    return pd.DataFrame(out, index=rates.index)


def voice_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    """La representación que los datos SÍ sostienen: dos ejes continuos y una
    partición binaria, sobre tasas encogidas.

    Los 4 arquetipos A/B/C/D se conservan porque los documentos existentes los
    citan, pero no son reproducibles: bootstrap remuestreando los frames de
    cada empresa da Jaccard medio 0,52 con k=4 (por debajo de 0,6 = no
    reproducible) contra 0,81 con k=2. El primer factor es el eje real del
    corpus —riesgo hipotético en un extremo, despliegue afirmado y concreto en
    el otro— y explica más que cualquiera de los cortes de k=4."""
    rates = metrics[VOICE_FEATURES]
    counts = metrics["n_frames"]
    shrunk = shrink_rates(rates, counts)
    X = StandardScaler().fit_transform(shrunk.values)
    factors = FactorAnalysis(n_components=2, random_state=SEED).fit(X)
    scores = factors.transform(X)
    labels = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(X)
    # Orientación estable: el factor 1 crece hacia riesgo/hipotético, y el
    # grupo 1 es el de mayor `realized_share`. Sin fijarlo, un re-ajuste
    # invierte los signos y las etiquetas sin que cambie nada de los datos.
    if np.corrcoef(scores[:, 0], rates["risk_share"])[0, 1] < 0:
        scores[:, 0] *= -1
    if np.corrcoef(scores[:, 1], rates["quantified_rate"])[0, 1] < 0:
        scores[:, 1] *= -1
    realized_by_group = pd.Series(rates["realized_share"].values).groupby(labels).mean()
    if realized_by_group.idxmax() != 1:
        labels = 1 - labels
    out = metrics[["ticker", "n_frames"]].copy()
    out["voice_factor_risk_vs_deployment"] = scores[:, 0]
    out["voice_factor_quantified_vs_governance"] = scores[:, 1]
    out["voice_group"] = np.where(labels == 1, "deployment_asserted", "risk_hypothetical")
    for column in VOICE_FEATURES:
        out[f"shrunk_{column}"] = shrunk[column].values
    return out


def concept_shares(frames: pd.DataFrame, keys: list[str], concepts: list[str]) -> pd.DataFrame:
    """Fraction of a group's frames carrying each concept. A frame can carry
    several, so these are per-concept rates that do NOT sum to 1."""
    exploded = frames[keys + ["concepts"]].explode("concepts").dropna(subset=["concepts"])
    counts = exploded.groupby(keys + ["concepts"]).size().unstack("concepts", fill_value=0)
    totals = frames.groupby(keys).size()
    shares = counts.div(totals, axis=0)
    for concept in concepts:            # a concept absent from this corpus is 0, not missing
        if concept not in shares.columns:
            shares[concept] = 0.0
    return shares[concepts].fillna(0.0)


def domain_shares(frames: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    counts = frames.groupby(keys + ["domain"]).size().unstack("domain", fill_value=0)
    shares = counts.div(frames.groupby(keys).size(), axis=0)
    for domain in DOMAINS:
        if domain not in shares.columns:
            shares[domain] = 0.0
    return shares[DOMAINS].fillna(0.0).rename(columns={d: f"domain_{d}" for d in DOMAINS})


def _label_voice_archetypes(profiles: pd.DataFrame) -> dict[int, str]:
    """Map K-means ids to the stable letters of docs/analytics/01.

    Assigned from each cluster's own profile, claiming the most
    discriminating marker first: D by promotional rhetoric (the vocal
    leaders' defining trait — highest promotional AND high specificity at
    once), then C by quantification among what is left, then A by risk
    boilerplate, and B is the residual middle the doc calls the "average
    firm". Without this, a re-fit that renumbers the clusters would
    silently relabel every firm in the panel and every downstream doc.

    Order matters and was wrong once: claiming C by `quantified_rate`
    FIRST handed the letter to the MSFT/GOOGL/NVDA cluster, because in
    this population the vocal leaders also quantify the most. The tiny
    7-firm "concrete quantifier" group the letter originally described
    does not survive the re-fit as its own cluster, so whatever gets C now
    is the next-most-quantifying group, not that group — read the printed
    profile rather than the letter."""
    labels: dict[int, str] = {}
    remaining = list(profiles.index)
    for letter, column in (("D", "promotional_rate"), ("C", "quantified_rate"), ("A", "risk_share")):
        pick = profiles.loc[remaining, column].idxmax()
        labels[pick] = letter
        remaining.remove(pick)
    labels[remaining[0]] = "B"
    return labels


def _label_behavior_clusters(profiles: pd.DataFrame) -> dict[int, int]:
    """Same idea for the behavior clusters, whose numbering docs/analytics/06
    gives meaning to: 0 internal deployers, 2 product deployers, 3
    infrastructure investors, 1 the minimal-behavior residual.

    Each marker is claimed by the cluster that separates on it MOST, and
    the order runs from the widest-separated marker to the narrowest:
    `ai_infrastructure` (6,3% vs ~1-2%), then `domain_customer_facing`
    (44% vs 29% vs 13%), then `deployed`. Claiming by `revenue_outcome`
    first — what an earlier version did, mirroring the old cluster names —
    was unstable: clusters 0 and 2 sit at 6,5% and 6,1% on that column, so
    a last-digit difference in the means flipped which one got label 0 and
    silently renamed 225 of 446 firms between runs."""
    labels: dict[int, int] = {}
    remaining = list(profiles.index)
    for target, column in ((3, "ai_infrastructure"), (2, "domain_customer_facing"),
                           (0, "deployed")):
        pick = profiles.loc[remaining, column].idxmax()
        labels[pick] = target
        remaining.remove(pick)
    labels[remaining[0]] = 1
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DB)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.database), read_only=True)
    try:
        frames = load_frames(con)
        mentions = load_entity_mentions(con)
    finally:
        con.close()
    by_form = frames["form_type"].value_counts().to_dict()
    print(f"{len(frames):,} frames de {frames['ticker'].nunique():,} empresas | por formulario: "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(by_form.items(), key=lambda kv: -kv[1])))

    # ---- voz: 9 métricas pooled por empresa -> centroides ----
    pooled = voice_metrics(frames, ["ticker"])
    pooled = pooled.merge(frames.groupby("ticker")["cik"].first().reset_index(), on="ticker")
    population = (pooled[pooled["n_frames"] >= POOLED_MIN_FRAMES]
                  .sort_values("ticker").reset_index(drop=True))
    print(f"Población pooled (>= {POOLED_MIN_FRAMES} frames): {len(population):,} empresas")

    scaler = StandardScaler().fit(population[VOICE_FEATURES])
    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(
        scaler.transform(population[VOICE_FEATURES]))
    population["cluster_id"] = kmeans.labels_
    voice_profiles = population.groupby("cluster_id")[VOICE_FEATURES].mean()
    population["archetype"] = population["cluster_id"].map(_label_voice_archetypes(voice_profiles))

    print("\n=== Arquetipos de voz ===")
    summary = population.groupby("archetype").agg(
        n_empresas=("ticker", "size"), specificity_index=("specificity_index", "mean"),
        quantified_rate=("quantified_rate", "mean"), promotional_rate=("promotional_rate", "mean"),
        realized_share=("realized_share", "mean"), risk_share=("risk_share", "mean"),
        hypothetical_share=("hypothetical_share", "mean"), n_frames=("n_frames", "mean"),
    ).sort_index()
    print(summary.round(4).to_string())
    for archetype in summary.index:
        top = (population[population["archetype"] == archetype]
               .nlargest(8, "n_frames")["ticker"].tolist())
        print(f"  {archetype}: {', '.join(top)}")

    # ---- comportamiento: 15 features, clustering independiente ----
    behavior = concept_shares(frames, ["ticker"], BEHAVIOR_CONCEPTS)
    domains = domain_shares(frames, ["ticker"])
    firm_behavior = behavior.join(domains)
    firm_behavior["n_frames"] = frames.groupby("ticker").size()
    firm_behavior = (firm_behavior.loc[firm_behavior["n_frames"] >= POOLED_MIN_FRAMES]
                     .sort_index())

    behavior_scaler = StandardScaler().fit(firm_behavior[BEHAVIOR_FEATURES])
    behavior_km = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(
        behavior_scaler.transform(firm_behavior[BEHAVIOR_FEATURES]))
    firm_behavior["cluster_id"] = behavior_km.labels_
    behavior_profiles = firm_behavior.groupby("cluster_id")[BEHAVIOR_FEATURES].mean()
    firm_behavior["behavior_cluster"] = firm_behavior["cluster_id"].map(
        _label_behavior_clusters(behavior_profiles))
    firm_behavior = firm_behavior.drop(columns=["cluster_id"]).reset_index()

    print("\n=== Clusters de comportamiento ===")
    print(firm_behavior.groupby("behavior_cluster").agg(
        n=("ticker", "size"), deployed=("deployed", "mean"),
        revenue_outcome=("revenue_outcome", "mean"), ai_investment=("ai_investment", "mean"),
        ai_infrastructure=("ai_infrastructure", "mean"),
        customer_facing=("domain_customer_facing", "mean"),
    ).round(4).to_string())

    # ---- cruce voz x comportamiento ----
    crossed = population[["ticker", "archetype"]].merge(
        firm_behavior[["ticker", "behavior_cluster"]], on="ticker", how="inner")
    print(f"\n=== Cruce voz x comportamiento ({len(crossed):,} empresas con ambas etiquetas) ===")
    crosstab = pd.crosstab(crossed["archetype"], crossed["behavior_cluster"], normalize="index") * 100
    print(crosstab.round(1).to_string())

    # ---- panel empresa-año, proyectado sobre los centroides pooled ----
    fy = voice_metrics(frames, ["ticker", "year"])
    fy = fy.merge(frames.groupby(["ticker", "year"])["cik"].first().reset_index(),
                  on=["ticker", "year"])
    fy = (fy[fy["n_frames"] >= PANEL_MIN_FRAMES]
          .sort_values(["ticker", "year"]).reset_index(drop=True))
    distances = kmeans.transform(scaler.transform(fy[VOICE_FEATURES]))
    fy["archetype"] = pd.Series(distances.argmin(axis=1)).map(
        _label_voice_archetypes(voice_profiles)).values
    fy["archetype_dist"] = distances.min(axis=1)

    fy_behavior = concept_shares(frames, ["ticker", "year"], PANEL_BEHAVIOR_CONCEPTS)
    fy_behavior.columns = [f"behavior_share_{c}" for c in fy_behavior.columns]
    fy_domains = domain_shares(frames, ["ticker", "year"])
    fy_domains.columns = [c.replace("domain_", "domain_share_") for c in fy_domains.columns]
    mention_stats = mentions.groupby(["ticker", "year"]).agg(
        n_entity_mentions=("term", "size"),
        entities_named=("term", lambda s: ", ".join(sorted(set(s)))),
    )
    panel = (fy.set_index(["ticker", "year"])
               .join(fy_behavior).join(fy_domains).join(mention_stats).reset_index())
    panel["n_entity_mentions"] = panel["n_entity_mentions"].fillna(0).astype(int)
    panel["entities_named"] = panel["entities_named"].fillna("")
    ordered = ["ticker", "cik", "year", "n_frames", "archetype", "archetype_dist",
               *VOICE_FEATURES,
               *[f"behavior_share_{c}" for c in PANEL_BEHAVIOR_CONCEPTS],
               *[f"domain_share_{d}" for d in DOMAINS],
               "n_entity_mentions", "entities_named"]
    panel = panel[ordered]

    print(f"\n=== Panel empresa-año ===")
    print(f"{len(panel):,} filas | {panel['ticker'].nunique():,} empresas | "
          f"años {panel['year'].min()}-{panel['year'].max()}")
    print(panel.groupby(["year", "archetype"]).size().unstack(fill_value=0).to_string())
    repeated = panel.groupby("ticker").size()
    print(f"Empresas en >=2 años: {int((repeated >= 2).sum()):,} de {len(repeated):,}")

    scores = voice_scores(pooled)
    print("\n=== Voz: representación que los datos sostienen ===")
    print(scores.groupby("voice_group").agg(
        empresas=("ticker", "size"), frames_medianos=("n_frames", "median"),
        factor_riesgo=("voice_factor_risk_vs_deployment", "mean")).round(2).to_string())
    print("  (los 4 arquetipos A/B/C/D siguen saliendo por compatibilidad con los docs,")
    print("   pero su partición NO es reproducible — ver cluster_diagnostics.py)")

    paths = {
        "firm_behavior_clusters": firm_behavior,
        "voice_x_behavior": crossed,
        "firm_year_archetype_behaviors": panel,
        "firm_voice_scores": scores,
    }
    for name, table in paths.items():
        destination = args.output_dir / f"{name}.parquet"
        table.to_parquet(destination, index=False)
        print(f"-> {destination} ({len(table):,} filas)")

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/analytics/build_firm_clusters.py",
        "frames": int(len(frames)), "frames_by_form": {k: int(v) for k, v in by_form.items()},
        "pooled_firms": int(len(population)), "panel_rows": int(len(panel)),
        "pooled_min_frames": POOLED_MIN_FRAMES, "panel_min_frames": PANEL_MIN_FRAMES,
        "seed": SEED,
        "voice_archetype_sizes": summary["n_empresas"].to_dict(),
        "behavior_cluster_sizes": firm_behavior["behavior_cluster"].value_counts().sort_index().to_dict(),
    }
    manifest_path = args.output_dir / "firm_clusters_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"-> {manifest_path}")


if __name__ == "__main__":
    main()
