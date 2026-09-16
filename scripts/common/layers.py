"""
scripts/common/layers.py — the single catalog of the parquet data layers.

    raw/       downloaded filings, prices, XBRL (untouched, written by raw_ingestion)
    interim/   extraction outputs + ADDITIVE outputs (see ADDITIVE_SOURCES)
    bronze/    one cleaned table per source: typed, deduplicated to the
               current run, no analysis-universe filter
    silver/    conformed tables for gold: analysis universe applied, additive
               outputs broadcast from unique texts to paragraph instances
    gold/      spine-based data by entity grain (docs/gold_pipeline.md)
    results/   analytics outputs (tables, figures, reports), by topic

Every table is addressed by a dotted name ("bronze.paragraphs",
"silver.ai_frames") and read with `scan()` / `read()`. Nobody else hardcodes
a bronze/silver path.

Lineage uses the natural keys already carried by the additive outputs, so
any silver row can be walked back to the raw filing:

    (country_code, form, accession_number, item_key)      section  -> bronze.extraction_trace (run_id, part_file)
    + paragraph_index                                      paragraph instance
    text_hash                                              paragraph content (join key of every additive output)
    frame_id / activity_id                                 LLM frame / activity within a text

Each written table gets a `<table>._manifest.json` next to it: row count,
key uniqueness, the input files (path, size, mtime) it was built from, the
builder script and the git commit — enough to tell whether a rebuild would
change anything.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
INTERIM = DATA / "interim"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
RESULTS = DATA / "results"
MODELS = REPO_ROOT / "models"

# The US analysis universe: S&P 500 members as of 2021-01-01 (502 tickers),
# tagged in firm_universe.membership_groups.
ANALYSIS_PANEL = "sp500_2021_start_panel"

# Gold: data/gold/<kind>/<grain>/<family>.parquet (docs/gold_pipeline.md).
# Every table starts with its grain's spine columns: id, keys, date (firm has
# no date). Values are joined on the spine, one row per key. A grain's dataset
# (datasets/<grain>/<grain>) is its spine joined with every covariate and
# target family of the grain.
GOLD_KINDS = ("spines", "covariates", "targets", "datasets")
GOLD_FAMILY_KINDS = ("covariates", "targets")
GOLD_SPINE_COLUMNS: dict[str, list[str]] = {
    "document": ["id", "ticker", "accession_number", "fecha"],
    "activity": ["id", "ticker", "text_hash", "activity_id", "fecha"],
    "call": ["id", "ticker", "call_accession_number", "fecha"],
    "firm_quarter": ["id", "ticker", "quarter", "as_of_date"],
    "firm_year": ["id", "ticker", "year", "as_of_date"],
    "firm": ["id", "ticker"],
}
GOLD_GRAINS = tuple(GOLD_SPINE_COLUMNS)
GOLD_DATES = {"document": "fecha", "activity": "fecha", "call": "fecha",
              "firm_quarter": "as_of_date", "firm_year": "as_of_date", "firm": None}


def gold_keys(grain: str) -> list[str]:
    """Spine key columns of a grain (spine columns without id and date)."""
    return [c for c in GOLD_SPINE_COLUMNS[grain] if c not in ("id", GOLD_DATES[grain])]

# Outputs that cost GPU hours or LLM calls to produce. Append-only: producers
# only ever add new run/session part files. The layer builders READ them and
# must never write, move or delete anything under these paths
# (`write_table` refuses). They also live in B2; never delete them there either.
ADDITIVE_SOURCES: dict[str, dict] = {
    "embeddings": {"path": INTERIM / "embeddings", "cost": "cloud GPU",
                   "producer": "scripts/enrichment/ai_embed.py", "key": ["text_hash"]},
    "prefilter_scores": {"path": INTERIM / "prefilter_scores", "cost": "embedding compute",
                         "producer": "scripts/enrichment/ai_prefilter.py", "key": ["text_hash", "run_id"]},
    "prefilter_scores_unique": {"path": INTERIM / "prefilter_scores_unique", "cost": "embedding compute",
                                "producer": "scripts/enrichment/ai_prefilter.py", "key": ["text_hash", "run_id"]},
    "prefilter_sentence_scores": {"path": INTERIM / "prefilter_sentence_scores", "cost": "embedding compute",
                                  "producer": "scripts/enrichment/ai_prefilter_sentences.py", "key": ["text_hash"]},
    "force_recompute_sentence_scores": {"path": INTERIM / "force_recompute_sentence_scores.parquet",
                                        "cost": "embedding compute",
                                        "producer": "scripts/enrichment/ai_prefilter_sentences.py",
                                        "key": ["text_hash"]},
    "prefilter_predictions": {"path": INTERIM / "prefilter_predictions", "cost": "frozen model deployment",
                              "producer": "scripts/enrichment/ai_prefilter_deploy.py, ai_prefilter_apply_frozen.py",
                              "key": ["text_hash", "model_version"]},
    "prefilter_predictions_unique": {"path": INTERIM / "prefilter_predictions_unique",
                                     "cost": "frozen model deployment",
                                     "producer": "scripts/enrichment/ai_prefilter_deploy.py, ai_prefilter_apply_frozen.py",
                                     "key": ["text_hash", "model_version"]},
    "golden_set": {"path": INTERIM / "golden_set", "cost": "LLM judge",
                   "producer": "scripts/enrichment/golden_set.py", "key": ["text_hash", "session_id"]},
    "golden_set_forms": {"path": INTERIM / "golden_set_forms", "cost": "LLM judge",
                         "producer": "scripts/enrichment/golden_set_forms.py", "key": ["text_hash", "session_id"]},
    "ai_classify": {"path": INTERIM / "ai_classify", "cost": "LLM",
                    "producer": "scripts/enrichment/ai_classify.py", "key": ["text_hash", "frame_id"]},
    "ai_activities": {"path": INTERIM / "ai_activities", "cost": "LLM",
                      "producer": "scripts/enrichment/ai_activities_from_frames.py",
                      "key": ["text_hash", "frame_id", "activity_id"]},
    "ai_entity_mentions": {"path": INTERIM / "ai_entity_mentions", "cost": "lexical + LLM-seeded terms",
                           "producer": "scripts/enrichment/ai_entity_mentions.py", "key": ["text_hash", "term"]},
    "audits": {"path": INTERIM / "audits", "cost": "LLM", "producer": "scripts/verif", "key": []},
}

# name -> (path, partition columns). A path ending in .parquet is a single
# file; anything else is a hive-partitioned directory.
TABLES: dict[str, tuple[Path, list[str]]] = {
    # bronze
    "bronze.firm_universe": (BRONZE / "firm_universe.parquet", []),
    "bronze.filing_manifest": (BRONZE / "filing_manifest.parquet", []),
    "bronze.filing_manifest_10q": (BRONZE / "filing_manifest_10q.parquet", []),
    "bronze.extraction_trace": (BRONZE / "extraction_trace", ["form"]),
    "bronze.paragraphs": (BRONZE / "paragraphs", ["form"]),
    "bronze.sentences": (BRONZE / "sentences", ["form"]),
    "bronze.unique_paragraphs": (BRONZE / "unique_paragraphs.parquet", []),
    "bronze.market_prices": (BRONZE / "market_prices.parquet", []),
    "bronze.market_factors_daily": (BRONZE / "market_factors_daily.parquet", []),
    "bronze.market_factors_monthly": (BRONZE / "market_factors_monthly.parquet", []),
    "bronze.prefilter_scores": (BRONZE / "prefilter_scores.parquet", []),
    "bronze.prefilter_predictions": (BRONZE / "prefilter_predictions.parquet", []),
    "bronze.ai_frames": (BRONZE / "ai_frames.parquet", []),
    "bronze.ai_activities": (BRONZE / "ai_activities.parquet", []),
    "bronze.ai_entity_mentions": (BRONZE / "ai_entity_mentions.parquet", []),
    "bronze.prefilter_anchors": (BRONZE / "prefilter_anchors.parquet", []),
    "bronze.prefilter_entity_terms": (BRONZE / "prefilter_entity_terms.parquet", []),
    "bronze.patents_firm_year": (BRONZE / "patents_firm_year.parquet", []),
    "bronze.patents_preshock": (BRONZE / "patents_preshock.parquet", []),
    "bronze.xbrl_facts": (BRONZE / "xbrl_facts.parquet", []),
    "bronze.sec_filing_index": (BRONZE / "sec_filing_index.parquet", []),
    "bronze.sec_company_names": (BRONZE / "sec_company_names.parquet", []),
    "bronze.call_transcripts": (BRONZE / "call_transcripts.parquet", []),
    # silver
    "silver.firm_universe": (SILVER / "firm_universe.parquet", []),
    "silver.filing_manifest": (SILVER / "filing_manifest.parquet", []),
    "silver.filing_manifest_10q": (SILVER / "filing_manifest_10q.parquet", []),
    "silver.market_prices": (SILVER / "market_prices.parquet", []),
    "silver.ai_frames": (SILVER / "ai_frames.parquet", []),
    "silver.ai_activities": (SILVER / "ai_activities.parquet", []),
    "silver.ai_entity_mentions": (SILVER / "ai_entity_mentions.parquet", []),
    "silver.ai_vendor_mentions": (SILVER / "ai_vendor_mentions.parquet", []),
    "silver.patents_firm_year": (SILVER / "patents_firm_year.parquet", []),
    "silver.patents_preshock": (SILVER / "patents_preshock.parquet", []),
}


def path(name: str) -> Path:
    return TABLES[name][0]


def scan(name: str) -> pl.LazyFrame:
    p, partitions = TABLES[name]
    if not p.exists():
        raise FileNotFoundError(f"{name} not built: {p} (run `make bronze silver`)")
    if p.suffix == ".parquet":
        return pl.scan_parquet(p)
    return pl.scan_parquet(p, hive_partitioning=True, try_parse_hive_dates=False,
                           hive_schema={col: pl.String for col in partitions})


def read(name: str) -> pl.DataFrame:
    return scan(name).collect()


def gold_path(kind: str, grain: str, name: str) -> Path:
    """data/gold/<kind>/<grain>/<name>.parquet (parent created)."""
    if kind not in GOLD_KINDS or grain not in GOLD_GRAINS:
        raise ValueError(f"gold kind must be in {GOLD_KINDS} and grain in {GOLD_GRAINS}, got {kind}/{grain}")
    p = GOLD / kind / grain / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_gold(kind: str, grain: str, family: str, df, *, builder: str,
               inputs: list[Path] | tuple = (), extra: dict | None = None) -> Path:
    """Write data/gold/<kind>/<grain>/<family>.parquet and its manifest.

    The frame must carry the grain's spine columns; they are moved first; a
    covariate/target family drops the spine's attribute columns. A spine must
    have unique keys. A covariate/target family or a dataset must
    have exactly the spine's rows, with the same id, keys and date. Rows are
    sorted by the spine keys."""
    if kind == "datasets" and family != grain:
        raise ValueError(f"datasets/{grain}/{family}: a grain has one dataset, datasets/{grain}/{grain}")
    spine_cols, keys = GOLD_SPINE_COLUMNS[grain], gold_keys(grain)
    if kind in GOLD_FAMILY_KINDS:
        import pyarrow.parquet as pq
        # spine attributes (e.g. delisted, sic2) belong to the spine, not to a family read off it
        attributes = set(pq.read_schema(GOLD / "spines" / grain / f"{grain}.parquet").names) - set(spine_cols)
        df = df.drop(columns=[c for c in df.columns if c in attributes])
    missing = [c for c in spine_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{kind}/{grain}/{family}: missing spine columns {missing}")
    df = df[spine_cols + [c for c in df.columns if c not in spine_cols]]
    if df.duplicated(subset=keys).any():
        raise ValueError(f"{kind}/{grain}/{family}: keys {keys} are not unique")
    df = df.sort_values(keys, kind="stable").reset_index(drop=True)
    if kind != "spines":
        spine = read_gold(grain)[spine_cols].sort_values(keys, kind="stable").reset_index(drop=True)
        if len(spine) != len(df) or not spine.astype(str).equals(df[spine_cols].astype(str)):
            raise ValueError(f"{kind}/{grain}/{family}: id/keys/date differ from spines/{grain}/{grain}")
    target = gold_path(kind, grain, family)
    tmp = target.with_suffix(".parquet.partial")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)
    manifest = {
        "path": str(target.relative_to(REPO_ROOT)),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "spine": f"spines/{grain}/{grain}",
        "keys": keys,
        "builder": builder,
        "git_sha": _git_sha(),
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": input_fingerprint([p for p in inputs if Path(p).exists()]),
        **(extra or {}),
    }
    (target.parent / f"{family}._manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"-> {target.relative_to(REPO_ROOT)} ({len(df):,} rows, {df.shape[1]} columns)")
    return target


def read_gold(grain: str, *families, spine_columns: list[str] | None = None):
    """The spine of `grain` left-joined on `id` with gold families.

    Each family is (kind, family) for all its value columns or
    (kind, family, [columns]). `spine_columns` restricts the spine columns
    read (default: all, including spine attributes such as call `fe`)."""
    import pandas as pd

    out = pd.read_parquet(GOLD / "spines" / grain / f"{grain}.parquet", columns=spine_columns)
    base = set(GOLD_SPINE_COLUMNS[grain])
    for spec in families:
        kind, family, *cols = spec
        p = GOLD / kind / grain / f"{family}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} not built (run `make gold`)")
        columns = ["id"] + list(cols[0]) if cols else None
        values = pd.read_parquet(p, columns=columns)
        values = values.drop(columns=[c for c in values.columns if c in base and c != "id"])
        out = out.merge(values, on="id", how="left", validate="one_to_one")
    return out


def dataset_columns(grain: str, kind: str, family: str, columns: list[str] | None = None) -> list[str]:
    """Names in datasets/<grain>/<grain> of a family's value columns (all, or
    `columns`): the family's own name, or `<family>__<column>` where the dataset
    builder prefixed a collision with differing values."""
    import pyarrow.parquet as pq

    if columns is None:
        columns = pq.read_schema(gold_path(kind, grain, family)).names[len(GOLD_SPINE_COLUMNS[grain]):]
    manifest = GOLD / "datasets" / grain / f"{grain}._manifest.json"
    collisions = json.loads(manifest.read_text()).get("collisions", {}) if manifest.exists() else {}
    return [f"{family}__{c}" if f"{family}__{c}" in collisions.get(c, []) else c for c in columns]


def read_dataset(grain: str, *families, columns: list[str] | None = None,
                 spine_columns: list[str] | None = None):
    """Columns of datasets/<grain>/<grain> (scripts/gold/<grain>/build_dataset.py),
    the spine joined with every family of the grain.

    Same selection as `read_gold`: the spine columns (`spine_columns`, default
    all, including spine attributes), then each family (kind, family) or
    (kind, family, [columns]) under its dataset names (`dataset_columns`), then
    `columns` (dataset names, e.g. cross-grain `fq__` columns). No selection
    reads the whole dataset."""
    import pandas as pd
    import pyarrow.parquet as pq

    p = GOLD / "datasets" / grain / f"{grain}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not built (run `make gold`)")
    if not families and columns is None and spine_columns is None:
        return pd.read_parquet(p)
    if spine_columns is None:
        spine_columns = pq.read_schema(GOLD / "spines" / grain / f"{grain}.parquet").names
    selected = list(spine_columns)
    for spec in families:
        kind, family, *cols = spec
        selected += dataset_columns(grain, kind, family, list(cols[0]) if cols else None)
    selected += list(columns or [])
    return pd.read_parquet(p, columns=selected)[selected]


def firm_delistings():
    """ticker, delisted, delisting_date (datetime64, NaT when listed) of every
    universe firm (silver.firm_universe; scripts/silver/universe.py) — the
    spine attributes of the firm, firm_year, firm_quarter and call grains."""
    out = read("silver.firm_universe").select("ticker", "delisted", "delisting_date").to_pandas()
    out["delisting_date"] = out["delisting_date"].astype("datetime64[ns]")
    return out


def results_path(topic: str, name: str) -> Path:
    """data/results/<topic>/<name> — written only by scripts/analytics (parent created)."""
    p = RESULTS / topic / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _assert_not_additive(target: Path) -> None:
    target = target.resolve()
    for src in ADDITIVE_SOURCES.values():
        additive = src["path"].resolve()
        if target == additive or additive in target.parents:
            raise PermissionError(f"refusing to write inside additive source {additive}")


def input_fingerprint(paths: list[Path]) -> list[dict]:
    """(path, size, mtime) of every input file, sorted — what `_manifest.json` records."""
    out = []
    for p in sorted({Path(x) for x in paths}):
        st = p.stat()
        out.append({"path": str(p.relative_to(REPO_ROOT)), "size": st.st_size,
                    "mtime": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat()})
    return out


def write_manifest(name: str, *, rows: int, columns: list[str], keys: list[str] | None,
                   keys_unique: bool | None, inputs: list[Path], builder: str,
                   extra: dict | None = None) -> None:
    p = path(name)
    manifest = {
        "table": name,
        "path": str(p.relative_to(REPO_ROOT)),
        "rows": rows,
        "columns": columns,
        "keys": keys,
        "keys_unique": keys_unique,
        "builder": builder,
        "git_sha": _git_sha(),
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": input_fingerprint(inputs),
        **(extra or {}),
    }
    manifest_path = p.parent / f"{p.name.removesuffix('.parquet')}._manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_table(name: str, frame: pl.DataFrame | pl.LazyFrame, *, keys: list[str] | None,
                inputs: list[Path], builder: str, sort_by: list[str] | None = None,
                extra: dict | None = None) -> pl.DataFrame:
    """Write a single-file table atomically (tmp file, then rename), sorted
    deterministically by `sort_by` (default: `keys`), and record its manifest.
    Raises if `keys` are given and not unique."""
    p, partitions = TABLES[name]
    if partitions:
        raise ValueError(f"{name} is partitioned; use write_partition()")
    _assert_not_additive(p)
    df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    order = sort_by if sort_by is not None else keys
    if order:
        df = df.sort(order, nulls_last=True, maintain_order=True)
    keys_unique = None
    if keys:
        keys_unique = df.select(keys).is_duplicated().sum() == 0
        if not keys_unique:
            raise ValueError(f"{name}: keys {keys} are not unique")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".parquet.partial")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    tmp.replace(p)
    write_manifest(name, rows=df.height, columns=df.columns, keys=keys, keys_unique=keys_unique,
                   inputs=inputs, builder=builder, extra=extra)
    return df


def partition_dir(name: str, **values: str) -> Path:
    """Directory of one partition (e.g. form='10-K'), created empty. Callers
    write one or more `part-*.parquet` files into it; `finalize_partitioned`
    then records the manifest."""
    p, partitions = TABLES[name]
    _assert_not_additive(p)
    if list(values) != partitions:
        raise ValueError(f"{name} partitions are {partitions}, got {list(values)}")
    d = p
    for col in partitions:
        d = d / f"{col}={values[col]}"
    return d


def reset_partition(name: str, **values: str) -> Path:
    d = partition_dir(name, **values)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def finalize_partitioned(name: str, *, keys: list[str] | None, inputs: list[Path], builder: str,
                         extra: dict | None = None) -> None:
    lf = scan(name)
    rows = lf.select(pl.len()).collect().item()
    keys_unique = None
    if keys:
        dupes = lf.group_by(keys).len().filter(pl.col("len") > 1).select(pl.len()).collect().item()
        keys_unique = dupes == 0
        if not keys_unique:
            raise ValueError(f"{name}: keys {keys} are not unique ({dupes} duplicated keys)")
    write_manifest(name, rows=rows, columns=lf.collect_schema().names(), keys=keys,
                   keys_unique=keys_unique, inputs=inputs, builder=builder, extra=extra)
