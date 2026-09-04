"""Trains the final AI-relevance classifier (logistic regression on the
prefilter's 11 signals — see docs/prefilter_evaluation.md §8/§8.1) and
applies it to the FULL scored corpus to produce the AI-candidate set that
scripts/common/ai_classify.py's frame extraction should actually run over.

Methodology (see §8.2 in the doc for the full writeup):
1. GroupKFold(5) by accession_number over the complete golden set (9,884
   labels) — same as §8.1's refit, but this time the out-of-fold predicted
   PROBABILITIES are kept (not just the default 0.5-cutoff predictions), so
   a decision threshold can be chosen properly instead of assumed.
2. Threshold: the value (scanned in 0.01 steps) that maximizes weighted F1
   (`inclusion_weight`) on the pooled out-of-fold probabilities. This is
   the honest, unbiased performance estimate — every prediction used to
   pick both the model's weights AND the threshold came from a fold that
   never saw that row during training.
3. Only THEN: refit on all 9,884 labeled rows (standard practice — CV is
   for validating/tuning, the deployed model uses every label it has) and
   apply that final model + threshold to the corpus of UNIQUE paragraph
   texts (latest anchors run only — a different anchors_fingerprint is a
   different, incompatible score population, see ai_prefilter.py's own
   anchors_fingerprint() docstring).

DEDUP IS A PHASE, NOT A COLUMN (docs/prefilter_evaluation.md §8.8): this
script used to score and apply the model to all 3,281,038 paragraph
INSTANCES, then separately compute a `_unique_text` column on every funnel
stage for reporting -- dedup was a report-time afterthought, recomputed
per-consumer, which is exactly what caused §8.7's hash-mismatch bug. Now
`ai_prefilter.py --source-relation unique_paragraphs` (build_duckdb.py's
canonical dedup table) scores each of the 1,651,191 unique texts exactly
ONCE, and this script trains/applies/reports over THAT population as the
one and only unit of work. The funnel has one number per stage, not two;
`duplicate_count` (carried through from `unique_paragraphs`) is the only
place instance-level scale still shows up, as a single derived total.

Output: data/interim/prefilter_predictions_unique/prefilter_predictions__run=<id>.parquet
(one row per UNIQUE paragraph text, every one — not just the positives —
so a downstream reader never has to treat "absent" as "negative" by
assumption) plus a manifest JSON with the CV metrics, threshold,
coefficients (plain JSON, not a pickle — this is an 11-feature linear
model, no reason to carry a full sklearn object with its version-pinning
risk), and the funnel counts at each stage (total corpus -> lexical gate ->
this model's positives), all over unique texts.

The earlier data/interim/prefilter_predictions/ (per-instance, superseded)
is kept on disk, never deleted, per the project's data-retention rule --
it is simply no longer what ai_classify.py reads.

Usage:
    uv run python scripts/common/ai_prefilter_classify.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
# Scores over unique_paragraphs (docs/prefilter_evaluation.md §8.8), NOT the
# older per-instance run -- that population is superseded, not deleted.
LATEST_PREFILTER_RUN = "20260904T132237Z"
PREFILTER_SCORES_GLOB = (
    f"data/interim/prefilter_scores_unique/prefilter_scores__run={LATEST_PREFILTER_RUN}__part=*.parquet")
OUT_DIR = REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique"

# Deterministic override, independent of the learned model and of the
# anchors-tracked strong/weak lexical lists in configs/ai_prefilter.yaml —
# see docs/prefilter_evaluation.md §8.4. Proper-noun AI products/companies
# only, NOT generic acronyms ("ai"/"ml"/"nlp") — those are exactly the terms
# prone to boilerplate coincidence (§8.3's examples: "forward-looking
# statements", generic risk-factor lists). A paragraph naming one of these
# is essentially never noise, so it bypasses the model entirely: even if the
# model's 11-signal combination would score it below threshold (verified
# real case: a paragraph about "net losses from investments in OpenAI" that
# strong_lexical_match=True didn't save, because the model treats that flag
# as one signal among 11, not a guarantee), naming a specific product/company
# still forces inclusion. Deliberately excludes ambiguous common words even
# when they ARE real AI products elsewhere ("watson" a surname, "bard" a
# common noun, "grok"/"sora" collide with unrelated names) — a false
# positive here can't be un-included downstream, so err conservative.
NAMED_AI_ENTITIES = (
    "openai", "chatgpt", "anthropic", "claude", "gemini", "copilot",
    "midjourney", "dall-e", "dalle-2", "dalle-3", "stable diffusion",
    "watsonx", "azure openai", "vertex ai", "hugging face", "huggingface",
    "mistral ai", "perplexity ai", "character.ai", "stability ai",
    "deepmind", "meta ai", "llama 2", "llama 3",
)

SIGNAL_COLUMNS = [
    "score_ai_use", "score_ai_exploration", "score_ai_capability", "score_ai_outcome",
    "score_ai_risk", "score_ai_governance", "score_ai_strategy",
    "negative_similarity", "semantic_margin", "strong_lexical_match", "weak_lexical_match",
]
PARAGRAPH_KEY = ("country_code", "form", "accession_number", "item_key", "paragraph_index")


def _named_entity_sql(text_expr: str) -> str:
    """Same word-boundary regex technique as ai_prefilter.py's _term_regex,
    kept independent of that module's anchors-tracked strong/weak lists on
    purpose — this override must survive an anchors/lexical-config change
    without needing a full corpus rescore (see module docstring)."""
    boundary_before, boundary_after = "(^|[^a-z0-9])", "([^a-z0-9]|$)"
    alt = "|".join(term.replace("'", "''").replace(".", "\\.").replace(" ", "[ ]")
                   for term in NAMED_AI_ENTITIES)
    return f"regexp_matches({text_expr}, '{boundary_before}({alt}){boundary_after}')"


def load_golden() -> pd.DataFrame:
    """A golden-set label's OWN instance key is not necessarily
    `unique_paragraphs`' chosen representative for its text (the golden
    sampler picked its own dedup representative independently, per
    docs/golden_set_sampling.md §5 -- verified to match §8.7's finding of
    zero duplicate text_hash within the golden set, but that doesn't mean
    every golden row's key equals `unique_paragraphs`' pick for the same
    text). Since `prefilter_scores_unique` is only computed for
    `unique_paragraphs`' representative keys, the join has to go through
    `text_hash`, not the instance key, or most golden rows would silently
    find no score row at all."""
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute(f"""
            SELECT l.is_ai_disclosure, l.inclusion_weight, l.accession_number,
                   {', '.join(f'p.{c}' for c in SIGNAL_COLUMNS)},
                   {_named_entity_sql("lower(coalesce(par.paragraph_text, ''))")} AS named_entity_match
            FROM read_parquet('data/interim/golden_set/golden_set_labels__session=*__part=*.parquet',
                               union_by_name=True) l
            JOIN paragraphs par
                ON par.country_code = l.country_code AND par.form = l.form
               AND par.accession_number = l.accession_number AND par.item_key = l.item_key
               AND par.paragraph_index = l.paragraph_index
            JOIN unique_paragraphs up ON up.text_hash = par.text_hash
            JOIN read_parquet('{PREFILTER_SCORES_GLOB}') p
                ON p.country_code = up.country_code AND p.form = up.form
               AND p.accession_number = up.accession_number AND p.item_key = up.item_key
               AND p.paragraph_index = up.paragraph_index
            WHERE l.error IS NULL
        """).df()
    finally:
        con.close()


C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def cv_threshold_and_metrics(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                              groups: np.ndarray) -> tuple[float, float, dict]:
    """Honest nested CV — see docs/prefilter_evaluation.md §8.3/§8.5.
    Both `C` and the threshold are chosen using ONLY the training split of
    each outer fold (grid-searched on that fold's own weighted F1), never
    the fold the resulting prediction is scored on. `sample_weight=weights`
    in `.fit()` — not just at evaluation time — corrects for the golden
    set's deliberately non-representative stratified sampling (see
    docs/golden_set_sampling.md); omitting it (as the original §8.2 model
    did) trains against the sample's inflated positive rate instead of the
    corpus's real one. Verified (scripts/verif/prefilter_model_variants_eval.py)
    this alone + a C grid takes F1 pond. from 0.745 to 0.814 under the SAME
    nested discipline — not a leak, a genuinely better-specified model.

    Returns (chosen_c_for_deployment, chosen_threshold_for_deployment, oof_metrics).
    The deployment C/threshold come from a second, non-nested pass over ALL
    the data (standard hyperparameter selection, not a performance claim) —
    `oof_metrics` from the nested loop above is what's actually cited."""
    # §8.6: raw inclusion_weight spans ~5 orders of magnitude (0.005 to
    # 1,100.79), so fitting with it directly collapses effective sample
    # size to ~1,964/9,884 rows and lets a tiny high-weight stratum
    # dominate the fit -- verified concretely: 3 known-good, unambiguous
    # golden-set positives (all from the low-weight stage1_tech_oversample
    # stratum) dropped from predicted_proba ~0.8-0.9 under the unweighted
    # model to ~0.1-0.2 under a raw-weight fit. sqrt(inclusion_weight)
    # compresses that range enough to fix the stratification bias without
    # this collapse: better nested-CV F1 pond. (0.745 -> 0.826) AND the 3
    # sanity cases stay correctly classified (~0.65-0.82). Fitting uses the
    # sqrt-compressed weight; every metric below still uses the RAW weight,
    # since that is what makes a metric here mean "corpus-representative".
    fit_weights = np.sqrt(weights)
    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y), dtype=int)
    fold_cs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        best_c, best_t, best_f1 = C_GRID[0], 0.5, -1.0
        for c in C_GRID:
            clf = LogisticRegression(max_iter=2000, C=c)
            clf.fit(X[train_idx], y[train_idx], sample_weight=fit_weights[train_idx])
            proba_train = clf.predict_proba(X[train_idx])[:, 1]
            for t in np.arange(0.05, 0.96, 0.02):
                f1w = f1_score(y[train_idx], (proba_train >= t).astype(int),
                               sample_weight=weights[train_idx], zero_division=0)
                if f1w > best_f1:
                    best_c, best_t, best_f1 = c, float(t), f1w
        fold_cs.append(best_c)
        clf = LogisticRegression(max_iter=2000, C=best_c)
        clf.fit(X[train_idx], y[train_idx], sample_weight=fit_weights[train_idx])
        proba_test = clf.predict_proba(X[test_idx])[:, 1]
        oof_pred[test_idx] = (proba_test >= best_t).astype(int)

    metrics = {
        "f1_estrato": float(f1_score(y, oof_pred)),
        "prec_pond": float(precision_score(y, oof_pred, sample_weight=weights)),
        "recall_pond": float(recall_score(y, oof_pred, sample_weight=weights)),
        "f1_pond": float(f1_score(y, oof_pred, sample_weight=weights)),
        "fold_cs": fold_cs,
    }

    # Deployment hyperparameters: one more search, this time over ALL the
    # golden set pooled (no held-out fold) — fine for PICKING a C/threshold,
    # not for reporting performance (that's `metrics` above).
    deploy_c, deploy_t, deploy_f1 = C_GRID[0], 0.5, -1.0
    for c in C_GRID:
        clf = LogisticRegression(max_iter=2000, C=c)
        clf.fit(X, y, sample_weight=fit_weights)
        proba = clf.predict_proba(X)[:, 1]
        for t in np.arange(0.05, 0.96, 0.01):
            f1w = f1_score(y, (proba >= t).astype(int), sample_weight=weights, zero_division=0)
            if f1w > deploy_f1:
                deploy_c, deploy_t, deploy_f1 = c, float(t), f1w

    return deploy_c, deploy_t, metrics


def funnel_counts(con) -> dict:
    """One number per stage, over `unique_paragraphs` -- THE unit of work
    for this whole script (docs/prefilter_evaluation.md §8.8). No more
    parallel `_unique_text` column: since scoring itself now runs on
    `unique_paragraphs`, every stage here already IS deduplicated by
    construction, not by a report-time COUNT(DISTINCT). `*_instances`
    (via `duplicate_count`) is reported once, at the end, as the single
    place corpus scale still matters."""
    total = con.execute("SELECT COUNT(*) FROM unique_paragraphs").fetchone()[0]
    total_instances = con.execute("SELECT SUM(duplicate_count) FROM unique_paragraphs").fetchone()[0]
    lexical = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{PREFILTER_SCORES_GLOB}')
        WHERE strong_lexical_match OR weak_lexical_match
    """).fetchone()[0]
    strong_only = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{PREFILTER_SCORES_GLOB}')
        WHERE strong_lexical_match
    """).fetchone()[0]
    return {
        "total_unique_paragraphs": total, "total_paragraph_instances": int(total_instances),
        "lexical_strong_or_weak": lexical, "lexical_strong_only": strong_only,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Cargando golden set completo...")
    golden = load_golden()
    print(f"{len(golden):,} etiquetas")

    y = golden["is_ai_disclosure"].astype(int).values
    weights = golden["inclusion_weight"].astype(float).values
    fit_weights = np.sqrt(weights)
    groups = golden["accession_number"].values
    X = golden[SIGNAL_COLUMNS].astype(float).values

    print("GroupKFold(5) anidado: eligiendo C y threshold SOLO con cada fold de entrenamiento...")
    deploy_c, threshold, cv_metrics = cv_threshold_and_metrics(X, y, weights, groups)
    print(f"C/threshold para despliegue (elegidos sobre todo el set, no es la métrica reportada): "
          f"C={deploy_c}, threshold={threshold:.2f}")
    print(f"CV anidado (modelo solo, la métrica real): F1 estrato={cv_metrics['f1_estrato']:.3f} "
          f"prec pond.={cv_metrics['prec_pond']:.3f} recall pond.={cv_metrics['recall_pond']:.3f} "
          f"F1 pond.={cv_metrics['f1_pond']:.3f} | C por fold: {cv_metrics['fold_cs']}")

    named_entity = golden["named_entity_match"].astype(bool).values
    print(f"\nnamed_entity_match en golden set: {int(named_entity.sum())} filas "
          f"({int((named_entity & (y == 1)).sum())} positivos reales, "
          f"{int((named_entity & (y == 0)).sum())} negativos reales)")
    # Mismo procedimiento anidado que arriba (C y threshold elegidos solo en
    # el fold de entrenamiento), pero OR-combinando con named_entity_match
    # en el fold de test — para ver si el override ayuda al modelo YA
    # mejorado, no al modelo viejo sin sample_weight.
    gkf_check = GroupKFold(n_splits=5)
    combined_oof = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in gkf_check.split(X, y, groups):
        best_c, best_t, best_f1 = C_GRID[0], 0.5, -1.0
        for c in C_GRID:
            clf = LogisticRegression(max_iter=2000, C=c)
            clf.fit(X[train_idx], y[train_idx], sample_weight=fit_weights[train_idx])
            proba_train = clf.predict_proba(X[train_idx])[:, 1]
            for t in np.arange(0.05, 0.96, 0.02):
                f1w = f1_score(y[train_idx], (proba_train >= t).astype(int),
                               sample_weight=weights[train_idx], zero_division=0)
                if f1w > best_f1:
                    best_c, best_t, best_f1 = c, float(t), f1w
        clf = LogisticRegression(max_iter=2000, C=best_c)
        clf.fit(X[train_idx], y[train_idx], sample_weight=fit_weights[train_idx])
        proba_test = clf.predict_proba(X[test_idx])[:, 1]
        combined_oof[test_idx] = ((proba_test >= best_t) | named_entity[test_idx]).astype(int)
    combined_metrics = {
        "f1_estrato": float(f1_score(y, combined_oof)),
        "prec_pond": float(precision_score(y, combined_oof, sample_weight=weights)),
        "recall_pond": float(recall_score(y, combined_oof, sample_weight=weights)),
        "f1_pond": float(f1_score(y, combined_oof, sample_weight=weights)),
    }
    print(f"CV anidado (modelo OR named_entity): F1 estrato={combined_metrics['f1_estrato']:.3f} "
          f"prec pond.={combined_metrics['prec_pond']:.3f} recall pond.={combined_metrics['recall_pond']:.3f} "
          f"F1 pond.={combined_metrics['f1_pond']:.3f}")
    use_named_entity = combined_metrics["f1_pond"] > cv_metrics["f1_pond"]
    print(f"named_entity_match {'SÍ' if use_named_entity else 'NO'} mejora la métrica -> "
          f"{'se incluye' if use_named_entity else 'se descarta'} en el despliegue.")

    print("\nReajustando el modelo final sobre TODO el golden set...")
    final_model = LogisticRegression(max_iter=2000, C=deploy_c)
    final_model.fit(X, y, sample_weight=fit_weights)

    con = duckdb.connect(str(DB), read_only=True)
    print("Cargando el corpus de textos únicos (última corrida de anchors)...")
    corpus = con.execute(f"""
        SELECT p.{', p.'.join(PARAGRAPH_KEY)}, p.text_hash, up.duplicate_count,
               {', '.join(f'p.{c}' for c in SIGNAL_COLUMNS)},
               up.paragraph_text,
               {_named_entity_sql("lower(coalesce(up.paragraph_text, ''))")} AS named_entity_match
        FROM read_parquet('{PREFILTER_SCORES_GLOB}') p
        JOIN unique_paragraphs up USING ({', '.join(PARAGRAPH_KEY)})
    """).df()
    print(f"{len(corpus):,} textos únicos ({int(corpus['duplicate_count'].sum()):,} instancias en el corpus)")

    print("Aplicando el modelo final al corpus de textos únicos...")
    Xc = corpus[SIGNAL_COLUMNS].astype(float).values
    proba = final_model.predict_proba(Xc)[:, 1]
    named_entity_corpus = corpus["named_entity_match"].astype(bool).values
    model_positive = proba >= threshold
    is_positive = (model_positive | named_entity_corpus) if use_named_entity else model_positive
    rescued = int((named_entity_corpus & ~model_positive).sum()) if use_named_entity else 0
    instances_positive = int(corpus.loc[is_positive, "duplicate_count"].sum())
    print(f"Marcados como IA-relevantes: {int(is_positive.sum()):,} textos únicos / {len(corpus):,} "
          f"({100 * is_positive.mean():.2f}%), representando {instances_positive:,} instancias del corpus"
          + (f" — de los cuales {rescued:,} textos solo por named_entity_match" if use_named_entity else ""))

    print("Calculando el funnel...")
    funnel = funnel_counts(con)
    funnel["prefilter_model_only_positive"] = int(model_positive.sum())
    funnel["named_entity_rescued"] = rescued
    funnel["prefilter_model_positive"] = int(is_positive.sum())
    funnel["prefilter_model_positive_instances"] = instances_positive
    con.close()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = corpus[[*PARAGRAPH_KEY, "text_hash", "duplicate_count"]].copy()
    out["predicted_proba"] = proba.astype("float32")
    out["named_entity_match"] = named_entity_corpus
    out["is_ai_prefiltered"] = is_positive
    out["threshold"] = threshold
    out["model_version"] = run_id
    out["anchors_run"] = LATEST_PREFILTER_RUN

    out_path = OUT_DIR / f"prefilter_predictions__run={run_id}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="zstd")

    manifest = {
        "run_id": run_id, "anchors_run": LATEST_PREFILTER_RUN,
        "golden_set_labels": int(len(golden)),
        "deploy_c": deploy_c, "threshold": threshold,
        "cv_metrics": cv_metrics, "cv_metrics_with_named_entity": combined_metrics,
        "named_entity_used_in_deployment": use_named_entity,
        "named_ai_entities": list(NAMED_AI_ENTITIES),
        "coefficients": dict(zip(SIGNAL_COLUMNS, final_model.coef_[0].tolist())),
        "intercept": float(final_model.intercept_[0]),
        "funnel": funnel,
        "output": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = OUT_DIR / f"prefilter_predictions_manifest__run={run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nParquet -> {out_path}")
    print(f"Manifiesto -> {manifest_path}")
    print(f"\nFunnel:")
    for stage, count in funnel.items():
        print(f"  {stage}: {count:,}")


if __name__ == "__main__":
    main()
