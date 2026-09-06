"""Trains the final AI-MENTION classifier (logistic regression on the
prefilter's 12 signals — see docs/prefilter_evaluation.md §8/§8.1) and
applies it to the FULL scored corpus to produce the AI-candidate set that
scripts/common/ai_classify.py's frame extraction should actually run over.

Target is "does this paragraph mention AI at all" (`relevance != 'none'`
in the golden set), NOT "is this a substantive AI disclosure" (see §8.11).
This project's focus is AI-WASHING -- whether a mention is concrete or
vague/promotional IS the signal to study downstream, so filtering out
non-concrete mentions at this stage would throw away exactly the cases
that matter. Judging "substantive vs. vague" belongs to a later stage,
never to this prefilter.

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
PREFILTER_SCORES_GLOB = (
    "data/interim/prefilter_scores_unique/prefilter_scores__run=*__part=*.parquet")
# El juez por defecto es el mismo que etiqueta el golden set: si cambia allá,
# cambia acá, y no queda un string duplicado que se desincronice.
DEFAULT_JUDGE_MODEL = "qwen/qwen3.7-flash"

OUT_DIR = REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique"


def current_scores_run(con) -> str:
    """Run id de la corrida de scores vigente.

    `main()` referenciaba una constante `LATEST_PREFILTER_RUN` que dejó de
    existir cuando `scores_relation()` reemplazó el run fijo por "la config de
    anchors más nueva" (el pin congelaba el corpus). El camino de --apply-only
    se usó todos estos días y el de entrenamiento no, así que el NameError
    quedó latente hasta el primer refit. Se resuelve preguntándole a los
    propios scores cuál es la corrida vigente, que es la misma regla que usa
    `scores_relation()` para elegir la población."""
    return con.execute(
        f"SELECT max(run_id) FROM {scores_relation()}").fetchone()[0]


def scores_relation() -> str:
    """Every score part written under the CURRENT (model, anchors, dtype),
    across however many runs produced them — not one hardcoded run id.

    ai_prefilter.py is additive: it scores only the texts that don't have a
    row yet and writes them under a NEW run id, so one score population is
    normally spread over several runs (DEF 14A + 8-K arriving after 10-K/
    10-Q is exactly that case). Pinning a single run id silently froze the
    corpus at whatever had been scored that day.

    Unioning the whole directory blindly is the opposite mistake, and the
    reason the pin existed: score parts are append-only and never deleted,
    so after retuning the anchors the directory holds several COMPLETE
    populations of the same texts at incompatible score scales. The newest
    run's configuration wins and older ones are ignored — the same rule,
    and the same SQL, as build_duckdb.py's `ai_prefilter_scores` view."""
    return f"""(
        WITH all_scores AS (
            SELECT * FROM read_parquet('{PREFILTER_SCORES_GLOB}', union_by_name=True)
        ), current AS (
            SELECT model, anchors_fingerprint, dtype
            FROM all_scores ORDER BY run_id DESC LIMIT 1
        )
        SELECT s.* FROM all_scores s JOIN current c
          ON s.model = c.model
         AND s.anchors_fingerprint = c.anchors_fingerprint
         AND s.dtype = c.dtype
    )"""

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
    # No estadounidenses agregados 2026-09-04 (mismo criterio que
    # configs/ai_prefilter.yaml: marca de producto, no conglomerado matriz).
    "deepseek", "qwen", "ernie bot", "chatglm", "hunyuan", "kimi",
    "moonshot ai", "aleph alpha", "ai21 labs", "doubao", "hyperclova",
)

SIGNAL_COLUMNS = [
    "score_ai_use", "score_ai_exploration", "score_ai_capability", "score_ai_outcome",
    "score_ai_risk", "score_ai_governance", "score_ai_strategy",
    "negative_similarity", "semantic_margin",
]
# Escalones, no un booleano plano ni un conteo lineal (docs/prefilter_evaluation.md
# §8.11): `strong_lexical_match`/`weak_lexical_match` colapsaban "0 términos" vs
# "1+" a un solo bit, perdiendo que 2+ y 3+ términos matcheados son señal cada vez
# más fuerte (medido en el golden set: 0 términos -> 0,5% positivo, 1 -> 59%, 2 ->
# 72%, 3+ -> 89-100%). Un conteo LINEAL tampoco sirve -- probado y peor (F1 pond.
# 0,745 vs 0,829 del booleano) porque la relación real no es lineal, satura rápido.
# Dummies escalonadas SÍ dejan que la logística aprenda un salto distinto en cada
# nivel: F1 pond. 0,829 -> 0,858 bajo el mismo CV anidado (más granularidad, ej.
# strong>=4, no mejora más -- 3 escalones ya capturan casi toda la ganancia).
LEXICAL_STEP_COLUMNS = [
    ("strong_ge1", "len(p.strong_matched_terms) >= 1"),
    ("strong_ge2", "len(p.strong_matched_terms) >= 2"),
    ("weak_ge1", "len(p.weak_matched_terms) >= 1"),
]
ALL_SIGNAL_COLUMNS = SIGNAL_COLUMNS + [name for name, _ in LEXICAL_STEP_COLUMNS]
PARAGRAPH_KEY = ("country_code", "form", "accession_number", "item_key", "paragraph_index")


def _named_entity_sql(text_expr: str, entities=None) -> str:
    """Same word-boundary regex technique as ai_prefilter.py's _term_regex,
    kept independent of that module's anchors-tracked strong/weak lists on
    purpose — this override must survive an anchors/lexical-config change
    without needing a full corpus rescore (see module docstring)."""
    boundary_before, boundary_after = "(^|[^a-z0-9])", "([^a-z0-9]|$)"
    alt = "|".join(term.replace("'", "''").replace(".", "\\.").replace(" ", "[ ]")
                   for term in (NAMED_AI_ENTITIES if entities is None else entities))
    return f"regexp_matches({text_expr}, '{boundary_before}({alt}){boundary_after}')"


def load_golden(judge_model: str | None = None) -> pd.DataFrame:
    """A golden-set label's OWN instance key is not necessarily
    `unique_paragraphs`' chosen representative for its text (the golden
    sampler picked its own dedup representative independently, per
    docs/golden_set_sampling.md §5 -- verified to match §8.7's finding of
    zero duplicate text_hash within the golden set, but that doesn't mean
    every golden row's key equals `unique_paragraphs`' pick for the same
    text). Since `prefilter_scores_unique` is only computed for
    `unique_paragraphs`' representative keys, the join has to go through
    `text_hash`, not the instance key, or most golden rows would silently
    find no score row at all.

    Target is `relevance != 'none'` (`is_ai_mention`), NOT `is_ai_disclosure`
    (docs/prefilter_evaluation.md §8.11). `is_ai_disclosure` was defined by
    golden_set.py's judge prompt as true IFF relevance is "substantive" --
    an "incidental" AI mention (786 rows in the golden set, real, non-noise
    mentions of AI that just don't assert anything concrete) was always
    trained as a NEGATIVE. That is backwards for this project's actual
    question: whether a mention is concrete or vague IS the AI-washing
    signal to study downstream, not something to filter out here. This
    stage's job is "does this paragraph mention AI at all", not "is this
    substantive" -- that judgment belongs to a later stage, not the
    prefilter.

    `judge_model` restringe las etiquetas a UN juez. No es higiene opcional: el
    golden set fue etiquetado por dos modelos y no coinciden. gemini-3.8-flash
    etiquetó stage1 y parte de stage2; qwen3.7-flash el resto de stage2 y TODO
    stage3_random — el estrato que ancla la reponderación a la prevalencia del
    corpus. Dentro del MISMO estrato y el mismo keyword tier llaman IA a cosas
    distintas: en el tier léxico fuerte gemini dice "menciona IA" el 100,0% de
    las veces y qwen el 81,8% (tier débil: 13,3% vs 7,4%). Mezclarlos convierte
    el target en una mezcla de dos reglas de decisión correlacionada con la
    etapa de muestreo, y tanto el threshold como toda cifra ponderada heredan
    esa mezcla. Las filas del otro juez se conservan en disco (sirven para medir
    acuerdo); lo que no puede seguir es entrenar con las dos a la vez."""
    con = duckdb.connect(str(DB), read_only=True)
    try:
        query = f"""
            SELECT l.is_ai_disclosure, l.relevance, l.relevance != 'none' AS is_ai_mention,
                   l.inclusion_weight, l.accession_number,
                   {', '.join(f'p.{c}' for c in SIGNAL_COLUMNS)},
                   {', '.join(f'{sql} AS {name}' for name, sql in LEXICAL_STEP_COLUMNS)},
                   {_named_entity_sql("lower(coalesce(par.paragraph_text, ''))")} AS named_entity_match
            FROM read_parquet('data/interim/golden_set/golden_set_labels__session=*__part=*.parquet',
                               union_by_name=True) l
            JOIN paragraphs par
                ON par.country_code = l.country_code AND par.form = l.form
               AND par.accession_number = l.accession_number AND par.item_key = l.item_key
               AND par.paragraph_index = l.paragraph_index
            JOIN unique_paragraphs up ON up.text_hash = par.text_hash
            JOIN {scores_relation()} p
                ON p.country_code = up.country_code AND p.form = up.form
               AND p.accession_number = up.accession_number AND p.item_key = up.item_key
               AND p.paragraph_index = up.paragraph_index
            WHERE l.error IS NULL AND up.is_scorable
              JUDGE_CLAUSE
        """
        return con.execute(query.replace(
            "JUDGE_CLAUSE",
            f"AND l.judge_model = '{judge_model}'" if judge_model else "")).df()
    finally:
        con.close()


def load_golden_judges(judge_model: str | None = None) -> pd.Series:
    """Qué juez etiquetó cada fila que entra al fit. Va al manifiesto: un
    despliegue tiene que poder decir con qué criterio se definió su target."""
    con = duckdb.connect(str(DB), read_only=True)
    try:
        clause = f"AND l.judge_model = '{judge_model}'" if judge_model else ""
        return con.execute(f"""
            SELECT l.judge_model
            FROM read_parquet('data/interim/golden_set/golden_set_labels__session=*__part=*.parquet',
                               union_by_name=True) l
            JOIN paragraphs par
                ON par.country_code = l.country_code AND par.form = l.form
               AND par.accession_number = l.accession_number AND par.item_key = l.item_key
               AND par.paragraph_index = l.paragraph_index
            JOIN unique_paragraphs up ON up.text_hash = par.text_hash
            JOIN {scores_relation()} p
                ON p.country_code = up.country_code AND p.form = up.form
               AND p.accession_number = up.accession_number AND p.item_key = up.item_key
               AND p.paragraph_index = up.paragraph_index
            WHERE l.error IS NULL AND up.is_scorable {clause}
        """).df()["judge_model"]
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
    place corpus scale still matters.

    Scoped to `is_scorable` (junk paragraphs were never real candidates,
    see §8.9) AND to texts actually present in `PREFILTER_SCORES_GLOB` --
    `unique_paragraphs` itself now also holds Chile (docs/
    prefilter_evaluation.md §8.9: wired into the schema, deliberately NOT
    yet scored/embedded), and counting the full table here would silently
    imply Chile was part of this funnel when nothing downstream has
    touched it.

    Joined by `text_hash`, NOT the instance key (also §8.9): a handful of
    universally-generic short strings ("", a bullet, "100%", bare numbers)
    are byte-identical between US and CL text, so once CL entered
    `unique_paragraphs` some of those groups' representative KEY flipped
    to a Chilean instance ('cl' sorts before 'us' as a tie-break) even
    though the group is 99.99% US paragraphs. Joining on the instance key
    against the (pre-CL-merge) US-only score file would silently drop
    ~214,710 US paragraph instances that share those hashes -- joining on
    `text_hash`, the actual content identity, doesn't care which specific
    instance a later rebuild happens to pick as representative."""
    total, total_instances = con.execute(f"""
        SELECT count(*), COALESCE(sum(up.duplicate_count), 0)
        FROM unique_paragraphs up
        JOIN {scores_relation()} p ON p.text_hash = up.text_hash
        WHERE up.is_scorable
    """).fetchone()
    lexical = con.execute(f"""
        SELECT COUNT(*) FROM {scores_relation()} p
        JOIN unique_paragraphs up ON up.text_hash = p.text_hash
        WHERE up.is_scorable AND (strong_lexical_match OR weak_lexical_match)
    """).fetchone()[0]
    strong_only = con.execute(f"""
        SELECT COUNT(*) FROM {scores_relation()} p
        JOIN unique_paragraphs up ON up.text_hash = p.text_hash
        WHERE up.is_scorable AND strong_lexical_match
    """).fetchone()[0]
    return {
        "total_unique_paragraphs": total, "total_paragraph_instances": int(total_instances),
        "lexical_strong_or_weak": lexical, "lexical_strong_only": strong_only,
    }


def latest_manifest(out_dir: Path = OUT_DIR) -> Path:
    """Newest TRAINED model's manifest (filenames carry a UTC timestamp, so
    lexical order is chronological).

    Manifests written by --apply-only are skipped: they are copies of a
    model, not a model. Taking the newest file blindly means the second
    apply cites the first apply as its origin, the third cites the second,
    and `trained_run` decays into a chain of re-applications with the real
    training run buried at the end — same numbers, useless provenance."""
    manifests = [m for m in sorted(out_dir.glob("prefilter_predictions_manifest__run=*.json"))
                 if not json.loads(m.read_text()).get("applied_only")]
    if not manifests:
        raise FileNotFoundError(
            f"no trained model to apply in {out_dir} — run this script without "
            f"--apply-only once to train one")
    return manifests[-1]


def load_deployed_model(manifest_path: Path) -> dict:
    """The coefficients/intercept/threshold of an ALREADY-TRAINED run.

    Applying a stored model instead of refitting is not a shortcut, it is
    the point: new documents (DEF 14A, 8-K) joining the corpus must be
    judged by the SAME decision rule the earlier corpus was, or the funnel
    counts and every analysis built on them stop being comparable across
    the forms. Refitting would also move the threshold, silently
    reclassifying 10-K/10-Q paragraphs that nothing about them changed.

    The manifest stores a plain JSON linear model on purpose (see the
    module docstring), so this reconstructs `proba` arithmetically rather
    than unpickling a sklearn object: coefficients are keyed BY NAME and
    reordered to ALL_SIGNAL_COLUMNS here, so a future reordering of that
    list can't silently pair a coefficient with the wrong signal."""
    manifest = json.loads(manifest_path.read_text())
    coefficients = manifest["coefficients"]
    missing = [c for c in ALL_SIGNAL_COLUMNS if c not in coefficients]
    extra = [c for c in coefficients if c not in ALL_SIGNAL_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"{manifest_path.name} was trained on a different signal set than this script "
            f"builds (missing={missing}, unexpected={extra}); it cannot be applied as-is")
    return {
        "run_id": manifest["run_id"],
        "coef": np.array([coefficients[c] for c in ALL_SIGNAL_COLUMNS], dtype=float),
        "intercept": float(manifest["intercept"]),
        "threshold": float(manifest["threshold"]),
        "use_named_entity": bool(manifest["named_entity_used_in_deployment"]),
        "named_ai_entities": tuple(manifest["named_ai_entities"]),
        "deploy_c": manifest.get("deploy_c"),
        "cv_metrics": manifest.get("cv_metrics"),
        "cv_metrics_with_named_entity": manifest.get("cv_metrics_with_named_entity"),
        "golden_set_labels": manifest.get("golden_set_labels"),
        "manifest_path": str(manifest_path),
    }


def apply_only(manifest_path: Path) -> None:
    """Score the current corpus with a stored model. No golden set is read
    and nothing is fitted."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deployed = load_deployed_model(manifest_path)
    print(f"Modelo desplegado: {manifest_path.name} (entrenado en run {deployed['run_id']})")
    print(f"  threshold={deployed['threshold']:.2f} C={deployed['deploy_c']} "
          f"named_entity={deployed['use_named_entity']} "
          f"({len(deployed['named_ai_entities'])} entidades)")
    if tuple(deployed["named_ai_entities"]) != NAMED_AI_ENTITIES:
        print("  AVISO: NAMED_AI_ENTITIES cambió desde ese entrenamiento; se usa la lista "
              "del manifiesto para no alterar la regla de decisión desplegada.")

    con = duckdb.connect(str(DB), read_only=True)
    print("Cargando el corpus de textos únicos (misma config de anchors)...")
    corpus = con.execute(f"""
        SELECT p.{', p.'.join(PARAGRAPH_KEY)}, p.text_hash, up.duplicate_count,
               {', '.join(f'p.{c}' for c in SIGNAL_COLUMNS)},
               {', '.join(f'{sql} AS {name}' for name, sql in LEXICAL_STEP_COLUMNS)},
               {_named_entity_sql("lower(coalesce(up.paragraph_text, ''))",
                                  deployed["named_ai_entities"])} AS named_entity_match
        FROM {scores_relation()} p
        JOIN unique_paragraphs up ON up.text_hash = p.text_hash
        WHERE up.is_scorable
    """).df()
    print(f"{len(corpus):,} textos únicos ({int(corpus['duplicate_count'].sum()):,} instancias)")

    Xc = corpus[ALL_SIGNAL_COLUMNS].astype(float).values
    logit = Xc @ deployed["coef"] + deployed["intercept"]
    proba = 1.0 / (1.0 + np.exp(-logit))
    named_entity_corpus = corpus["named_entity_match"].astype(bool).values
    model_positive = proba >= deployed["threshold"]
    is_positive = ((model_positive | named_entity_corpus)
                   if deployed["use_named_entity"] else model_positive)
    rescued = int((named_entity_corpus & ~model_positive).sum()) if deployed["use_named_entity"] else 0
    instances_positive = int(corpus.loc[is_positive, "duplicate_count"].sum())
    print(f"Marcados como IA-relevantes: {int(is_positive.sum()):,} textos únicos / {len(corpus):,} "
          f"({100 * is_positive.mean():.2f}%), representando {instances_positive:,} instancias"
          + (f" — {rescued:,} solo por named_entity_match" if deployed["use_named_entity"] else ""))

    print("Calculando el funnel...")
    funnel = funnel_counts(con)
    funnel["prefilter_model_only_positive"] = int(model_positive.sum())
    funnel["named_entity_rescued"] = rescued
    funnel["prefilter_model_positive"] = int(is_positive.sum())
    funnel["prefilter_model_positive_instances"] = instances_positive
    # Per-form funnel: the whole reason to re-apply is that new document
    # types entered the corpus, so "how many candidates did each form
    # contribute" is the number a reader will ask for first.
    by_form = con.execute(f"""
        SELECT up.country_code, up.form, count(*) AS unique_texts,
               sum(up.duplicate_count) AS instances
        FROM {scores_relation()} p
        JOIN unique_paragraphs up ON up.text_hash = p.text_hash
        WHERE up.is_scorable GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    con.close()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = corpus[[*PARAGRAPH_KEY, "text_hash", "duplicate_count"]].copy()
    out["predicted_proba"] = proba.astype("float32")
    out["named_entity_match"] = named_entity_corpus
    out["is_ai_prefiltered"] = is_positive
    out["threshold"] = deployed["threshold"]
    # `model_version` stays THIS run's id, because that is what downstream
    # orders by: ai_classify.py resolves a text present in several
    # prediction files with `QUALIFY ... ORDER BY model_version DESC`, so
    # stamping the training run's id here would make a fresh apply TIE with
    # the deployment it supersedes instead of winning. The model's own
    # identity is not lost — it rides along in `trained_run`.
    out["model_version"] = run_id
    out["trained_run"] = deployed["run_id"]
    out_path = OUT_DIR / f"prefilter_predictions__run={run_id}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="zstd")

    manifest = {
        "run_id": run_id, "applied_only": True,
        "trained_run": deployed["run_id"], "trained_manifest": deployed["manifest_path"],
        "golden_set_labels": deployed["golden_set_labels"],
        "deploy_c": deployed["deploy_c"], "threshold": deployed["threshold"],
        "cv_metrics": deployed["cv_metrics"],
        "cv_metrics_with_named_entity": deployed["cv_metrics_with_named_entity"],
        "named_entity_used_in_deployment": deployed["use_named_entity"],
        "named_ai_entities": list(deployed["named_ai_entities"]),
        "coefficients": dict(zip(ALL_SIGNAL_COLUMNS, deployed["coef"].tolist())),
        "intercept": deployed["intercept"],
        "funnel": funnel,
        "funnel_by_form": by_form.to_dict("records"),
        "output": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = OUT_DIR / f"prefilter_predictions_manifest__run={run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nParquet -> {out_path}")
    print(f"Manifiesto -> {manifest_path}")
    print("\nFunnel:")
    for stage, count in funnel.items():
        print(f"  {stage}: {count:,}")
    print("\nPor formulario:")
    for row in by_form.to_dict("records"):
        print(f"  {row['country_code']}/{row['form']}: {int(row['unique_texts']):,} textos únicos, "
              f"{int(row['instances']):,} instancias")


def main(judge_model: str | None = DEFAULT_JUDGE_MODEL) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cargando golden set (juez: {judge_model or 'TODOS — mezcla, ver load_golden'})...")
    golden = load_golden(judge_model)
    golden_judges = load_golden_judges(judge_model)
    print(f"{len(golden):,} etiquetas | jueces: {golden_judges.value_counts().to_dict()}")

    y = golden["is_ai_mention"].astype(int).values
    weights = golden["inclusion_weight"].astype(float).values
    fit_weights = np.sqrt(weights)
    groups = golden["accession_number"].values
    X = golden[ALL_SIGNAL_COLUMNS].astype(float).values

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
    # FORZADO A True (2026-09-04, §8.13) -- no por el criterio automático de
    # arriba, que rechaza el override por una diferencia de 0.001 en F1 pond.
    # (0.946 vs 0.947) sobre solo 76 filas del golden set con named_entity_match.
    # Esa muestra es demasiado chica y no representativa para este caso: el
    # patrón real que el override rescata ("net losses from investments in
    # OpenAI" dentro de un párrafo financiero genérico, proba 0,60-0,69 bajo
    # el threshold 0,75 nuevo) casi no aparece en el golden set, pero SÍ
    # aparece en el corpus real -- verificado: 17 de 59 textos únicos con
    # "openai" en todo el corpus quedaban excluidos del prefiltro sin este
    # override. El costo medido (10 falsos positivos ponderados en 76 filas)
    # es real pero mucho menor que el beneficio cualitativo de nunca perder
    # una mención nombrada explícita -- exactamente la premisa original de
    # NAMED_AI_ENTITIES ("una mención de estos es esencialmente nunca ruido").
    use_named_entity = True
    print(f"named_entity_match: forzado a incluir en el despliegue (ver comentario en código, "
          f"§8.13) pese a que el criterio automático (F1 pond. {combined_metrics['f1_pond']:.3f} "
          f"vs {cv_metrics['f1_pond']:.3f}) lo hubiera descartado.")

    print("\nReajustando el modelo final sobre TODO el golden set...")
    final_model = LogisticRegression(max_iter=2000, C=deploy_c)
    final_model.fit(X, y, sample_weight=fit_weights)

    con = duckdb.connect(str(DB), read_only=True)
    print("Cargando el corpus de textos únicos (última corrida de anchors)...")
    # `is_scorable` (build_duckdb.py) descarta párrafos sin contenido real
    # (<=3 caracteres útiles, o sin ningún alfanumérico) -- verificado en
    # producción (docs/prefilter_evaluation.md §8.9) que sin este filtro el
    # modelo puede marcar basura como positiva: el párrafo literal "AI" (2
    # caracteres, is_scorable=false) salía con predicted_proba=0.86 solo por
    # matchear el término léxico fuerte, sin nada de contenido detrás.
    #
    # JOIN por text_hash, no por llave de instancia (docs/
    # prefilter_evaluation.md §8.9): un puñado de strings basura ("", un
    # bullet, "100%", números sueltos) son idénticos entre US y CL, así que
    # al entrar CL a `unique_paragraphs` el representante de esos grupos
    # saltó a una instancia chilena ('cl' ordena antes que 'us') aunque el
    # grupo sea ~100% párrafos de EE.UU. Uniendo por llave de instancia
    # contra la corrida de scores (pre-fusión con CL, solo US) esos grupos
    # desaparecían del corpus enteros -- ~214.710 instancias de EE.UU.
    # perdidas silenciosamente. El text_hash es la identidad real del
    # contenido y no cambia aunque el representante elegido sí lo haga.
    corpus = con.execute(f"""
        SELECT p.{', p.'.join(PARAGRAPH_KEY)}, p.text_hash, up.duplicate_count,
               {', '.join(f'p.{c}' for c in SIGNAL_COLUMNS)},
               {', '.join(f'{sql} AS {name}' for name, sql in LEXICAL_STEP_COLUMNS)},
               up.paragraph_text,
               {_named_entity_sql("lower(coalesce(up.paragraph_text, ''))")} AS named_entity_match
        FROM {scores_relation()} p
        JOIN unique_paragraphs up ON up.text_hash = p.text_hash
        WHERE up.is_scorable
    """).df()
    print(f"{len(corpus):,} textos únicos ({int(corpus['duplicate_count'].sum()):,} instancias en el corpus)")

    print("Aplicando el modelo final al corpus de textos únicos...")
    Xc = corpus[ALL_SIGNAL_COLUMNS].astype(float).values
    proba = final_model.predict_proba(Xc)[:, 1]
    named_entity_corpus = corpus["named_entity_match"].astype(bool).values
    model_positive = proba >= threshold
    is_positive = (model_positive | named_entity_corpus) if use_named_entity else model_positive
    rescued = int((named_entity_corpus & ~model_positive).sum()) if use_named_entity else 0
    instances_positive = int(corpus.loc[is_positive, "duplicate_count"].sum())
    print(f"Marcados como IA-relevantes: {int(is_positive.sum()):,} textos únicos / {len(corpus):,} "
          f"({100 * is_positive.mean():.2f}%), representando {instances_positive:,} instancias del corpus"
          + (f" — de los cuales {rescued:,} textos solo por named_entity_match" if use_named_entity else ""))

    anchors_run = current_scores_run(con)
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
    out["anchors_run"] = anchors_run

    out_path = OUT_DIR / f"prefilter_predictions__run={run_id}.parquet"
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="zstd")

    manifest = {
        "run_id": run_id, "anchors_run": anchors_run,
        "golden_set_labels": int(len(golden)),
        # Provenance del target: sin esto el manifiesto no dice con qué criterio
        # se decidió "esto menciona IA", y el golden set tiene dos jueces.
        "judge_model": judge_model or "MEZCLA (todos los jueces del golden set)",
        "labels_by_judge": {str(k): int(v) for k, v in
                            golden_judges.value_counts().items()},
        "deploy_c": deploy_c, "threshold": threshold,
        "cv_metrics": cv_metrics, "cv_metrics_with_named_entity": combined_metrics,
        "named_entity_used_in_deployment": use_named_entity,
        "named_ai_entities": list(NAMED_AI_ENTITIES),
        "coefficients": dict(zip(ALL_SIGNAL_COLUMNS, final_model.coef_[0].tolist())),
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-only", action="store_true",
                        help="Apply an already-trained model to the current corpus instead of "
                             "refitting. Use when new documents joined the corpus and the "
                             "decision rule must stay identical across forms.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help="Ajustar SÓLO con las etiquetas de este juez. 'all' mezcla los "
                             "dos jueces del golden set, que es lo que hacía la versión "
                             "anterior y no es defendible (ver load_golden).")
    parser.add_argument("--from-manifest", type=Path, default=None,
                        help="Which deployed model --apply-only uses (default: newest manifest "
                             "in data/interim/prefilter_predictions_unique/).")
    cli = parser.parse_args()
    if cli.apply_only:
        apply_only(cli.from_manifest or latest_manifest())
    elif cli.from_manifest is not None:
        parser.error("--from-manifest only means something with --apply-only")
    else:
        main(None if cli.judge_model == "all" else cli.judge_model)
