"""Trains the final AI-MENTION classifier (logistic regression on the
prefilter's 12 signals — see docs/prefilter_evaluation.md §8/§8.1) and
applies it to the FULL scored corpus to produce the AI-candidate set that
scripts/enrichment/ai_classify.py's frame extraction should actually run over.

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
`ai_prefilter.py --source-relation unique_paragraphs` (bronze.unique_paragraphs,
the canonical dedup table) scores each of the 1,651,191 unique texts exactly
ONCE, and this script trains/applies/reports over THAT population as the
one and only unit of work. The funnel has one number per stage, not two;
`duplicate_count` (carried through from `unique_paragraphs`) is the only
place instance-level scale still shows up, as a single derived total.

Output: data/interim/prefilter_predictions_unique/prefilter_predictions__run=<id>.parquet
(one row per UNIQUE paragraph text, every one — not just the positives —
so a downstream reader never has to treat "absent" as "negative" by
assumption) plus a manifest JSON with the deployment decisions (`deploy_c`,
threshold, coefficients — plain JSON, not a pickle, since this is an
11-feature linear model with no reason to carry a full sklearn object and
its version-pinning risk). The out-of-fold CV metrics and the funnel counts
are computed separately, over this same bronze/silver data, by
scripts/analytics/prefilter/logit_cv_metrics.py and
scripts/analytics/prefilter/funnel.py.

The earlier data/interim/prefilter_predictions/ (per-instance, superseded)
is kept on disk, never deleted, per the project's data-retention rule --
it is simply no longer what ai_classify.py reads.

Usage:
    uv run python scripts/enrichment/ai_prefilter_classify.py
"""

import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

GOLDEN_DIR = L.INTERIM / "golden_set"
GOLDEN_LABEL_GLOB = "golden_set_labels__session=*__part=*.parquet"
# El juez por defecto es el mismo que etiqueta el golden set: si cambia allá,
# cambia acá, y no queda un string duplicado que se desincronice.
DEFAULT_JUDGE_MODEL = "qwen/qwen3.7-flash"

OUT_DIR = REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique"


def read_parts(files: Iterable[Path]) -> pl.LazyFrame:
    """Une partes parquet por nombre de columna (una columna ausente en una
    parte queda nula; tipos compatibles se promueven)."""
    files = list(files)
    if not files:
        raise FileNotFoundError("no parquet parts matched")
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")


def current_scores_run() -> str:
    """Run id de la corrida de scores vigente: la más nueva dentro de la
    población que `scores_relation()` elige."""
    return scores_relation().select(pl.col("run_id").max()).collect().item()


def scores_relation() -> pl.LazyFrame:
    """The current score population, one row per text: `bronze.prefilter_scores`
    (rules in scripts/bronze/prefilter.py — newest (model, anchors, dtype)
    configuration, newest score per text). Rebuild bronze after
    ai_prefilter.py appends new score parts."""
    return L.scan("bronze.prefilter_scores")

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
    ("strong_ge1", pl.col("strong_matched_terms").list.len() >= 1),
    ("strong_ge2", pl.col("strong_matched_terms").list.len() >= 2),
    ("weak_ge1", pl.col("weak_matched_terms").list.len() >= 1),
]


def lower_text(text: pl.Expr) -> pl.Expr:
    """`lower(coalesce(text, ''))` con mapeo simple por carácter: la "İ" pasa a
    "i" (no a "i" + punto combinante) y la sigma mayúscula siempre a "σ" (sin
    forma final según contexto), para que los largos en caracteres no dependan
    de reglas de minúsculas sensibles al contexto."""
    return (text.fill_null("").str.replace_all("İ", "i", literal=True)
            .str.replace_all("Σ", "σ", literal=True).str.to_lowercase())


# Límite de palabra ASCII (`(?-u:\b)`) y espacio en blanco ASCII: un espacio no
# separable (U+00A0) o una letra acentuada NO son espacio ni carácter de palabra.
WORD_BOUNDARY = r"(?-u:\b)"
WHITESPACE_RUN = r"[\t\n\f\r ]+"

# ---------------------------------------------------------------------------
# Señales de TEXTO (2026-09-06)
# ---------------------------------------------------------------------------
# Motivo: el modelo de 11 señales rendía F1 pond. 0,925 en 10-K/10-Q y 0,755 /
# 0,647 en DEF 14A / 8-K (§8.15). Al leer sus falsos positivos en esos
# formularios, todos comparten forma, no tema:
#
#   - matrices de habilidades del directorio: "artificial intelligence" es UNA
#     celda dentro de una tabla de 14.350 caracteres;
#   - biografías de directores: "Mr. Wang is the founder and CEO of Scale AI";
#   - resultados de votación en 8-K: "a stockholder proposal regarding a report
#     on risks of discrimination in GenAI was not approved";
#   - regulación de terceros: "Export Control Framework for Artificial
#     Intelligence Diffusion... Federal Register";
#   - viñetas de temas de comité: "emerging technologies, including artificial
#     intelligence".
#
# El vector denso no puede verlos: en un párrafo de 14 mil caracteres la
# similitud semántica se diluye, y las dummies léxicas sólo dicen "apareció el
# término". Lo que distingue esos casos es DÓNDE y CÓMO aparece el término:
# densidad, marcado de tabla, contexto de biografía/votación/regulación, y sobre
# todo si el párrafo habla en primera persona (una divulgación de la propia
# empresa dice "we"/"our"; una biografía o una norma, no).
#
# NINGUNA de estas señales mira el tipo de documento. Meter `form` como feature
# sería un atajo: el modelo aprendería "los proxies mencionan más IA" en vez de
# leer el párrafo, y como el hallazgo central del proyecto ES una comparación
# entre formularios (docs/analytics/apendice/descriptivos_sql_corpus.md #8), la medición quedaría circular.
# Todo lo de abajo se calcula del texto y valdría igual si el mismo párrafo
# apareciera en cualquier otro documento.
STRONG_SPELLED_OUT = (
    "artificial intelligence", "generative ai", "gen ai", "machine learning",
    "deep learning", "large language model", "large language models",
    "foundation model", "foundation models", "neural network", "neural networks",
)
STRONG_ALTERNATION = (
    "ai|ml|llm|llms|agi|nlp|gen[ ]ai|artificial[ ]intelligence|generative[ ]ai|genai|"
    "machine[ ]learning|deep[ ]learning|large[ ]language[ ]models?|"
    "foundation[ ]models?|neural[ ]networks?"
)
BIO_CONTEXT = "mr\\.|ms\\.|mrs\\.|dr\\.|director since|age [0-9]{2}|ph\\.d|founder of|served as"
VOTE_CONTEXT = "stockholder proposal|shareholder proposal|votes cast|broker non-vote|abstentions"
RULE_CONTEXT = "federal register|final rule|executive order|export control"


def text_feature_columns(text: pl.Expr) -> list[tuple[str, pl.Expr]]:
    """(nombre, expresión) de cada señal de texto, para el golden set y para el
    corpus. Las señales de anchors y de términos se leen de las columnas de
    score (`score_ai_*`, `strong_matched_terms`) del mismo frame.

    `text` es la columna que trae el texto del párrafo en esa consulta — cambia
    según desde dónde se lea, pero la definición de la señal no puede cambiar,
    y por eso se genera desde acá en vez de escribirse dos veces."""
    lower = lower_text(text)
    length = pl.max_horizontal(lower.str.len_chars().cast(pl.Int64), pl.lit(1, pl.Int64))
    words = pl.max_horizontal(lower.str.count_matches(WHITESPACE_RUN).cast(pl.Int64) + 1,
                              pl.lit(1, pl.Int64))
    strong = f"{WORD_BOUNDARY}({STRONG_ALTERNATION}){WORD_BOUNDARY}"
    first_person = f"{WORD_BOUNDARY}(we|our|us|the company){WORD_BOUNDARY}"
    # Partición barata en oraciones: puntuación final, salto de línea o celda de
    # tabla. No es un segmentador lingüístico y no necesita serlo — lo que
    # importa es aislar el fragmento donde aparece el término. Los tramos entre
    # separadores de más de 2 caracteres son las oraciones.
    sentences = lower.str.extract_all(r"[^.!?\n|]+").list.eval(
        pl.element().filter(pl.element().str.len_chars() > 2))
    n_sentences = pl.max_horizontal(sentences.list.len().cast(pl.Int64), pl.lit(1, pl.Int64))
    anchors = [pl.col(c) for c in SIGNAL_COLUMNS[:7]]
    sorted_anchors = pl.concat_list(anchors).list.sort(descending=True)

    def rate(count: pl.Expr, scale: float, denominator: pl.Expr) -> pl.Expr:
        return count.cast(pl.Float64) * scale / denominator

    return [
        # Longitud: los falsos positivos de tabla son párrafos gigantes.
        ("log_len", (length + 1).cast(pl.Float64).log()),
        # Densidad del término: "IA" una vez en 14 mil caracteres no es
        # divulgación; tres veces en 400 caracteres, casi siempre sí.
        ("strong_per_1k", rate(lower.str.count_matches(strong), 1000.0, length)),
        # Primera persona: una empresa que divulga habla de sí misma. Una
        # biografía, una norma o un resultado de votación, no.
        ("first_person_per_100w", rate(lower.str.count_matches(first_person), 100.0, words)),
        # Marcado: tablas y viñetas son donde vive la enumeración sin contenido.
        ("pipe_rate", rate(lower.str.count_matches("|", literal=True), 100.0, length)),
        ("bullet_rate", rate(lower.str.count_matches("•", literal=True), 100.0, length)),
        ("digit_rate", rate(lower.str.count_matches("[0-9]"), 100.0, length)),
        ("bio_context", lower.str.contains(BIO_CONTEXT)),
        ("vote_context", lower.str.contains(VOTE_CONTEXT)),
        ("rule_context", lower.str.contains(RULE_CONTEXT)),
        # Sólo siglas: "AI" suelto es mucho más ambiguo que "artificial
        # intelligence" escrito completo (y es de donde salen los choques con
        # nombres propios y con "AI" dentro de otro token).
        ("acronym_only", (pl.col("strong_matched_terms").list.len() >= 1)
                         & (pl.col("strong_matched_terms").list.eval(
                             pl.element().is_in(list(STRONG_SPELLED_OUT))).list.sum() == 0)),
        # Señales de ORACIÓN. El problema de fondo con las tablas y las
        # biografías es de UNIDAD: el término de IA vive en una oración y el
        # resto del párrafo no tiene nada que ver, pero tanto el embedding como
        # la densidad promedian sobre todo el párrafo. Partir por puntuación y
        # mirar sólo las oraciones que contienen el término separa "we use AI to
        # do X" de "Mr. Wang, founder of Scale AI, age 57, director since 2017".
        ("ai_sent_share", rate(sentences.list.eval(pl.element().str.contains(strong)).list.sum(),
                               1.0, n_sentences)),
        ("ai_sent_first_person", sentences.list.eval(
            pl.element().str.contains(strong) & pl.element().str.contains(first_person)).list.any()),
        ("log_n_sentences", (n_sentences + 1).cast(pl.Float64).log()),
        # Nitidez del match semántico: qué tan por encima está el mejor anchor
        # del segundo. Un párrafo genuino se parece a UNA categoría; el
        # boilerplate se parece un poco a todas.
        ("anchor_gap", sorted_anchors.list.get(0) - sorted_anchors.list.get(1)),
        ("anchor_mean", pl.sum_horizontal([a.cast(pl.Float64) for a in anchors]) / len(anchors)),
    ]


TEXT_FEATURE_NAMES = [name for name, _ in text_feature_columns(pl.lit(""))]

# Señales de ORACIÓN (scripts/enrichment/ai_prefilter_sentences.py). El embedding
# del párrafo se diluye justo donde más falla el prefiltro: una celda de tabla
# con "artificial intelligence" entre cientos de celdas produce un vector que no
# se parece a ninguna divulgación. Estas columnas puntúan SÓLO las oraciones que
# mencionan IA, con el mismo modelo y los mismos anchors, y se agregan por
# texto. Un texto sin fila acá es uno que no pasó la compuerta léxica: se
# rellena con 0, que es lo que corresponde ("no hay oración de IA que puntuar").
SENTENCE_SCORES_DIR = L.INTERIM / "prefilter_sentence_scores"
SENTENCE_SCORES_GLOB = "prefilter_sentence_scores__run=*.parquet"
SENTENCE_FEATURE_NAMES = [
    "sent_n_ai", "sent_max_margin", "sent_mean_margin", "sent_max_semantic",
    "sent_min_negative",
] + [f"sent_max_{c}" for c in ("ai_use", "ai_exploration", "ai_capability",
                              "ai_outcome", "ai_risk", "ai_governance", "ai_strategy")]


def sentence_scores_relation() -> pl.LazyFrame:
    """La corrida de oraciones más nueva, una fila por `text_hash`."""
    return (read_parts(sorted(SENTENCE_SCORES_DIR.glob(SENTENCE_SCORES_GLOB)))
            .sort("run_id", descending=True, maintain_order=True)
            .unique("text_hash", keep="first", maintain_order=True))


def join_sentence_scores(frame: pl.LazyFrame) -> pl.LazyFrame:
    """LEFT JOIN por `text_hash` contra `sentence_scores_relation()`."""
    return frame.join(sentence_scores_relation().select("text_hash", *SENTENCE_FEATURE_NAMES),
                      on="text_hash", how="left")


def sentence_feature_columns() -> list[pl.Expr]:
    """Señales de oración, 0 para los textos sin fila (ver arriba)."""
    return [pl.col(name).fill_null(0).alias(name) for name in SENTENCE_FEATURE_NAMES]


ALL_SIGNAL_COLUMNS = SIGNAL_COLUMNS + [name for name, _ in LEXICAL_STEP_COLUMNS]
PARAGRAPH_KEY = ("country_code", "form", "accession_number", "item_key", "paragraph_index")
# Columnas de `scores_relation()` que consumen los modelos y sus señales.
SCORE_FIELDS = [*SIGNAL_COLUMNS, "max_semantic_score", "strong_matched_terms", "weak_matched_terms",
                "strong_lexical_match", "weak_lexical_match"]


def lexical_step_columns() -> list[pl.Expr]:
    return [expr.alias(name) for name, expr in LEXICAL_STEP_COLUMNS]


def _named_entity_expr(lowered_text: pl.Expr, entities=None) -> pl.Expr:
    """Same word-boundary regex technique as ai_prefilter.py's _term_regex,
    kept independent of that module's anchors-tracked strong/weak lists on
    purpose — this override must survive an anchors/lexical-config change
    without needing a full corpus rescore (see module docstring).
    `lowered_text` is already `lower_text(...)`."""
    boundary_before, boundary_after = "(^|[^a-z0-9])", "([^a-z0-9]|$)"
    alt = "|".join(term.replace(".", "\\.").replace(" ", "[ ]")
                   for term in (NAMED_AI_ENTITIES if entities is None else entities))
    return lowered_text.str.contains(f"{boundary_before}({alt}){boundary_after}")


def scored_unique_texts() -> pl.LazyFrame:
    """`scores_relation()` JOIN bronze.unique_paragraphs por `text_hash`, sólo
    textos `is_scorable`: la población sobre la que se aplica el modelo. Trae la
    llave de instancia del score, `duplicate_count` y `paragraph_text`."""
    unique = L.scan("bronze.unique_paragraphs").select(
        "text_hash", "duplicate_count", "paragraph_text", "is_scorable",
        pl.col("country_code").alias("up_country_code"), pl.col("form").alias("up_form"))
    return (scores_relation().select(*PARAGRAPH_KEY, "text_hash", *SCORE_FIELDS)
            .join(unique, on="text_hash").filter(pl.col("is_scorable")))


def join_scores_via_paragraphs(labels: pl.LazyFrame) -> pl.LazyFrame:
    """Etiquetas con llave de instancia -> el texto de ESA instancia
    (bronze.paragraphs, columna `paragraph_text`) -> su texto único
    (bronze.unique_paragraphs, por `text_hash`) -> la fila de score, unida por la
    llave del representante que eligió unique_paragraphs (ver `load_golden`).
    Deja `is_scorable` sin filtrar; el `text_hash` propio de la etiqueta se
    descarta a favor del de bronze.paragraphs."""
    keys = list(PARAGRAPH_KEY)
    accessions = labels.select("accession_number").unique().collect().to_series()
    paragraphs = (L.scan("bronze.paragraphs").filter(pl.col("accession_number").is_in(accessions.implode()))
                  .select(*keys, "text_hash", "paragraph_text"))
    representative = [f"up_{k}" for k in keys]
    unique = L.scan("bronze.unique_paragraphs").select(
        "text_hash", "is_scorable", "duplicate_count",
        *[pl.col(k).alias(r) for k, r in zip(keys, representative)])
    scores = scores_relation().select(*[pl.col(k).alias(r) for k, r in zip(keys, representative)],
                                      *SCORE_FIELDS)
    return (labels.drop("text_hash", "paragraph_text", strict=False)
            .join(paragraphs, on=keys).join(unique, on="text_hash").join(scores, on=representative))


def join_scores_via_unique_paragraphs(labels: pl.LazyFrame) -> pl.LazyFrame:
    """Etiquetas cuya llave ES la del representante de bronze.unique_paragraphs
    (muestras de DEF 14A / 8-K / calls) -> ese texto único -> su score por
    `text_hash`. Deja `is_scorable` sin filtrar."""
    keys = list(PARAGRAPH_KEY)
    unique = L.scan("bronze.unique_paragraphs").select(
        *keys, "text_hash", "paragraph_text", "is_scorable", "duplicate_count")
    return (labels.drop("text_hash", "paragraph_text", "is_scorable", "duplicate_count", strict=False)
            .join(unique, on=keys)
            .join(scores_relation().select("text_hash", *SCORE_FIELDS), on="text_hash"))


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
    return (golden_scored(judge_model)
            .select("is_ai_disclosure", "relevance",
                    (pl.col("relevance") != "none").alias("is_ai_mention"),
                    "inclusion_weight", "accession_number", *SIGNAL_COLUMNS,
                    *lexical_step_columns(),
                    _named_entity_expr(lower_text(pl.col("paragraph_text"))).alias("named_entity_match"))
            .collect().to_pandas())


def golden_labels(judge_model: str | None = None) -> pl.LazyFrame:
    """Etiquetas legibles del golden set (`error` nulo), opcionalmente de un
    solo juez."""
    labels = read_parts(sorted(GOLDEN_DIR.glob(GOLDEN_LABEL_GLOB))).filter(pl.col("error").is_null())
    if judge_model:
        labels = labels.filter(pl.col("judge_model") == judge_model)
    return labels


def golden_scored(judge_model: str | None = None) -> pl.LazyFrame:
    """Golden set unido a sus scores (ver `join_scores_via_paragraphs`), sólo
    textos `is_scorable`, en orden de llave de instancia."""
    return (join_scores_via_paragraphs(golden_labels(judge_model))
            .filter(pl.col("is_scorable"))
            .sort([*PARAGRAPH_KEY, "session_id"], maintain_order=True))


def load_golden_judges(judge_model: str | None = None) -> pd.Series:
    """Qué juez etiquetó cada fila que entra al fit. Va al manifiesto: un
    despliegue tiene que poder decir con qué criterio se definió su target."""
    return golden_scored(judge_model).select("judge_model").collect().to_pandas()["judge_model"]


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

    print("Cargando el corpus de textos únicos (misma config de anchors)...")
    corpus = (scored_unique_texts()
              .select(*PARAGRAPH_KEY, "text_hash", "duplicate_count", *SIGNAL_COLUMNS,
                      *lexical_step_columns(),
                      _named_entity_expr(lower_text(pl.col("paragraph_text")),
                                         deployed["named_ai_entities"]).alias("named_entity_match"))
              .sort([*PARAGRAPH_KEY, "text_hash"]).collect().to_pandas())
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

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = corpus[[*PARAGRAPH_KEY, "text_hash", "duplicate_count"]].copy()
    out["predicted_proba"] = proba.astype("float32")
    out["named_entity_match"] = named_entity_corpus
    out["is_ai_prefiltered"] = is_positive
    out["threshold"] = deployed["threshold"]
    # `model_version` stays THIS run's id, because that is what downstream
    # orders by: bronze.prefilter_predictions resolves a text present in
    # several prediction files by the latest `model_version`, so
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
        "named_entity_used_in_deployment": deployed["use_named_entity"],
        "named_ai_entities": list(deployed["named_ai_entities"]),
        "coefficients": dict(zip(ALL_SIGNAL_COLUMNS, deployed["coef"].tolist())),
        "intercept": deployed["intercept"],
        "output": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = OUT_DIR / f"prefilter_predictions_manifest__run={run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nParquet -> {out_path}")
    print(f"Manifiesto -> {manifest_path}")


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
    deploy_c, threshold, _oof_metrics = cv_threshold_and_metrics(X, y, weights, groups)
    print(f"C/threshold para despliegue (elegidos sobre todo el set, no es la métrica reportada): "
          f"C={deploy_c}, threshold={threshold:.2f}")
    print("Métrica CV completa (modelo solo y modelo OR named_entity): "
          "scripts/analytics/prefilter/logit_cv_metrics.py")

    # FORZADO A True (2026-09-04, §8.13) -- el criterio automático de comparar
    # nested-CV con y sin este override rechaza la diferencia por ser chica
    # (0,001 F1 pond. sobre sólo 76 filas del golden set con
    # named_entity_match; ver logit_cv_metrics.py para esa comparación). Esa
    # muestra es demasiado chica y no representativa para este caso: el
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
    print("named_entity_match: forzado a incluir en el despliegue (ver comentario en código, §8.13).")

    print("\nReajustando el modelo final sobre TODO el golden set...")
    final_model = LogisticRegression(max_iter=2000, C=deploy_c)
    final_model.fit(X, y, sample_weight=fit_weights)

    print("Cargando el corpus de textos únicos (última corrida de anchors)...")
    # `is_scorable` (bronze.unique_paragraphs) descarta párrafos sin contenido real
    # (<=3 caracteres útiles, o sin ningún alfanumérico) -- verificado en
    # producción (docs/prefilter_evaluation.md §8.9) que sin este filtro el
    # modelo puede marcar basura como positiva: el párrafo literal "AI" (2
    # caracteres, is_scorable=false) salía con predicted_proba=0.86 solo por
    # matchear el término léxico fuerte, sin nada de contenido detrás.
    #
    # JOIN por text_hash, no por llave de instancia (docs/
    # prefilter_evaluation.md §8.9): el text_hash es la identidad real del
    # contenido y no cambia aunque el representante elegido sí lo haga.
    corpus = (scored_unique_texts()
              .select(*PARAGRAPH_KEY, "text_hash", "duplicate_count", *SIGNAL_COLUMNS,
                      *lexical_step_columns(),
                      _named_entity_expr(lower_text(pl.col("paragraph_text"))).alias("named_entity_match"))
              .sort([*PARAGRAPH_KEY, "text_hash"]).collect().to_pandas())
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

    anchors_run = current_scores_run()

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
        "named_entity_used_in_deployment": use_named_entity,
        "named_ai_entities": list(NAMED_AI_ENTITIES),
        "coefficients": dict(zip(ALL_SIGNAL_COLUMNS, final_model.coef_[0].tolist())),
        "intercept": float(final_model.intercept_[0]),
        "output": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = OUT_DIR / f"prefilter_predictions_manifest__run={run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nParquet -> {out_path}")
    print(f"Manifiesto -> {manifest_path}")


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
