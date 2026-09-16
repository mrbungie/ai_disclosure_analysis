"""Semantic frame extraction over AI-disclosure paragraphs -- Pass-1 of the
v2 pipeline (spec: docs, 2026-09-12). Replaces the deprecated
scripts/deprecated/ai_classify.py -- see data/deprecated/POINTER.json for why.

Population classified: the deployed prefilter's `is_ai_prefiltered=True`
texts in bronze.prefilter_predictions (latest `model_version` per text over
`unique_paragraphs`) -- one row per unique paragraph TEXT.

Same operational contract as the deprecated version: additive (nothing is
ever overwritten or deleted), atomic part files every `--part-rows` rows,
a failed API call is written as an error row (not skipped) so a
credit/network cutoff retries itself next run, and coverage is checked by
`text_hash` (judge-model-agnostic).

Checkpointing: rows are buffered and flushed to an atomic parquet part every
`--part-rows` (default 250). On SIGINT/SIGTERM, in-flight tasks are
cancelled and whatever is currently buffered is flushed immediately before
exit -- nothing collected so far is lost, nothing partially-written is left
in a non-atomic state (write to `.partial`, then `replace()`).

Usage:
    uv run python scripts/enrichment/ai_classify.py --limit 50   # smoke test
    uv run python scripts/enrichment/ai_classify.py              # everything pending
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

# every earnings-call source keeps its own manifest (Hugging Face, Equibles,
# stockanalysis, the Equibles backfill): the ticker of a call can live in any of them
EARNINGS_CALLS_MANIFESTS = sorted((L.INTERIM / "manifests").glob("filing_manifest_earnings_calls*.parquet"))
DEFAULT_DIR = REPO_ROOT / "data" / "interim" / "ai_classify"
FRAME_GLOB = "ai_frames__session=*.parquet"

PARAGRAPH_KEY = ("country_code", "form", "accession_number", "item_key", "paragraph_index")
DEFAULT_JUDGE_MODEL = "openai/gpt-5.6-luna"  # ver docs/judge_model_selection.md
PROMPT_VERSION = "v2"
DEFAULT_CONCURRENCY = 6  # el spec pedía 32; medido en la práctica contra OpenRouter/qwen -- ver comentario en argparse


# --------------------------------------------------------------------------
# Salida estructurada -- spec v2 §1.1
# --------------------------------------------------------------------------

Subject = Literal["firm", "partners", "customers", "industry", "regulators"]
AIType = Literal["generative", "predictive_ml", "unspecified"]
Temporal = Literal["realized", "forward", "hypothetical"]

Concept = Literal[
    # adopción y habilitadores
    "deployed", "pilot", "exploring", "scaling",
    "proprietary_ai", "third_party_ai", "infrastructure", "talent", "investment",
    # resultados
    "productivity_outcome", "cost_outcome", "revenue_outcome", "customer_outcome",
    # riesgos
    "risk_cyber", "risk_privacy", "risk_regulatory", "risk_ip", "risk_bias",
    "risk_reliability", "risk_competitive", "risk_workforce", "risk_operational",
    "risk_reputational",
    # gobernanza
    "gov_board", "gov_management", "gov_policy", "gov_technical", "gov_human", "gov_vendor",
]

Specificity = Literal["process", "product", "vendor", "metric", "timeline"]
Rhetoric = Literal["promotional", "strategic", "hedged"]
Valence = Literal["positive", "negative", "mixed", "not_applicable"]
MODEL_INCONSISTENT = "model_inconsistent"  # never part of Valence -- the model must not see or choose this

ADOPTION_CONCEPTS = {
    "deployed", "pilot", "exploring", "scaling",
    "proprietary_ai", "third_party_ai", "infrastructure", "talent", "investment",
}
GOVERNANCE_CONCEPTS = {"gov_board", "gov_management", "gov_policy", "gov_technical", "gov_human", "gov_vendor"}
OUTCOME_CONCEPTS = {"productivity_outcome", "cost_outcome", "revenue_outcome", "customer_outcome"}


class AIFrame(BaseModel):
    subject: Subject
    ai_type: AIType
    temporal: Temporal
    concepts: list[Concept] = Field(min_length=1)
    specificity: list[Specificity] = Field(default_factory=list)
    rhetoric: list[Rhetoric] = Field(default_factory=list)
    valence: Valence  # required, no default -- "null" must never be a free lazy answer
    sentence_ids: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _dedupe_lists(self):
        # The model sometimes repeats the same value twice in one list (seen:
        # concepts=['productivity_outcome', 'productivity_outcome']) -- harmless
        # to keep, but it silently inflates concept/specificity/rhetoric counts
        # for anything downstream that counts list membership. dict.fromkeys
        # preserves first-seen order, unlike set().
        self.concepts = list(dict.fromkeys(self.concepts))  # type: ignore[assignment]
        self.specificity = list(dict.fromkeys(self.specificity))  # type: ignore[assignment]
        self.rhetoric = list(dict.fromkeys(self.rhetoric))  # type: ignore[assignment]
        return self

    @model_validator(mode="after")
    def _valence_iff_outcome(self):
        # Never raises: a cross-field mismatch here would fail the whole tool
        # call under reasoning=none (no room for the model to self-correct),
        # dropping every OTHER correct field (subject/temporal/concepts) in
        # this frame along with it. Instead, flag the mismatch explicitly so
        # it stays auditable -- "model_inconsistent" is never confusable with
        # a real "not_applicable" judgment.
        has_outcome = bool(OUTCOME_CONCEPTS & set(self.concepts))
        chose_applicable = self.valence != "not_applicable"
        if has_outcome != chose_applicable:
            # Deliberately outside the `Valence` type declared above (which is
            # exactly what pydantic-ai turns into the model-facing schema) --
            # the model must never see or choose this value. Pydantic v2 does
            # not revalidate plain attribute assignment by default, so this is
            # safe at runtime; a static type checker will (correctly) flag it.
            self.valence = MODEL_INCONSISTENT  # type: ignore[assignment]
        return self


class ParagraphExtraction(BaseModel):
    frames: list[AIFrame] = Field(default_factory=list)


SYSTEM_PROMPT = """\
You extract structured "frames" about AI from one paragraph of a corporate filing \
(10-K/10-Q), for research on AI disclosure. You are told which firm files the \
document (ticker and name): that firm, its subsidiaries and its products \
('our X', or the firm named in third person, e.g. a quoted press release) are \
subject=firm. Sentences are numbered [0], [1], ... Cite them by index; never \
reproduce sentence text.

A frame is one coherent proposition about AI: who (subject), what kind of AI \
(ai_type), when (temporal), and which concepts it asserts. Create the minimum \
number of frames that keeps pairings correct: one sentence asserting several \
concepts about the same subject and time is ONE frame. Split only when merging \
would pair information from different propositions (e.g. a realized use and a \
planned use). Zero frames is a valid and common answer.

Rules:
- subject is whoever performs the action. "We face evolving AI regulation" is \
subject=firm with risk_regulatory; "the EU AI Act requires providers to..." is \
subject=regulators; customer or market demand for the firm's products is \
subject=customers. partners = a named supplier, vendor, or joint-venture \
partner acting on its own (e.g. "NVIDIA's CUDA platform").
- A frame captures only what the text affirms; a negated statement ("we do not \
use generative AI") earns no frame.
- temporal tracks the state of the fact, for whichever subject: realized = \
already exists, done, or ongoing (a competitor's or the industry's current \
capability counts, even mentioned briefly or in a list); forward = a stated \
future intent ("plans to", "will", "expects to"); hypothetical = conditional \
possibility or risk ("could", "may result in", "if... then").
- valence: every frame states one. positive for gains or savings, negative \
for losses or cost increases, mixed for both -- whenever a *_outcome concept \
is present. not_applicable otherwise.
- specificity lists the concrete evidence given: process (a named business \
process), product (a named product or system), vendor (a named vendor or \
partner), metric (a quantified figure), timeline (a date or period).
- rhetoric: promotional = superlatives or self-praise; strategic = AI framed \
as central to strategy; hedged = non-committal wording about the firm's own \
realized or forward activity ("may", "could", "seek to") -- reserved for that \
case, since risk-factor conditionals already carry temporal=hypothetical.
- ai_type follows whatever the text names, for whichever entity: assign \
predictive_ml/generative the moment the proposition says machine learning, \
deep learning, an LLM, or generative AI, even for a competitor or the industry.
- sentence_ids cites the full set of sentences behind subject, ai_type, \
temporal and every concept -- both sentences when a frame's evidence spans two."""


def build_prompt(sentences: list[str], firm_ticker: str = "", firm_name: str = "") -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    header = f"Filing firm: {firm_ticker} ({firm_name})\n" if firm_ticker or firm_name else ""
    return f"{header}{numbered}"


def validate_frames(extraction: ParagraphExtraction, n_sentences: int) -> list[str]:
    """Post-parse checks beyond what Pydantic enforces structurally (needs the
    runtime sentence count, which the schema doesn't have access to)."""
    errors = []
    for i, frame in enumerate(extraction.frames):
        bad = [s for s in frame.sentence_ids if not (0 <= s < n_sentences)]
        if bad:
            errors.append(f"frame {i}: sentence_ids {bad} out of range [0, {n_sentences})")
    return errors


def merge_duplicate_frames(frames: list[AIFrame]) -> list[AIFrame]:
    """Fixes a real model failure mode post-hoc: splitting one proposition into
    several near-identical frames instead of respecting the prompt's "minimum
    number of frames" rule (same subject and temporal -> one frame). Merges
    only when frames ALSO share at least one evidence sentence -- a strong
    signal it's the same proposition, not two genuinely distinct ones that
    happen to share subject/temporal by coincidence."""
    merged: list[AIFrame] = []
    for frame in frames:
        match = next((m for m in merged
                      if m.subject == frame.subject and m.temporal == frame.temporal
                      and set(m.sentence_ids) & set(frame.sentence_ids)), None)
        if match is None:
            merged.append(frame.model_copy(deep=True))
            continue
        match.concepts = list(dict.fromkeys(match.concepts + frame.concepts))
        match.specificity = list(dict.fromkeys(match.specificity + frame.specificity))
        match.rhetoric = list(dict.fromkeys(match.rhetoric + frame.rhetoric))
        match.sentence_ids = sorted(set(match.sentence_ids) | set(frame.sentence_ids))
        if match.ai_type == "unspecified" and frame.ai_type != "unspecified":
            match.ai_type = frame.ai_type  # type: ignore[assignment]
        has_outcome = bool(OUTCOME_CONCEPTS & set(match.concepts))
        if has_outcome:
            real_valence = next((v for v in (match.valence, frame.valence)
                                 if v in ("positive", "negative", "mixed")), None)
            match.valence = real_valence if real_valence else MODEL_INCONSISTENT  # type: ignore[assignment]
        else:
            match.valence = "not_applicable"
    return merged


# --------------------------------------------------------------------------
# Partes: verificación y escritura atómica -- mismo contrato que v1
# --------------------------------------------------------------------------

FRAME_SCHEMA = pa.schema([
    ("country_code", pa.string()), ("form", pa.string()),
    ("accession_number", pa.string()), ("item_key", pa.string()),
    ("paragraph_index", pa.int64()), ("text_hash", pa.uint64()),
    ("duplicate_count", pa.int64()),
    ("frame_id", pa.int64()), ("has_frame", pa.bool_()),
    ("subject", pa.string()), ("ai_type", pa.string()), ("temporal", pa.string()),
    ("concepts", pa.list_(pa.string())),
    ("specificity", pa.list_(pa.string())),
    ("rhetoric", pa.list_(pa.string())),
    ("valence", pa.string()),
    ("sentence_ids", pa.list_(pa.int64())),
    ("judge_model", pa.string()), ("prompt_version", pa.string()),
    ("session_id", pa.string()), ("classified_at", pa.string()), ("error", pa.string()),
])


def frame_parts(directory: Path) -> list[Path]:
    return sorted(directory.glob(FRAME_GLOB))


def commit_part(rows: list[dict], directory: Path, session_id: str, index: int) -> Path:
    final = directory / f"ai_frames__session={session_id}__part={index:05d}.parquet"
    staging = final.with_suffix(".parquet.partial")
    pq.write_table(pa.Table.from_pylist(rows, schema=FRAME_SCHEMA), staging, compression="zstd")
    staging.replace(final)
    return final


def _base_record(row: dict, judge_model: str, session_id: str) -> dict:
    return {
        **{key: row[key] for key in PARAGRAPH_KEY},
        "text_hash": row["text_hash"],
        "duplicate_count": row["duplicate_count"],
        "judge_model": judge_model, "prompt_version": PROMPT_VERSION,
        "session_id": session_id, "classified_at": datetime.now(timezone.utc).isoformat(),
    }


def _empty_frame_fields() -> dict:
    return {"subject": None, "ai_type": None, "temporal": None, "concepts": [],
            "specificity": [], "rhetoric": [], "valence": None, "sentence_ids": []}


def _frame_rows(row: dict, judge_model: str, session_id: str, extraction: ParagraphExtraction) -> list[dict]:
    base = _base_record(row, judge_model, session_id)
    frames = merge_duplicate_frames(extraction.frames)
    if not frames:
        return [{**base, "frame_id": None, "has_frame": False, **_empty_frame_fields(), "error": None}]
    return [{**base, "frame_id": i, "has_frame": True,
             "subject": f.subject, "ai_type": f.ai_type, "temporal": f.temporal,
             "concepts": list(f.concepts), "specificity": list(f.specificity),
             "rhetoric": list(f.rhetoric), "valence": f.valence,
             "sentence_ids": list(f.sentence_ids), "error": None}
            for i, f in enumerate(frames)]


def _error_row(row: dict, judge_model: str, session_id: str, error: Exception) -> dict:
    return {**_base_record(row, judge_model, session_id), "frame_id": None, "has_frame": None,
            **_empty_frame_fields(), "error": f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"}


async def classify_rows(
    rows: list[dict],
    directory: Path,
    session_id: str,
    judge_model: str,
    concurrency: int,
    part_rows: int,
    progress_every: int,
) -> tuple[list[Path], dict]:
    from pydantic_ai import Agent
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider

    # Qwen-via-Alibaba workaround for a real, closed-as-not-planned pydantic-ai
    # bug (github.com/pydantic/pydantic-ai/issues/5287) -- harmless no-op for
    # any other model/provider, kept as a generic safety net.
    class QwenSafeOpenRouterModel(OpenRouterModel):
        async def _map_messages(self, messages, model_request_parameters, *, model_settings=None):
            mapped = await super()._map_messages(
                messages, model_request_parameters, model_settings=model_settings)
            for m in mapped:
                if m.get("role") == "assistant" and m.get("content") is None:
                    m["content"] = ""
            return mapped

    model = QwenSafeOpenRouterModel(
        judge_model,
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
        profile=OpenRouterModelProfile(openrouter_supports_cache_control=True),
        settings={"openrouter_cache_instructions": True},
    )
    # docs/judge_model_selection.md: openai/gpt-5.6-luna via amazon-bedrock/
    # us-east-1 with reasoning disabled won the 6-paragraph comparison on
    # speed (~3s/call) AND valence consistency (0% model_inconsistent vs
    # 9-33% for every other candidate tried). allow_fallbacks=True (unlike
    # the pinned, no-fallback comparisons) -- a full-corpus run should
    # degrade to another OpenRouter provider rather than stall entirely if
    # amazon-bedrock has a bad moment; a hard failure still just becomes an
    # error row, retried next run, never fatal to the batch.
    model_settings = OpenRouterModelSettings(
        openrouter_provider={"order": ["amazon-bedrock/us-east-1"], "allow_fallbacks": True},
        extra_body={"reasoning": {"effort": "none"}},
    )
    # retries=2: pydantic-ai's own structural self-correction (schema/Literal/
    # our valence-iff-outcome validator) -- separate from, and in addition to,
    # our own single semantic retry below (needs the runtime sentence count to
    # validate sentence_ids, which a bare Pydantic validator can't see). Either
    # layer failing for good is NOT fatal to the run: it's written as an error
    # row (error IS NOT NULL) and picked up again by `fetch_pending` next run.
    agent = Agent(model, output_type=ParagraphExtraction, system_prompt=SYSTEM_PROMPT, retries=2,
                 model_settings=model_settings)

    written: list[Path] = []
    buffered: list[dict] = []
    done = failed = n_frames = 0
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    interrupted = asyncio.Event()

    def flush() -> None:
        nonlocal buffered
        if buffered:
            written.append(commit_part(buffered, directory, session_id, len(written)))
            buffered = []

    async def classify_one(row: dict) -> list[dict]:
        nonlocal done, failed, n_frames
        async with semaphore:
            if interrupted.is_set():
                raise asyncio.CancelledError
            prompt = build_prompt(row["sentences"], row.get("ticker", ""), row.get("firm", ""))
            n = len(row["sentences"])
            try:
                result = await agent.run(prompt)
                errors = validate_frames(result.output, n)
                if errors:
                    retry_prompt = (f"{prompt}\n\n---\nYour previous answer was invalid: "
                                     f"{'; '.join(errors)}. Correct it and answer again.")
                    result = await agent.run(retry_prompt)
                    errors = validate_frames(result.output, n)
                    if errors:
                        raise ValueError(f"invalid sentence_ids after retry: {'; '.join(errors)}")
                out = _frame_rows(row, judge_model, session_id, result.output)
                done += 1
                n_frames += sum(1 for r in out if r["has_frame"])
                return out
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failed += 1
                return [_error_row(row, judge_model, session_id, error)]

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, interrupted.set)
        except (NotImplementedError, RuntimeError):
            pass

    tasks = [asyncio.create_task(classify_one(row)) for row in rows]
    try:
        for completed in asyncio.as_completed(tasks):
            try:
                buffered.extend(await completed)
            except asyncio.CancelledError:
                continue
            total = done + failed
            if progress_every and total % progress_every == 0:
                rate = total / max(time.perf_counter() - started, 1e-9)
                eta = (len(rows) - total) / rate if rate else 0
                print(f"[ai-classify] {total:,}/{len(rows):,} paragraphs | {done:,} ok "
                      f"({n_frames:,} frames) | {failed:,} error | {rate:.1f} it/s | "
                      f"ETA {int(eta // 60):02d}:{int(eta % 60):02d}", flush=True)
            if len(buffered) >= part_rows:
                flush()
                print(f"  parte escrita ({len(written)} en esta sesión)", flush=True)
    finally:
        # SIGINT/SIGTERM: cancel whatever hasn't started/finished and flush
        # whatever made it into `buffered` so far -- never lost, never left
        # as a partial (non-atomic) file (commit_part writes .partial then
        # replaces).
        for task in tasks:
            task.cancel()
        flush()

    return written, {"classified": done, "failed": failed, "frames": n_frames,
                     "interrupted": interrupted.is_set(), "seconds": time.perf_counter() - started}


# --------------------------------------------------------------------------
# Población pendiente
# --------------------------------------------------------------------------

def doc_firm() -> pl.LazyFrame:
    """accession_number -> ticker y nombre de la empresa (10-K/proxy/8-K,
    10-Q y earnings calls, cuyo `document_id` hace de accession_number).
    Cuando hay más de un valor se toma el menor, de forma determinista."""
    us = pl.col("country_code") == "us"
    docs = pl.concat([
        L.scan("silver.filing_manifest").filter(us).select("accession_number", "ticker"),
        L.scan("silver.filing_manifest_10q").filter(us).select("accession_number", "ticker"),
        *[pl.scan_parquet(m).select(pl.col("document_id").alias("accession_number"), "ticker")
          for m in EARNINGS_CALLS_MANIFESTS],
    ])
    names = (L.scan("silver.firm_universe").filter(us).group_by("ticker")
             .agg(pl.col("company_name").drop_nulls().min()))
    return (docs.join(names, on="ticker", how="left").group_by("accession_number")
            .agg(pl.col("ticker").drop_nulls().min(), pl.col("company_name").drop_nulls().min()))


def attach_sentences(reps: list[dict]) -> list[dict]:
    """Oraciones (bronze.sentences) de cada representante, en orden de llave y
    de `sentence_index`."""
    keys = list(PARAGRAPH_KEY)
    frame = pl.DataFrame([{k: r[k] for k in keys} for r in reps],
                         schema={"country_code": pl.String, "form": pl.String,
                                 "accession_number": pl.String, "item_key": pl.String,
                                 "paragraph_index": pl.Int64})
    return (frame.lazy()
            .join(L.scan("bronze.sentences")
                  .filter(pl.col("accession_number").is_in(frame["accession_number"].unique().implode()))
                  .select(*keys, "sentence_index", "sentence_text"), on=keys)
            .sort(*keys, "sentence_index").collect().to_dicts())


def fetch_pending(output_dir: Path, limit: int) -> tuple[list[dict], dict]:
    """Returns (representative_rows_to_classify, stats). Population is deduped
    by `text_hash` BEFORE any LLM call -- only one representative paragraph
    instance per unique text is ever classified."""
    existing_parts = frame_parts(output_dir)
    if existing_parts:
        classified = (pl.concat([pl.scan_parquet(p) for p in existing_parts], how="diagonal_relaxed")
                      .filter(pl.col("error").is_null()).select("text_hash").unique().collect())
    else:
        classified = pl.DataFrame(schema={"text_hash": pl.UInt64})

    # bronze.prefilter_predictions: la predicción del `model_version` más nuevo
    # por texto, y entre despliegues del mismo modelo, la del archivo más nuevo
    # (`model_version` es el id del modelo congelado, no de la corrida).
    positives = L.scan("bronze.prefilter_predictions").filter(pl.col("is_ai_prefiltered"))
    done = positives.join(classified.lazy(), on="text_hash", how="semi")
    counts = pl.concat([
        positives.select(pl.col("duplicate_count").sum().alias("total_positive_instances"),
                         pl.len().alias("unique_positive_texts")),
        done.select(pl.len().alias("already_classified_texts"),
                    pl.col("duplicate_count").sum().alias("already_classified_instances")),
    ], how="horizontal").collect().row(0, named=True)
    stats = {k: int(v or 0) for k, v in counts.items()}

    # subject=firm sólo tiene sentido si el modelo sabe quién es "la firma" --
    # mismo join que ai_activities_from_frames.py usa para su "Filing firm: ...".
    pending_df = (positives.join(classified.lazy(), on="text_hash", how="anti")
                  .join(doc_firm(), on="accession_number", how="left")
                  .select(*PARAGRAPH_KEY, "text_hash", "duplicate_count",
                          pl.col("ticker").fill_null(""), pl.col("company_name").fill_null("").alias("firm"))
                  .sort(*PARAGRAPH_KEY).collect())
    stats["pending_texts"] = pending_df.height
    stats["pending_instances_covered"] = int(pending_df["duplicate_count"].sum())
    reps = pending_df.to_dicts()
    if limit:
        reps = reps[: int(limit)]

    if not reps:
        return [], stats

    sentences = attach_sentences(reps)

    info_by_key = {tuple(r[c] for c in PARAGRAPH_KEY): r for r in reps}
    rows_by_key: dict[tuple, dict] = {}
    for record in sentences:
        key = tuple(record[c] for c in PARAGRAPH_KEY)
        if key not in rows_by_key:
            info = info_by_key[key]
            rows_by_key[key] = {**{c: record[c] for c in PARAGRAPH_KEY},
                                "text_hash": info["text_hash"],
                                "sentences": [], "sentence_indices": [],
                                "duplicate_count": info["duplicate_count"],
                                "ticker": info.get("ticker", ""), "firm": info.get("firm", "")}
        rows_by_key[key]["sentences"].append(record["sentence_text"])
        rows_by_key[key]["sentence_indices"].append(int(record["sentence_index"]))
    return list(rows_by_key.values()), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Clasificar sólo N párrafos pendientes")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="4-8 recomendado contra OpenRouter/qwen; el spec pedía 32 pero eso "
                             "asume vLLM propio -- vía OpenRouter falla/rate-limita a esa escala")
    parser.add_argument("--part-rows", type=int, default=250,
                        help="Filas (no párrafos) por parte atómica")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en .env")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending, pop_stats = fetch_pending(args.output_dir, args.limit)
    print(f"Modelo: {args.judge_model} | prompt {PROMPT_VERSION}", flush=True)
    print(f"Población positiva del prefiltro: {pop_stats['total_positive_instances']:,} instancias "
          f"({pop_stats['unique_positive_texts']:,} textos únicos)", flush=True)
    print(f"Ya clasificados: {pop_stats['already_classified_texts']:,} textos "
          f"({pop_stats['already_classified_instances']:,} instancias que los comparten)", flush=True)
    print(f"Pendientes en total: {pop_stats['pending_texts']:,} textos únicos "
          f"({pop_stats['pending_instances_covered']:,} instancias que quedarán cubiertas)", flush=True)
    if args.limit and pop_stats["pending_texts"] > len(pending):
        print(f"Se clasifican {len(pending):,} de esos textos en ESTA corrida (--limit)", flush=True)
    if not pending:
        print("Nada pendiente.")
        return

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written, stats = asyncio.run(classify_rows(
        pending, args.output_dir, session_id, args.judge_model,
        args.concurrency, args.part_rows, args.progress_every,
    ))

    manifest = {
        "session_id": session_id, "judge_model": args.judge_model, "prompt_version": PROMPT_VERSION,
        "requested_texts": len(pending), "population_stats": pop_stats,
        "parts_written": [str(p) for p in written],
        **stats, "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    path = args.output_dir / f"ai_classify_manifest__session={session_id}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n{stats['classified']:,} párrafos clasificados ({stats['frames']:,} frames), "
          f"{stats['failed']:,} con error"
          f"{' (INTERRUMPIDO — lo hecho quedó en disco)' if stats['interrupted'] else ''}")
    print(f"Partes: {len(written)} | Manifiesto -> {path}")


if __name__ == "__main__":
    main()
