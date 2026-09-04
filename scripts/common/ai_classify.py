"""Semantic frame extraction over AI-disclosure paragraphs — see
docs/classification_model.md for the full design (schema, worked examples,
negation rule, downstream aggregation plan). This script is the extraction
step only: paragraph text in, zero-or-more `AIFrame`s out.

Population classified: paragraphs `scripts/common/ai_prefilter_classify.py`'s
final logistic-regression model marked `is_ai_prefiltered=True` — see
docs/prefilter_evaluation.md §8.2 for the funnel (12,840 of 3,281,038
paragraphs, 0.39%, threshold chosen via GroupKFold CV, not eyeballed). This
replaced an earlier version of this script that used the golden set's
`is_ai_disclosure=True` labels as a stand-in population, back when no
prefilter-scored candidate set existed yet in this environment — that
subset (1,719 paragraphs) is a subset of / mostly overlaps with the real
12,840, so its results aren't discarded, just superseded as the source of
truth for "what's pending."

Same operational contract as golden_set.py, deliberately kept close so the
two don't drift: additive (nothing is ever overwritten or deleted), atomic
part files every `--part-rows` rows, a failed API call is written as an
error row (not skipped) so a credit/network cutoff retries itself next
run, and "already classified" is judge-model-agnostic (any prior
successful run counts as coverage — see golden_set.py's cmd_label for why:
switching judge models must never silently reprocess everything).

Grounding: each paragraph is split into its constituent `sentences` (the
materialized DuckDB table scripts/common/build_duckdb.py already builds)
and sent to the model as `[0] ... [1] ...`, per docs/classification_model.md
§1 — the model returns sentence indices as evidence, never reproduced text,
so there's nothing to hallucinate-and-not-match the way a copied quote
could.

Usage:
    uv run python scripts/common/ai_classify.py --limit 50   # smoke test
    uv run python scripts/common/ai_classify.py              # everything pending
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = REPO_ROOT / "duckdb" / "thesis.duckdb"
DEFAULT_DIR = REPO_ROOT / "data" / "interim" / "ai_classify"
FRAME_GLOB = "ai_frames__session=*.parquet"

PARAGRAPH_KEY = ("country_code", "form", "accession_number", "item_key", "paragraph_index")
DEFAULT_JUDGE_MODEL = "qwen/qwen3.7-flash"
PROMPT_VERSION = "v1"


# --------------------------------------------------------------------------
# Salida estructurada — docs/classification_model.md
# --------------------------------------------------------------------------

Subject = Literal["firm", "suppliers_or_partners", "customers", "competitors_or_industry"]
AIType = Literal["generative", "predictive_ml", "unspecified"]
Temporal = Literal["realized", "planned", "expected", "hypothetical"]
Domain = Literal["internal", "customer_facing", "unspecified"]

Concept = Literal[
    "deployed", "pilot_or_testing", "exploring", "use_stage_unspecified", "expansion_or_scaling",
    "proprietary_ai", "third_party_ai", "ai_infrastructure", "ai_talent", "ai_investment",
    "productivity_outcome", "cost_outcome", "revenue_outcome", "customer_outcome",
    "risk_cybersecurity", "risk_privacy", "risk_regulatory_or_legal", "risk_intellectual_property",
    "risk_bias_or_fairness", "risk_reliability_or_accuracy", "risk_competitive_or_disruption",
    "risk_workforce", "risk_operational_dependence",
    "gov_board_oversight", "gov_management_oversight", "gov_ai_policy_or_framework",
    "gov_technical_controls", "gov_human_oversight", "gov_vendor_governance",
]


class SpecificityEvidence(BaseModel):
    business_process: bool = Field(description=(
        "True when a concrete business process, organizational function, workflow, "
        "or operational activity is identified."))
    product_or_system: bool = Field(description=(
        "True when a specific AI-enabled product, model, system, platform, tool, "
        "or application is identified."))
    vendor_or_partner: bool = Field(description=(
        "True when a specific vendor, technology provider, supplier, partner, or "
        "external AI provider is identified."))
    quantified_metric: bool = Field(description=(
        "True when the frame contains an explicit numerical metric, percentage, "
        "monetary amount, quantity, or measured result."))
    date_or_timeline: bool = Field(description=(
        "True when the frame includes an explicit date, period, deadline, "
        "milestone, implementation schedule, or timeline."))


class RhetoricalSignals(BaseModel):
    promotional: bool = Field(description=(
        "True when the proposition uses strongly positive, transformational, "
        "superiority-oriented, leadership-oriented, revolutionary, or similarly "
        "promotional language about AI."))
    strategic_importance: bool = Field(description=(
        "True when the proposition explicitly frames AI as central, critical, "
        "material, strategically important, or a strategic priority."))


class FrameEvidence(BaseModel):
    sentence_ids: list[int] = Field(
        min_length=1,
        description="Indices of all numbered sentences necessary to support this frame.")


class AIFrame(BaseModel):
    """One semantically coherent proposition about AI. Multiple concepts can
    belong to the same frame — do NOT create one frame per concept. Create
    separate frames only when keeping the information together would destroy
    an important semantic relationship (see docs/classification_model.md §3
    for the full rule and worked examples, including negation §3.1)."""

    subject: Subject = Field(description=(
        "Whose AI activity, capability, outcome, risk, or governance arrangement is "
        "being described. 'customers' is the firm's own customers/users adopting or "
        "reacting to AI. 'competitors_or_industry' is adoption, capability, or "
        "pressure attributed to competitors or the industry at large — kept separate "
        "from 'customers' because customer adoption and competitive pressure are "
        "distinct phenomena (market-pull vs. competitive-push)."))
    ai_type: AIType = Field(description=(
        "'generative' includes generative AI, LLMs, and foundation-model uses. "
        "'predictive_ml' refers to traditional predictive or discriminative "
        "machine-learning applications. 'unspecified' when the disclosure refers "
        "only to AI generally."))
    temporal: Temporal = Field(description=(
        "'realized': already occurred, currently exists, or is ongoing. 'planned': "
        "concrete intention, commitment, announced action, or implementation plan. "
        "'expected': management expectation, forecast, target, or anticipated "
        "outcome. 'hypothetical': may, might, could, potentially, or another "
        "possibility without a concrete commitment."))
    domain: Domain = Field(description=(
        "'internal': employees, internal operations, internal decision-making, or "
        "internal business processes. 'customer_facing': products, services, "
        "interfaces, customer interactions, or other externally facing "
        "applications. 'unspecified': the application domain is unclear, absent, "
        "or not informative."))
    concepts: list[Concept] = Field(
        min_length=1,
        description="All thesis concepts asserted within this same semantic frame.")
    specificity: SpecificityEvidence = Field(description=(
        "Observable evidence indicating how concretely this proposition is "
        "grounded in the disclosure."))
    rhetoric: RhetoricalSignals = Field(description=(
        "Rhetorical characteristics associated specifically with this frame."))
    evidence: FrameEvidence = Field(description=(
        "References to the numbered input sentences supporting this frame."))


class ParagraphExtraction(BaseModel):
    """Structured extraction from one paragraph, pre-segmented into numbered
    sentences. Zero frames is valid when the paragraph does not contain
    information relevant to the ontology, INCLUDING when its only AI-related
    content is a negated proposition (docs/classification_model.md §3.1) —
    do not create a frame to represent something the text says did NOT
    happen."""

    frames: list[AIFrame] = Field(
        default_factory=list,
        description=("All distinct semantic AI frames supported by the paragraph. "
                      "Use the minimum number of frames necessary to preserve the "
                      "relevant semantic pairings. A negated claim (e.g. 'we do not "
                      "use AI') produces NO frame."))


SYSTEM_PROMPT = """\
You extract structured semantic frames about AI from paragraphs of corporate \
disclosures (10-K/10-Q/Memoria Anual), for academic research on AI disclosure. \
Every paragraph you see was already judged to contain substantive AI-related \
content, but your job is finer-grained than that binary judgment: identify every \
distinct, semantically coherent proposition ("frame") about AI in the paragraph.

Core rule: create the MINIMUM number of frames necessary to preserve the \
relationships between subject, AI type, temporal status, domain, concepts, and \
evidence. Do not split one proposition into several frames just because it \
asserts several concepts at once (e.g. "we use generative AI, cutting costs and \
improving service" is ONE frame with multiple concepts). Split into separate \
frames only when merging would incorrectly pair information across distinct \
propositions (e.g. one AI type/temporal pairing for a realized use, a different \
pairing for an expected future use).

Negation rule: a negated proposition ("we do not currently use generative AI") is \
NOT positive evidence of adoption, capability, outcome, risk, or governance, even \
though it names those same concepts. Never create a frame that would assert the \
opposite of what the text says. A paragraph whose only AI-relevant content is \
negated produces zero frames for that content.

Subject attribution rule: subject=firm with temporal=realized/planned/expected means \
the FIRM ITSELF is the one using, building, or committing to AI. A mention of \
customer, market, or competitor demand for AI-related products or applications is NOT \
evidence of the firm's own AI adoption, even when that demand drives the firm's \
revenue. For example, "net sales increased due to growth in AI-related applications" \
describes market/customer demand for the firm's (non-AI) products — this is \
subject=customers or subject=competitors_or_industry with concepts like \
revenue_outcome, never subject=firm with ai_type=generative/predictive_ml implying \
the firm itself deployed AI. Ask: is the FIRM the one doing the AI activity, or is \
someone else (a customer, the market, a competitor) doing it and merely affecting \
the firm's results? Only the former gets subject=firm.

Input is the paragraph's sentences, numbered:
[0] First sentence.
[1] Second sentence.
Return sentence indices as evidence — never reproduce the sentence text.

Zero frames is a valid, common answer when, on closer reading, nothing in the \
paragraph actually fits the ontology (this can happen even though the paragraph \
passed an earlier coarse AI-relevance check)."""


def build_prompt(sentences: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return f"{numbered}\n---\nExtract the semantic AI frames for this paragraph."


# --------------------------------------------------------------------------
# Partes: verificación y escritura atómica — mismo contrato que golden_set.py
# --------------------------------------------------------------------------

FRAME_SCHEMA = pa.schema([
    ("country_code", pa.string()), ("form", pa.string()),
    ("accession_number", pa.string()), ("item_key", pa.string()),
    ("paragraph_index", pa.int64()), ("text_hash", pa.uint64()),
    ("frame_index", pa.int64()), ("has_frame", pa.bool_()),
    ("subject", pa.string()), ("ai_type", pa.string()),
    ("temporal", pa.string()), ("domain", pa.string()),
    ("concepts", pa.list_(pa.string())),
    ("specificity_business_process", pa.bool_()),
    ("specificity_product_or_system", pa.bool_()),
    ("specificity_vendor_or_partner", pa.bool_()),
    ("specificity_quantified_metric", pa.bool_()),
    ("specificity_date_or_timeline", pa.bool_()),
    ("rhetoric_promotional", pa.bool_()), ("rhetoric_strategic_importance", pa.bool_()),
    ("evidence_sentence_ids", pa.list_(pa.int64())),
    # Los índices reales de la tabla `sentences` (DuckDB), en el MISMO orden
    # en que se numeraron en el prompt ([0]=sentence_indices[0], etc.) —
    # necesario porque `sentence_index` en esa tabla no siempre empieza en 0
    # (verificado: arranca en 1 en varias filings reales), así que
    # `evidence_sentence_ids` (posiciones dentro del prompt) NO son
    # directamente el `sentence_index` real. sentence_indices[i] traduce la
    # posición i del prompt al sentence_index verdadero, sin tener que
    # re-derivar el mismo orden más tarde.
    ("sentence_indices", pa.list_(pa.int64())),
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
        "text_hash": int.from_bytes(
            hashlib.blake2b(row["paragraph_text"].encode("utf-8"), digest_size=8).digest(),
            "big", signed=False),
        "sentence_indices": list(row["sentence_indices"]),
        "judge_model": judge_model, "prompt_version": PROMPT_VERSION,
        "session_id": session_id, "classified_at": datetime.now(timezone.utc).isoformat(),
    }


def _frame_rows(row: dict, judge_model: str, session_id: str, extraction: ParagraphExtraction) -> list[dict]:
    base = _base_record(row, judge_model, session_id)
    if not extraction.frames:
        return [{
            **base, "frame_index": None, "has_frame": False,
            "subject": None, "ai_type": None, "temporal": None, "domain": None,
            "concepts": [], "specificity_business_process": None, "specificity_product_or_system": None,
            "specificity_vendor_or_partner": None, "specificity_quantified_metric": None,
            "specificity_date_or_timeline": None, "rhetoric_promotional": None,
            "rhetoric_strategic_importance": None, "evidence_sentence_ids": [], "error": None,
        }]
    rows = []
    for i, frame in enumerate(extraction.frames):
        rows.append({
            **base, "frame_index": i, "has_frame": True,
            "subject": frame.subject, "ai_type": frame.ai_type,
            "temporal": frame.temporal, "domain": frame.domain,
            "concepts": list(frame.concepts),
            "specificity_business_process": frame.specificity.business_process,
            "specificity_product_or_system": frame.specificity.product_or_system,
            "specificity_vendor_or_partner": frame.specificity.vendor_or_partner,
            "specificity_quantified_metric": frame.specificity.quantified_metric,
            "specificity_date_or_timeline": frame.specificity.date_or_timeline,
            "rhetoric_promotional": frame.rhetoric.promotional,
            "rhetoric_strategic_importance": frame.rhetoric.strategic_importance,
            "evidence_sentence_ids": list(frame.evidence.sentence_ids),
            "error": None,
        })
    return rows


def _error_row(row: dict, judge_model: str, session_id: str, error: Exception) -> dict:
    base = _base_record(row, judge_model, session_id)
    return {
        **base, "frame_index": None, "has_frame": None,
        "subject": None, "ai_type": None, "temporal": None, "domain": None, "concepts": [],
        "specificity_business_process": None, "specificity_product_or_system": None,
        "specificity_vendor_or_partner": None, "specificity_quantified_metric": None,
        "specificity_date_or_timeline": None, "rhetoric_promotional": None,
        "rhetoric_strategic_importance": None, "evidence_sentence_ids": [],
        "error": f"{type(error).__name__}: {str(error).splitlines()[0][:300]}",
    }


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
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider

    # Qwen-via-Alibaba workaround for a real, closed-as-not-planned pydantic-ai
    # bug (github.com/pydantic/pydantic-ai/issues/5287, verified against this
    # project's own runs 2026-09-04: same paragraph, same settings, ~20%
    # non-deterministic 400 "The content field is a required field", always
    # and only when a prior assistant turn had only tool calls and no text).
    # OpenAI's spec allows `content: null` on a tool-call-only assistant
    # message and every other provider accepts it — Alibaba's endpoint alone
    # requires a non-null string. OpenRouter forwards the body verbatim, so
    # nothing short of coercing the field ourselves fixes it; this is the
    # exact subclass workaround pydantic-ai's own maintainer posted on that
    # issue before closing it.
    class QwenSafeOpenRouterModel(OpenRouterModel):
        async def _map_messages(self, messages, model_request_parameters, *, model_settings=None):
            mapped = await super()._map_messages(
                messages, model_request_parameters, model_settings=model_settings)
            for m in mapped:
                if m.get("role") == "assistant" and m.get("content") is None:
                    m["content"] = ""
            return mapped

    # Same OpenRouter caching setup as golden_set.py's label_rows() — see
    # that function's comment for why the profile override is necessary
    # (pydantic-ai only auto-enables cache_control for Anthropic/Google,
    # not Alibaba/Qwen, even though OpenRouter itself supports it for this
    # model). SYSTEM_PROMPT here is longer than golden_set's, so caching
    # matters more, not less.
    model = QwenSafeOpenRouterModel(
        judge_model,
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
        profile=OpenRouterModelProfile(openrouter_supports_cache_control=True),
        settings={"openrouter_cache_instructions": True},
    )
    # retries=2 (pydantic-ai's structured-output self-correction) is back on
    # now that the crash it used to trigger is fixed above — measured
    # (2026-09-04) that retries=0 alone traded the ~18% crash rate for a ~21%
    # "Exceeded maximum output retries (0)" rate instead (no chance to
    # self-correct a first-pass schema miss), which is worse, not better.
    agent = Agent(model, output_type=ParagraphExtraction, system_prompt=SYSTEM_PROMPT, retries=2)

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
            try:
                result = await agent.run(build_prompt(row["sentences"]))
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
        for task in tasks:
            task.cancel()
        flush()

    return written, {"classified": done, "failed": failed, "frames": n_frames,
                     "interrupted": interrupted.is_set(), "seconds": time.perf_counter() - started}


# --------------------------------------------------------------------------
# Población pendiente
# --------------------------------------------------------------------------

def fetch_pending(database: Path, output_dir: Path, limit: int) -> tuple[list[dict], int]:
    con = duckdb.connect(str(database), read_only=True)
    try:
        existing_parts = sorted(str(p) for p in frame_parts(output_dir))
        if existing_parts:
            files = ", ".join(f"'{p}'" for p in existing_parts)
            con.execute(f"CREATE OR REPLACE TEMP VIEW classified AS SELECT DISTINCT "
                        f"{', '.join(PARAGRAPH_KEY)} FROM read_parquet([{files}], union_by_name=True) "
                        f"WHERE error IS NULL")
        else:
            con.execute(f"CREATE OR REPLACE TEMP VIEW classified AS SELECT "
                        f"{', '.join(PARAGRAPH_KEY)} FROM paragraphs WHERE false")
        already = con.execute("SELECT count(*) FROM classified").fetchone()[0]

        # Población: el modelo final del prefiltro (scripts/common/
        # ai_prefilter_classify.py) marcó is_ai_prefiltered=True — ver
        # docs/prefilter_evaluation.md §8.2 (12,840/3,281,038, threshold
        # elegido por GroupKFold CV). QUALIFY se queda con la corrida más
        # reciente si alguna vez hay más de una (mismo patrón que
        # extraction_trace en build_duckdb.py).
        con.execute("""
            CREATE OR REPLACE TEMP VIEW positives AS
            SELECT country_code, form, accession_number, item_key, paragraph_index
            FROM read_parquet('data/interim/prefilter_predictions/prefilter_predictions__run=*.parquet',
                               union_by_name=True)
            WHERE is_ai_prefiltered = true
            QUALIFY row_number() OVER (
                PARTITION BY country_code, form, accession_number, item_key, paragraph_index
                ORDER BY model_version DESC
            ) = 1
        """)
        keys = " AND ".join(f"c.{c} IS NOT DISTINCT FROM p.{c}" for c in PARAGRAPH_KEY)
        pending_df = con.execute(f"""
            SELECT p.*, s.sentence_index, s.sentence_text
            FROM positives p
            JOIN paragraphs par USING ({', '.join(PARAGRAPH_KEY)})
            JOIN sentences s USING ({', '.join(PARAGRAPH_KEY)})
            WHERE NOT EXISTS (SELECT 1 FROM classified c WHERE {keys})
            ORDER BY hash({' || '.join(f'p.{c}' for c in PARAGRAPH_KEY)}), s.sentence_index
            {f'LIMIT {int(limit) * 200}' if limit else ''}
        """).df()
    finally:
        con.close()

    # Reconstruye párrafos completos desde las filas de sentences (una fila
    # por oración) — igual que build_prompt necesita, en orden.
    rows_by_key: dict[tuple, dict] = {}
    for record in pending_df.to_dict("records"):
        key = tuple(record[c] for c in PARAGRAPH_KEY)
        if key not in rows_by_key:
            rows_by_key[key] = {**{c: record[c] for c in PARAGRAPH_KEY},
                                "paragraph_text": "", "sentences": [], "sentence_indices": []}
        rows_by_key[key]["sentences"].append(record["sentence_text"])
        rows_by_key[key]["sentence_indices"].append(int(record["sentence_index"]))
    rows = list(rows_by_key.values())
    for r in rows:
        r["paragraph_text"] = " ".join(r["sentences"])
    if limit:
        rows = rows[:int(limit)]
    return rows, already


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Clasificar sólo N párrafos pendientes")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--part-rows", type=int, default=250,
                        help="Filas (no párrafos) por parte atómica")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en .env")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending, already = fetch_pending(args.database, args.output_dir, args.limit)
    print(f"Modelo: {args.judge_model} | prompt {PROMPT_VERSION} | "
          f"{already:,} ya clasificados", flush=True)
    print(f"Pendientes en esta sesión: {len(pending):,}", flush=True)
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
        "requested": len(pending), "already_classified": already,
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
