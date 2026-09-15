"""Actividades de IA divulgadas, a partir de los frames conductuales ya
extraídos por Pass-1 -- Pass-2 del pipeline v2 (spec 2026-09-12). Reemplaza
scripts/deprecated/ai_activities_from_frames.py -- ver data/deprecated/POINTER.json.

Un párrafo entra a Pass-2 si tiene al menos un frame "disparador" (trigger):
subject=firm Y (concepto de adopción, o concepto de gobernanza con
especificidad no vacía). Frames puramente risk_* nunca disparan Pass-2 --
describen un riesgo, no una actividad.

El modelo NO recibe el párrafo entero: recibe una ventana de oraciones
alrededor de las oraciones de evidencia de los frames disparadores (ventana
+-1), más la primera oración del párrafo (suele nombrar el producto o
programa). Los índices que ve son los ORIGINALES, no renumerados -- por eso
pueden ser no contiguos.

Cada actividad cita `frame_id` (el frame que realiza) y hereda temporal/
ai_type de ese frame -- Pass-2 no los vuelve a decidir.

Mismo contrato operativo que ai_classify.py: aditivo, partes atómicas cada
`--part-rows` (default 250), error como fila (se reintenta la corrida
siguiente), cobertura por `text_hash`. En SIGINT/SIGTERM se cancelan las
tareas en vuelo y se escribe de inmediato lo que había en el buffer.

Uso:
    uv run python scripts/enrichment/ai_activities_from_frames.py --limit 20
    uv run python scripts/enrichment/ai_activities_from_frames.py --concurrency 32
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_classify import (ADOPTION_CONCEPTS, DEFAULT_CONCURRENCY,  # noqa: E402
                         DEFAULT_JUDGE_MODEL, GOVERNANCE_CONCEPTS, PARAGRAPH_KEY, REPO_ROOT,
                         attach_sentences, doc_firm)

DEFAULT_DIR = REPO_ROOT / "data" / "interim" / "ai_activities"
PART_GLOB = "ai_activities__session=*.parquet"
PROMPT_VERSION = "v2"
FRAMES_GLOB = str(REPO_ROOT / "data/interim/ai_classify/ai_frames__session=*.parquet")


# --------------------------------------------------------------------------
# Salida estructurada -- spec v2 §2.1
# --------------------------------------------------------------------------

Action = Literal["deploy", "develop", "integrate", "procure", "partner",
                 "invest", "staff", "scale", "measure", "govern", "restrict"]
Source = Literal["own", "third_party", "co_developed", "acquired", "open_source", "mixed", "unspecified"]
EntityRole = Literal["own_brand", "provider", "model", "partner", "acquired", "customer", "channel", "benchmark"]
Target = Literal["employees", "customers", "developers", "internal_process", "partners", "unspecified"]
Stage = Literal["exploring", "piloting", "deployed", "scaled", "unspecified"]
Evidence = Literal["named", "metric", "vendor", "generic"]
MetricUnit = Literal["usd", "pct", "count", "headcount", "hours", "days", "years"]
Measures = Literal["capex", "opex_saving", "revenue", "headcount", "users", "efficiency", "time_saved", "other"]


class Metric(BaseModel):
    raw: str = Field(description="verbatim span from the text, max 6 words")
    value: Optional[float] = None
    value_max: Optional[float] = None  # solo para rangos; value = mínimo
    unit: MetricUnit
    measures: Measures

    @model_validator(mode="after")
    def _range_ordering(self):
        # Never raises: like ai_classify.py's valence fix, a cross-field
        # numeric slip here would fail the whole tool call under
        # reasoning=none, dropping every other correct field (action/object/
        # entities/other metrics) in this activity along with it. Unlike
        # valence's categorical mismatch, a numeric ordering slip has an
        # unambiguous deterministic fix -- no need for a "model_inconsistent"
        # sentinel, just normalize.
        if self.value_max is not None:
            if self.value is None:
                self.value, self.value_max = self.value_max, None
            elif self.value_max < self.value:
                self.value, self.value_max = self.value_max, self.value
        return self


class Entity(BaseModel):
    name: str
    role: EntityRole


class AIActivity(BaseModel):
    frame_id: int
    action: Action
    object: str = Field(description="1-3 lowercase generic words, e.g. 'fraud model', 'copilot'")
    function: str = Field(description="snake_case verb_noun, e.g. 'detect_fraud'; 'unspecified' if absent")
    target: Target
    stage: Stage
    source: Source
    entities: list[Entity] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    evidence_type: Evidence
    sentence_ids: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _dedupe_entities(self):
        # Same fix as ai_classify.py's _dedupe_lists: the model sometimes
        # repeats the same (name, role) pair.
        seen: set[tuple[str, str]] = set()
        deduped = []
        for e in self.entities:
            key = (e.name, e.role)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        self.entities = deduped
        return self


class ParagraphActivities(BaseModel):
    activities: list[AIActivity] = Field(default_factory=list)


SYSTEM_PROMPT = """\
You extract the concrete AI activities a firm discloses, for research on AI \
disclosure. You receive: the filing firm (ticker and name); selected numbered \
sentences from one paragraph (indices are the original ones and may be \
non-contiguous); and the frames already identified in that paragraph, each with \
a frame_id and its supporting sentence indices. Cite sentences by index; never \
reproduce sentence text.

An activity is: the firm performs ACTION on OBJECT for FUNCTION. Produce one \
activity per distinct (action, object, function); merge only exact repetitions. \
Every activity sets frame_id to the frame it realizes and cites at least one \
sentence. Zero activities is valid when the frames are too generic to name an \
object ("AI is important to our strategy").

Rules:
- Report only what the firm itself does, did or plans. Never customers, the \
market or competitors.
- The object must be AI/ML or something the text explicitly calls AI-enabled. \
Generic technology, digital or data investment without AI is not an activity.
- The filing firm, its subsidiaries and its "our X" products are own_brand with \
source=own. Only a clearly different company or its product is provider, \
model or partner.
- object: 1-3 lowercase generic words ("copilot", "fraud model"). Brand names, \
companies and model names go in entities with their role, every one of them.
- function: snake_case verb_noun; "unspecified" if the text gives none. Do not \
invent an object or function.
- stage is the maturity of the asset as stated (exploring, piloting, deployed, \
scaled), not the tense of the sentence. "We plan to deploy" is action=deploy; \
stage is whatever the text states, otherwise unspecified.
- metrics: one entry per number the text attaches to the activity. raw is the \
verbatim span (max 6 words). value is the normalized number: USD as a full \
float ($10M -> 10000000.0), percentages as decimals (15% -> 0.15), counts and \
headcounts as integers. Leave value null for vague quantities ("hundreds", \
"significant"). For a range set value to the minimum and value_max to the \
maximum. Never invent a number.
- evidence_type: named if a product or process is named, metric if a figure is \
given, vendor if a provider is named, generic otherwise. If several apply, \
pick the strongest in the order metric > named > vendor > generic."""


def context_window(n_sentences: int, trigger_frames: list[dict], window: int = 1) -> list[int]:
    keep = {0}  # la primera oración suele nombrar el producto o programa
    for f in trigger_frames:
        for sid in f["sentence_ids"]:
            for off in range(-window, window + 1):
                if 0 <= sid + off < n_sentences:
                    keep.add(sid + off)
    return sorted(s for s in keep if s < n_sentences)


def build_prompt(sentences: list[str], sentence_ids: list[int], frames: list[dict],
                 firm_ticker: str, firm_name: str) -> str:
    numbered = "\n".join(f"[{i}] {sentences[i]}" for i in sentence_ids)
    described = "\n".join(
        f"- frame {f['frame_id']}: concepts={f['concepts']}, temporal={f['temporal']}, "
        f"ai_type={f['ai_type']}, sentences={f['sentence_ids']}" for f in frames)
    return (f"Filing firm: {firm_ticker} ({firm_name})\n"
            f"{numbered}\n---\nFrames:\n{described}")


def validate_activities(extraction: ParagraphActivities, allowed_frame_ids: set[int],
                        allowed_sentence_ids: set[int]) -> list[str]:
    errors = []
    for i, a in enumerate(extraction.activities):
        if a.frame_id not in allowed_frame_ids:
            errors.append(f"activity {i}: frame_id {a.frame_id} not among trigger frames {sorted(allowed_frame_ids)}")
        bad_sids = [s for s in a.sentence_ids if s not in allowed_sentence_ids]
        if bad_sids:
            errors.append(f"activity {i}: sentence_ids {bad_sids} outside the sent window {sorted(allowed_sentence_ids)}")
    return errors


def merge_duplicate_activities(activities: list[AIActivity]) -> list[AIActivity]:
    """Same fix as ai_classify.py's merge_duplicate_frames: the model can split
    one activity into near-duplicate entries instead of respecting the
    prompt's own rule ("one activity per distinct (action, object, function);
    merge only exact repetitions"). Merges when frame_id+action+object+
    function all match -- exactly the identity the prompt already defines as
    "one activity", so no extra evidence-overlap guard is needed here."""
    merged: list[AIActivity] = []
    for act in activities:
        match = next((m for m in merged if m.frame_id == act.frame_id and m.action == act.action
                      and m.object == act.object and m.function == act.function), None)
        if match is None:
            merged.append(act.model_copy(deep=True))
            continue
        match.entities = match.entities + [e for e in act.entities if (e.name, e.role) not in
                                           {(x.name, x.role) for x in match.entities}]
        match.metrics = match.metrics + act.metrics
        match.sentence_ids = sorted(set(match.sentence_ids) | set(act.sentence_ids))
        if match.target == "unspecified" and act.target != "unspecified":
            match.target = act.target
        if match.stage == "unspecified" and act.stage != "unspecified":
            match.stage = act.stage
        if match.source == "unspecified" and act.source != "unspecified":
            match.source = act.source
    return merged


# --------------------------------------------------------------------------
# Partes: verificación y escritura atómica
# --------------------------------------------------------------------------

ENTITY_STRUCT = pa.struct([("name", pa.string()), ("role", pa.string())])
METRIC_STRUCT = pa.struct([("raw", pa.string()), ("value", pa.float64()),
                          ("value_max", pa.float64()), ("unit", pa.string()), ("measures", pa.string())])

ACTIVITY_SCHEMA = pa.schema([
    ("country_code", pa.string()), ("form", pa.string()), ("accession_number", pa.string()),
    ("item_key", pa.string()), ("paragraph_index", pa.int64()), ("text_hash", pa.uint64()),
    ("frame_id", pa.int64()), ("activity_id", pa.int64()), ("has_activity", pa.bool_()),
    ("action", pa.string()), ("object", pa.string()), ("function", pa.string()),
    ("target", pa.string()), ("stage", pa.string()), ("source", pa.string()),
    ("entities", pa.list_(ENTITY_STRUCT)), ("metrics", pa.list_(METRIC_STRUCT)),
    ("evidence_type", pa.string()), ("sentence_ids", pa.list_(pa.int64())),
    ("firm", pa.string()),
    ("judge_model", pa.string()), ("prompt_version", pa.string()), ("session_id", pa.string()),
    ("classified_at", pa.string()), ("error", pa.string()),
])


def activity_parts(directory: Path) -> list[Path]:
    return sorted(directory.glob(PART_GLOB))


def commit_part(rows: list[dict], directory: Path, session_id: str, index: int) -> Path:
    final = directory / f"ai_activities__session={session_id}__part={index:05d}.parquet"
    staging = final.with_suffix(".parquet.partial")
    pq.write_table(pa.Table.from_pylist(rows, schema=ACTIVITY_SCHEMA), staging, compression="zstd")
    staging.replace(final)
    return final


def _base(row: dict, judge_model: str, session_id: str) -> dict:
    return {**{c: row[c] for c in PARAGRAPH_KEY}, "text_hash": row["text_hash"],
            "firm": row.get("firm"), "judge_model": judge_model, "prompt_version": PROMPT_VERSION,
            "session_id": session_id, "classified_at": datetime.now(timezone.utc).isoformat(), "error": None}


def _empty_activity_fields() -> dict:
    return {"frame_id": None, "action": None, "object": None, "function": None, "target": None,
            "stage": None, "source": None, "entities": [], "metrics": [],
            "evidence_type": None, "sentence_ids": []}


def _rows(row: dict, judge_model: str, session_id: str, out: ParagraphActivities) -> list[dict]:
    base = _base(row, judge_model, session_id)
    activities = merge_duplicate_activities(out.activities)
    if not activities:
        return [{**base, "activity_id": 0, "has_activity": False, **_empty_activity_fields()}]
    return [{**base, "activity_id": i, "has_activity": True, "frame_id": a.frame_id,
             "action": a.action, "object": a.object.strip().lower()[:80],
             "function": a.function.strip().lower()[:60], "target": a.target, "stage": a.stage,
             "source": a.source,
             "entities": [{"name": e.name.strip()[:60], "role": e.role} for e in a.entities if e.name and e.name.strip()],
             "metrics": [{"raw": m.raw.strip()[:120], "value": m.value, "value_max": m.value_max,
                         "unit": m.unit, "measures": m.measures} for m in a.metrics],
             "evidence_type": a.evidence_type, "sentence_ids": list(a.sentence_ids)}
            for i, a in enumerate(activities)]


def _error_row(row: dict, judge_model: str, session_id: str, error: Exception) -> dict:
    return {**_base(row, judge_model, session_id), "activity_id": 0, "has_activity": False,
            **_empty_activity_fields(), "error": f"{type(error).__name__}: {error}"[:2000]}


async def run_rows(rows, directory, session_id, judge_model, concurrency, part_rows, progress_every):
    from pydantic_ai import Agent
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider

    class QwenSafeOpenRouterModel(OpenRouterModel):  # ver ai_classify.py: content null en Alibaba
        async def _map_messages(self, messages, model_request_parameters, *, model_settings=None):
            mapped = await super()._map_messages(messages, model_request_parameters, model_settings=model_settings)
            for m in mapped:
                if m.get("role") == "assistant" and m.get("content") is None:
                    m["content"] = ""
            return mapped

    model = QwenSafeOpenRouterModel(judge_model, provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
                                    profile=OpenRouterModelProfile(openrouter_supports_cache_control=True),
                                    settings={"openrouter_cache_instructions": True})
    # docs/judge_model_selection.md -- mismo modelo/proveedor ganador que
    # ai_classify.py, sin validar aparte para esta tarea (pendiente en ese doc).
    model_settings = OpenRouterModelSettings(
        openrouter_provider={"order": ["amazon-bedrock/us-east-1"], "allow_fallbacks": True},
        extra_body={"reasoning": {"effort": "none"}},
    )
    # retries=2: autocorrección estructural nativa de pydantic-ai (schema/Literal/
    # rango value_max>=value) -- capa aparte de, y además de, nuestro propio
    # reintento semántico único de abajo (frame_id/sentence_ids contra la ventana
    # real enviada, algo que un validador de Pydantic sin contexto no puede ver).
    # Si de todas formas falla en las dos capas, no es fatal para la corrida:
    # queda como fila de error (error IS NOT NULL) y `fetch_pending` la reintenta sola la próxima vez.
    agent = Agent(model, output_type=ParagraphActivities, system_prompt=SYSTEM_PROMPT, retries=2,
                 model_settings=model_settings)

    written, buffered = [], []
    done = failed = n_act = 0
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    interrupted = asyncio.Event()

    def flush():
        nonlocal buffered
        if buffered:
            written.append(commit_part(buffered, directory, session_id, len(written))); buffered = []

    async def one(row):
        nonlocal done, failed, n_act
        async with semaphore:
            if interrupted.is_set():
                raise asyncio.CancelledError
            allowed_frame_ids = {f["frame_id"] for f in row["trigger_frames"]}
            allowed_sentence_ids = set(row["sentence_ids_sent"])
            prompt = build_prompt(row["sentences"], row["sentence_ids_sent"], row["trigger_frames"],
                                  row.get("ticker", ""), row.get("firm", ""))
            try:
                result = await agent.run(prompt)
                errors = validate_activities(result.output, allowed_frame_ids, allowed_sentence_ids)
                if errors:
                    retry_prompt = (f"{prompt}\n\n---\nYour previous answer was invalid: "
                                     f"{'; '.join(errors)}. Correct it and answer again.")
                    result = await agent.run(retry_prompt)
                    errors = validate_activities(result.output, allowed_frame_ids, allowed_sentence_ids)
                    if errors:
                        raise ValueError(f"invalid frame_id/sentence_ids after retry: {'; '.join(errors)}")
                out = _rows(row, judge_model, session_id, result.output)
                done += 1; n_act += sum(1 for r in out if r["has_activity"])
                return out
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                failed += 1
                return [_error_row(row, judge_model, session_id, error)]

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, interrupted.set)
        except (NotImplementedError, RuntimeError):
            pass
    tasks = [asyncio.create_task(one(r)) for r in rows]
    try:
        for completed in asyncio.as_completed(tasks):
            try:
                buffered.extend(await completed)
            except asyncio.CancelledError:
                continue
            total = done + failed
            if progress_every and total % progress_every == 0:
                rate = total / max(time.perf_counter() - started, 1e-9); eta = (len(rows) - total) / rate if rate else 0
                print(f"[ai-activities] {total:,}/{len(rows):,} | {done:,} ok ({n_act:,} activities) | {failed:,} error | "
                      f"{rate:.1f} it/s | ETA {int(eta // 60):02d}:{int(eta % 60):02d}", flush=True)
            if len(buffered) >= part_rows:
                flush()
                print(f"  parte escrita ({len(written)} en esta sesión)", flush=True)
    finally:
        # SIGINT/SIGTERM: cancela lo que no alcanzó a correr y escribe de
        # inmediato lo que había en el buffer -- nunca se pierde, nunca queda
        # a medio escribir (commit_part escribe .partial y luego reemplaza).
        for t in tasks:
            t.cancel()
        flush()
    return written, {"classified": done, "failed": failed, "activities": n_act,
                     "interrupted": interrupted.is_set(), "seconds": time.perf_counter() - started}


def fetch_pending(output_dir: Path, limit: int) -> tuple[list[dict], dict]:
    import glob as _glob
    if not _glob.glob(FRAMES_GLOB):
        return [], {"population_texts": 0, "already_done": 0, "pending_texts": 0,
                    "note": f"no Pass-1 output yet at {FRAMES_GLOB} -- run ai_classify.py first"}
    keys = list(PARAGRAPH_KEY)
    parts = activity_parts(output_dir)
    if parts:
        done = (pl.concat([pl.scan_parquet(p) for p in parts], how="diagonal_relaxed")
                .filter(pl.col("error").is_null()).select("text_hash").unique().collect())
    else:
        done = pl.DataFrame(schema={"text_hash": pl.UInt64})

    # Pass-1 se lee directo del glob de partes, igual que ai_classify.py hace
    # con su propia población: la llamada más reciente por texto (session_id,
    # classified_at). §3.3: frame disparador = subject=firm Y (concepto de
    # adopción, O gobernanza con especificidad no vacía).
    all_frames = (pl.concat([pl.scan_parquet(p) for p in sorted(_glob.glob(FRAMES_GLOB))],
                            how="diagonal_relaxed")
                  .filter(pl.col("error").is_null()))
    latest_call = (all_frames.select("text_hash", "session_id", "classified_at").unique()
                   .sort("session_id", "classified_at", descending=True, nulls_last=True)
                   .unique("text_hash", keep="first"))
    latest_frames = all_frames.join(latest_call, on=["text_hash", "session_id", "classified_at"])

    def has_any(column: str, values) -> pl.Expr:
        return pl.col(column).list.eval(pl.element().is_in(sorted(values))).list.any()
    frame_fields = ["frame_id", "concepts", "temporal", "ai_type", "sentence_ids"]
    population = (latest_frames
                  .filter(pl.col("has_frame") & (pl.col("subject") == "firm")
                          & (has_any("concepts", ADOPTION_CONCEPTS)
                             | (has_any("concepts", GOVERNANCE_CONCEPTS)
                                & (pl.col("specificity").list.len() > 0))))
                  .select("text_hash", *keys, *frame_fields).unique()
                  .sort(*keys, "temporal", "ai_type", nulls_last=True)
                  .unique(["text_hash", "frame_id"], keep="first", maintain_order=True)
                  .sort("frame_id")
                  .group_by("text_hash", *keys, maintain_order=True)
                  .agg(pl.struct(frame_fields).alias("trigger_frames"))
                  .sort(*keys)
                  .unique("text_hash", keep="first", maintain_order=True)
                  .collect())
    stats = {"population_texts": population.height,
             "already_done": population.join(done, on="text_hash", how="semi").height}
    pend = (population.lazy().join(done.lazy(), on="text_hash", how="anti")
            .join(doc_firm(), on="accession_number", how="left")
            .with_columns(pl.col("company_name").fill_null("").alias("firm"), pl.col("ticker").fill_null(""))
            .drop("company_name")
            .sort(*keys).collect())
    stats["pending_texts"] = pend.height
    reps = pend.to_dicts()[: int(limit)] if limit else pend.to_dicts()
    if not reps:
        return [], stats
    sent = attach_sentences(reps)

    by_key = {tuple(r[c] for c in PARAGRAPH_KEY): r for r in reps}
    rows: dict[tuple, dict] = {}
    for rec in sent:
        key = tuple(rec[c] for c in PARAGRAPH_KEY)
        if key not in rows:
            info = by_key[key]
            trigger_frames = [{**f, "concepts": list(f["concepts"]), "sentence_ids": [int(x) for x in f["sentence_ids"]],
                               "frame_id": int(f["frame_id"])} for f in info["trigger_frames"]]
            rows[key] = {**{c: rec[c] for c in PARAGRAPH_KEY}, "text_hash": int(info["text_hash"]),
                        "firm": (info.get("firm") or "").strip(), "ticker": (info.get("ticker") or "").strip(),
                        "trigger_frames": trigger_frames, "sentences": []}
        rows[key]["sentences"].append(rec["sentence_text"])
    result = []
    for r in rows.values():
        r["sentence_ids_sent"] = context_window(len(r["sentences"]), r["trigger_frames"], window=1)
        result.append(r)
    return result, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0, help="Tomar N pendientes al azar (semilla 7) en vez de los primeros; para probar el esquema")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="4-8 recomendado contra OpenRouter/qwen; el spec pedía 32 pero eso "
                             "asume vLLM propio -- vía OpenRouter falla/rate-limita a esa escala")
    parser.add_argument("--part-rows", type=int, default=250)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en .env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending, pop = fetch_pending(args.output_dir, 0 if args.sample else args.limit)
    if args.sample:
        import random
        pending = random.Random(7).sample(pending, min(args.sample, len(pending)))
    print(f"Modelo: {args.judge_model} | prompt {PROMPT_VERSION}")
    print(f"Población: {pop['population_texts']:,} textos únicos con frames disparadores | hechos: {pop['already_done']:,} | pendientes: {pop['pending_texts']:,}")
    if args.limit and pop["pending_texts"] > len(pending):
        print(f"En esta corrida: {len(pending):,} (--limit)")
    if not pending:
        print("Nada pendiente."); return
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written, stats = asyncio.run(run_rows(pending, args.output_dir, session_id, args.judge_model,
                                          args.concurrency, args.part_rows, args.progress_every))
    manifest = {"session_id": session_id, "judge_model": args.judge_model, "prompt_version": PROMPT_VERSION,
                "requested_texts": len(pending), "population_stats": pop, "parts_written": [str(p) for p in written],
                **stats, "finished_at": datetime.now(timezone.utc).isoformat()}
    path = args.output_dir / f"ai_activities_manifest__session={session_id}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n{stats['classified']:,} párrafos ({stats['activities']:,} actividades), {stats['failed']:,} con error"
          f"{' (INTERRUMPIDO)' if stats['interrupted'] else ''} | partes: {len(written)} | {path}")


if __name__ == "__main__":
    main()
