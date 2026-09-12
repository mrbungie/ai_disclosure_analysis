"""Actividades de IA divulgadas, a partir de los frames conductuales ya
extraídos: segunda pasada de LLM que convierte "la empresa describe conducta"
en "la empresa hace la acción Y sobre el objeto Z para la función F".

Población: textos únicos de EE.UU. con al menos un frame `subject=firm` que
afirme un concepto conductual (despliegue, piloto, exploración, escalamiento,
capacidad propia, IA de terceros, infraestructura, talento, inversión, o un
resultado). Son ~17.400 textos únicos de los 30.579 clasificados; los frames
de puro riesgo o gobernanza no entran porque no describen una actividad.

No se vuelve a decidir si el párrafo habla de IA ni qué frames tiene: el
modelo recibe las oraciones numeradas MÁS los frames conductuales ya
identificados (conceptos, temporalidad, oraciones de evidencia) y sólo
devuelve, por cada actividad distinta, una estructura chica:

    action      qué hace la empresa (enum: deploy, develop, integrate, ...)
    object      sobre qué (texto corto: "copilot", "fraud model", "data center")
    function    para qué (snake_case normalizado: customer_service, coding, ...)
    target      para quién (enum: employees, customers, developers, ...)
    stage       en qué etapa (enum: exploring, piloting, deployed, scaled)
    ai_source   de dónde sale la IA (enum: own, third_party, co_developed, acquired,
                open_source, mixed, unspecified; `target=customers` ya dice si se vende)
    named_entities  TODAS las entidades nombradas con su rol (own_product_or_brand,
                external_provider, external_model, partner, acquired_company,
                customer, distribution_channel, competitor_or_reference)
    evidence    fuerza de la evidencia (enum: named_product_or_process, metric, vendor, generic)
    sentence_ids

v2 (2026-09-06): el prompt dice QUÉ EMPRESA presenta el documento (ticker y
nombre), para que el modelo distinga proveedor externo de autoreferencia:
`ai_source` dice de dónde sale la IA y `named_entities` lista TODAS las
entidades nombradas con su rol (v1 guardaba un solo proveedor como string,
perdía el segundo en el 2% de las actividades y confundía la marca propia con
un proveedor). Además el prompt exige una actividad por cada acción·objeto·función
distinta del párrafo (v1 pedía "el mínimo" y dejó una sola actividad en el
57% de los párrafos, 43% de los de ocho o más oraciones). Las partes v1 están
archivadas en `data/archive/interim/ai_activities/`.

`object` y `function` quedan libres a propósito: anticipar todos los casos de
uso produce una ontología monstruosa; `scripts/analytics/activity_profiles.py`
normaliza y agrupa las etiquetas después.

Mismo contrato operativo que `ai_classify.py`: aditivo, partes atómicas,
errores como filas (se reintentan en la corrida siguiente), cobertura por
`text_hash`. Salida en `data/interim/ai_activities/`.

Uso:
    uv run --frozen --no-sync python scripts/common/ai_activities_from_frames.py --limit 20
    uv run --frozen --no-sync python scripts/common/ai_activities_from_frames.py --concurrency 20
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
from typing import Literal

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_classify import DEFAULT_DATABASE, DEFAULT_JUDGE_MODEL, PARAGRAPH_KEY, REPO_ROOT  # noqa: E402

DEFAULT_DIR = REPO_ROOT / "data" / "interim" / "ai_activities"
PART_GLOB = "ai_activities__session=*.parquet"
PROMPT_VERSION = "v2"
BEHAVIOURAL_CONCEPTS = ["deployed", "pilot_or_testing", "exploring", "expansion_or_scaling",
                        "proprietary_ai", "third_party_ai", "ai_infrastructure", "ai_talent", "ai_investment",
                        "productivity_outcome", "cost_outcome", "revenue_outcome", "customer_outcome"]

Action = Literal["deploy", "develop", "integrate", "buy_or_license", "partner", "invest_infrastructure",
                 "hire_or_train", "pilot_or_explore", "scale", "measure_outcome", "govern_or_control", "restrict"]
AISource = Literal["own", "third_party", "co_developed", "acquired", "open_source", "mixed", "unspecified"]
EntityRole = Literal["own_product_or_brand", "external_provider", "external_model", "partner", "acquired_company",
                     "customer", "distribution_channel", "competitor_or_reference"]


class NamedEntity(BaseModel):
    name: str = Field(description="The entity as named in the text, 1-4 words (e.g. 'NVIDIA', 'Azure OpenAI', 'Ryzen AI', 'Fermyon').")
    role: EntityRole = Field(description=(
        "own_product_or_brand = the filing firm's own AI product, model or brand, including its subsidiaries and product lines "
        "('our X', 'the X platform' when X is the firm's); external_provider = a vendor whose AI, chips, "
        "cloud or tools the firm uses; external_model = a named third-party model or tool (ChatGPT, Gemini, Llama); partner = "
        "co-development, alliance or joint venture; acquired_company = a company or asset acquired for its AI; customer = a named "
        "customer where the firm's AI is deployed; distribution_channel = a marketplace or cloud through which the firm's AI is "
        "offered; competitor_or_reference = named only as competitor or comparison."))


Target = Literal["employees", "customers", "developers", "internal_process", "partners_or_suppliers", "unspecified"]
Stage = Literal["exploring", "piloting", "deployed", "scaled", "unspecified"]
Evidence = Literal["named_product_or_process", "metric", "vendor", "generic"]


class AIActivity(BaseModel):
    """One concrete disclosed AI activity: the firm performs `action` on `object`
    for `function`."""
    action: Action = Field(description=(
        "What the firm does. deploy = puts AI into use; develop = builds its own model/tool; "
        "integrate = embeds AI (often third-party) into a product or workflow; buy_or_license = "
        "acquires or licenses AI technology or a company; partner = alliance with an AI provider; "
        "invest_infrastructure = data centres, chips, compute, cloud capacity; hire_or_train = "
        "talent, upskilling; pilot_or_explore = testing or evaluating; scale = expanding an existing "
        "use; measure_outcome = reports a result of AI use; govern_or_control = policy, oversight, "
        "controls on AI use; restrict = limits or bans a use of AI."))
    object: str = Field(description=(
        "The thing acted on, in 1-4 words, lower case, generic (e.g. 'copilot', 'fraud model', "
        "'recommendation engine', 'foundation model', 'data center', 'chatbot'). Never a sentence."))
    function: str = Field(description=(
        "The business function served, as a short snake_case label (e.g. customer_service, "
        "software_development, marketing, fraud_detection, supply_chain, drug_discovery, "
        "underwriting, content_creation, search, product_feature, it_operations). Use "
        "'unspecified' if the text gives none."))
    target: Target = Field(description="Who the AI serves: employees, customers, developers, an internal process, partners, or unspecified.")
    stage: Stage = Field(description="Adoption stage as stated: exploring, piloting, deployed, scaled, or unspecified.")
    ai_source: AISource = Field(description=(
        "Where the AI in this activity comes from. own = the filing firm built it, owns it or sells it as its product "
        "('our models', 'we build', a product of the firm or its subsidiaries); third_party = ONLY when the text names or "
        "explicitly refers to an external vendor, tool or model that the firm uses ('third-party AI tools', 'ChatGPT', 'Azure "
        "OpenAI'); co_developed = built with a named partner or customer; acquired = obtained through an acquisition; open_source "
        "= based on open-source models; mixed = own product that embeds third-party models. When the text does not say, use "
        "unspecified, never third_party."))
    named_entities: list[NamedEntity] = Field(default_factory=list, description=(
        "EVERY company, product, model or brand named in connection with this activity, each with its role. Name both when "
        "two are named (e.g. NVIDIA and Intel both as external_provider). The filing firm's own products are "
        "own_product_or_brand, never external. Empty when nothing is named."))
    evidence_strength: Evidence = Field(description=(
        "Strongest evidence attached: named_product_or_process (a named product, system or "
        "process), metric (a number), vendor (a named provider), generic (none of those)."))
    sentence_ids: list[int] = Field(description="Indices of the numbered sentences supporting this activity.")


class ParagraphActivities(BaseModel):
    """Every distinct activity the paragraph discloses. A long paragraph usually
    contains several (e.g. deploying a copilot AND investing in data centres AND
    partnering with a provider): list each one. Zero is valid only when nothing
    concrete is stated (e.g. 'we use AI across our business')."""
    activities: list[AIActivity] = Field(default_factory=list, description=(
        "ALL distinct disclosed AI activities in the paragraph, one entry per distinct (action, object, "
        "function) combination. Every behavioural frame listed in the input should be covered by at least "
        "one activity unless it is too generic to name an object. Do not collapse different actions or "
        "different objects into one entry; do merge exact repetitions of the same activity."))


SYSTEM_PROMPT = """\
You extract the concrete AI ACTIVITIES a company discloses, for academic research \
on AI disclosure. Each paragraph you see has already been classified: it contains \
at least one statement in which the company itself (not customers, not competitors) \
describes adopting, building, buying, scaling, investing in, or obtaining results \
from AI. Those statements are given to you as "behavioural frames" with the \
sentences that support them. Your job is narrower and more concrete: turn each of \
those statements into activities of the form

    the firm performs ACTION on OBJECT for FUNCTION (target, stage, provider, evidence).

You are told which firm files the document (ticker and name): that firm, its subsidiaries and \
its products ('our X') are self-references (own_product_or_brand, ai_source=own), never external \
providers. Only a clearly different company or its tool is external.

The object of every activity must be AI or machine learning, or something the text explicitly \
calls AI-enabled or AI-driven. Generic technology, R&D, digital or data investments with no AI \
stated are NOT activities. Every activity needs at least one supporting sentence index; if no \
sentence supports it, do not create it.

Rules. Be exhaustive: list EVERY distinct activity, one per distinct (action, object, \
function); a paragraph that deploys a product, invests in infrastructure and partners \
with a provider yields three activities, not one. Cover every behavioural frame you are \
given unless it is too generic to name an object. Name EVERY entity the text attaches to an activity, \
with its role (both when two are named); the filing firm's own products are own_product_or_brand; \
say where the AI comes from in ai_source. Report only what the text states the \
FIRM does or did or plans; never what customers, the market or competitors do. Do not invent an object or function the \
text does not give: use 'unspecified'. Prefer generic objects ('copilot', 'chatbot', \
'fraud model') over brand names; put every brand, company or model in named_entities \
with its role. One activity per distinct (action, object, function); \
merge only exact repetitions. Return sentence indices as evidence, never sentence text. Zero \
activities is a valid answer when the statements are too generic to name any \
action-object pair (e.g. 'AI is important to our strategy')."""


def build_prompt(sentences: list[str], frames: list[dict], firm: str = "") -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    header = f"Filing firm: {firm}\n" if firm else ""
    described = "\n".join(
        f"- frame {f['frame_index']}: concepts={f['concepts']}, temporal={f['temporal']}, "
        f"domain={f['domain']}, evidence sentences={f['evidence_sentence_ids']}" for f in frames)
    return (f"{header}{numbered}\n---\nBehavioural frames already identified in this paragraph:\n{described}\n---\n"
            f"List the concrete AI activities the firm discloses.")


ACTIVITY_SCHEMA = pa.schema([
    ("country_code", pa.string()), ("form", pa.string()), ("accession_number", pa.string()),
    ("item_key", pa.string()), ("paragraph_index", pa.int64()), ("text_hash", pa.uint64()),
    ("activity_index", pa.int64()), ("has_activity", pa.bool_()),
    ("action", pa.string()), ("object", pa.string()), ("function", pa.string()), ("target", pa.string()),
    ("stage", pa.string()), ("ai_source", pa.string()),
    ("named_entities", pa.list_(pa.struct([("name", pa.string()), ("role", pa.string())]))),
    ("firm", pa.string()), ("evidence_strength", pa.string()),
    ("evidence_sentence_ids", pa.list_(pa.int64())), ("sentence_indices", pa.list_(pa.int64())),
    ("source_frame_indices", pa.list_(pa.int64())),
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
            "sentence_indices": row["sentence_indices"],
            "source_frame_indices": [f["frame_index"] for f in row["frames"]],
            "firm": row.get("firm"), "judge_model": judge_model, "prompt_version": PROMPT_VERSION, "session_id": session_id,
            "classified_at": datetime.now(timezone.utc).isoformat(), "error": None}


def _rows(row: dict, judge_model: str, session_id: str, out: ParagraphActivities) -> list[dict]:
    base = _base(row, judge_model, session_id)
    empty = {"action": None, "object": None, "function": None, "target": None, "stage": None,
             "ai_source": None, "named_entities": [], "evidence_strength": None, "evidence_sentence_ids": []}
    if not out.activities:
        return [{**base, **empty, "activity_index": 0, "has_activity": False}]
    n = len(row["sentences"])
    return [{**base, "activity_index": i, "has_activity": True, "action": a.action,
             "object": a.object.strip().lower()[:80], "function": a.function.strip().lower()[:60],
             "target": a.target, "stage": a.stage,
             "ai_source": a.ai_source,
             "named_entities": [{"name": e.name.strip()[:60], "role": e.role} for e in a.named_entities if e.name and e.name.strip()],
             "evidence_strength": a.evidence_strength,
             "evidence_sentence_ids": [s for s in a.sentence_ids if 0 <= s < n]}
            for i, a in enumerate(out.activities) if any(0 <= s < n for s in a.sentence_ids)] or \
        [{**base, **empty, "activity_index": 0, "has_activity": False}]


def _error_row(row: dict, judge_model: str, session_id: str, error: Exception) -> dict:
    return {**_base(row, judge_model, session_id), "activity_index": 0, "has_activity": False,
            "action": None, "object": None, "function": None, "target": None, "stage": None,
            "ai_source": None, "named_entities": [], "evidence_strength": None, "evidence_sentence_ids": [],
            "error": f"{type(error).__name__}: {error}"[:2000]}


async def run_rows(rows, directory, session_id, judge_model, concurrency, part_rows, progress_every):
    from pydantic_ai import Agent
    from pydantic_ai.models.openrouter import OpenRouterModel
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
    agent = Agent(model, output_type=ParagraphActivities, system_prompt=SYSTEM_PROMPT, retries=2)

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
            try:
                result = await agent.run(build_prompt(row["sentences"], row["frames"], row.get("firm", "")))
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
    finally:
        for t in tasks:
            t.cancel()
        flush()
    return written, {"classified": done, "failed": failed, "activities": n_act,
                     "interrupted": interrupted.is_set(), "seconds": time.perf_counter() - started}


def fetch_pending(database: Path, output_dir: Path, limit: int) -> tuple[list[dict], dict]:
    con = duckdb.connect(str(database), read_only=True)
    try:
        parts = sorted(str(p) for p in activity_parts(output_dir))
        if parts:
            files = ", ".join(f"'{p}'" for p in parts)
            con.execute(f"CREATE TEMP VIEW done_hashes AS SELECT DISTINCT text_hash FROM read_parquet([{files}], union_by_name=True) WHERE error IS NULL")
        else:
            con.execute("CREATE TEMP VIEW done_hashes AS SELECT CAST(NULL AS UBIGINT) AS text_hash WHERE false")
        # un representante por texto único: la instancia que `gold_ai_frames` ya
        # trae como representante clasificado es cualquiera de sus instancias;
        # se toma la primera por llave para reconstruir las oraciones
        con.execute(f"""
            CREATE TEMP VIEW doc_firm AS
            WITH docs AS (
                SELECT accession_number, ticker FROM filing_manifest WHERE country_code = 'us'
                UNION ALL SELECT accession_number, ticker FROM filing_manifest_10q WHERE country_code = 'us'
                UNION ALL SELECT document_id, ticker FROM read_parquet('{REPO_ROOT / "data/interim/manifests/filing_manifest_earnings_calls.parquet"}')
            ), names AS (SELECT ticker, any_value(company_name) AS company_name FROM firm_universe WHERE country_code = 'us' GROUP BY 1)
            SELECT accession_number, any_value(ticker) AS ticker, any_value(company_name) AS company_name
            FROM docs LEFT JOIN names USING (ticker) GROUP BY 1
        """)
        con.execute(f"""
            CREATE TEMP VIEW population_all AS
            SELECT text_hash, {', '.join(PARAGRAPH_KEY)},
                   list(struct_pack(frame_index := frame_index, concepts := concepts, temporal := temporal,
                                    domain := domain, evidence_sentence_ids := evidence_sentence_ids)) AS frames,
                   any_value(sentence_indices) AS sentence_indices
            FROM (
                SELECT DISTINCT text_hash, {', '.join(PARAGRAPH_KEY)}, frame_index, concepts, temporal, domain,
                       evidence_sentence_ids, sentence_indices
                FROM gold_ai_frames
                WHERE country_code = 'us' AND has_frame AND subject = 'firm'
                  AND list_has_any(concepts, {BEHAVIOURAL_CONCEPTS})
                QUALIFY row_number() OVER (PARTITION BY text_hash, frame_index ORDER BY {', '.join(PARAGRAPH_KEY)}) = 1
            )
            GROUP BY text_hash, {', '.join(PARAGRAPH_KEY)}
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW population AS
            SELECT * FROM population_all
            QUALIFY row_number() OVER (PARTITION BY text_hash ORDER BY {', '.join(PARAGRAPH_KEY)}) = 1
        """)
        stats = {k: int(v) for k, v in con.execute("""
            SELECT (SELECT count(*) FROM population) AS population_texts,
                   (SELECT count(*) FROM population p JOIN done_hashes d USING (text_hash)) AS already_done""").df().iloc[0].items()}
        pend = con.execute(f"""
            SELECT p.*, coalesce(f.company_name, '') || CASE WHEN f.ticker IS NOT NULL THEN ' (' || f.ticker || ')' ELSE '' END AS firm
            FROM population p LEFT JOIN doc_firm f USING (accession_number)
            WHERE NOT EXISTS (SELECT 1 FROM done_hashes d WHERE d.text_hash = p.text_hash)
            ORDER BY {', '.join(PARAGRAPH_KEY)}""").df()
        stats["pending_texts"] = len(pend)
        reps = pend.to_dict("records")[: int(limit)] if limit else pend.to_dict("records")
        if not reps:
            return [], stats
        con.register("reps", pd.DataFrame(reps)[list(PARAGRAPH_KEY)])
        sent = con.execute(f"""
            SELECT r.{', r.'.join(PARAGRAPH_KEY)}, s.sentence_index, s.sentence_text
            FROM reps r JOIN sentences s USING ({', '.join(PARAGRAPH_KEY)})
            ORDER BY {', '.join(f'r.{c}' for c in PARAGRAPH_KEY)}, s.sentence_index""").df()
    finally:
        con.close()
    by_key = {tuple(r[c] for c in PARAGRAPH_KEY): r for r in reps}
    rows: dict[tuple, dict] = {}
    for rec in sent.to_dict("records"):
        key = tuple(rec[c] for c in PARAGRAPH_KEY)
        if key not in rows:
            info = by_key[key]
            frames = [{**f, "concepts": list(f["concepts"]), "evidence_sentence_ids": [int(x) for x in f["evidence_sentence_ids"]],
                       "frame_index": int(f["frame_index"])} for f in info["frames"]]
            rows[key] = {**{c: rec[c] for c in PARAGRAPH_KEY}, "text_hash": int(info["text_hash"]), "firm": (info.get("firm") or "").strip(),
                         "frames": frames, "sentences": [], "sentence_indices": []}
        rows[key]["sentences"].append(rec["sentence_text"]); rows[key]["sentence_indices"].append(int(rec["sentence_index"]))
    return list(rows.values()), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0, help="Tomar N pendientes al azar (semilla 7) en vez de los primeros; para probar el esquema")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--part-rows", type=int, default=250)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en .env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending, pop = fetch_pending(args.database, args.output_dir, 0 if args.sample else args.limit)
    if args.sample:
        import random
        pending = random.Random(7).sample(pending, min(args.sample, len(pending)))
    print(f"Modelo: {args.judge_model} | prompt {PROMPT_VERSION}")
    print(f"Población: {pop['population_texts']:,} textos únicos con frames conductuales | hechos: {pop['already_done']:,} | pendientes: {pop['pending_texts']:,}")
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
