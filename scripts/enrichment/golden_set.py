"""Golden set: muestreo exploratorio + etiquetado con LLM para evaluar el prefiltro.

El diseño muestral, y sobre todo lo que este conjunto SÍ y NO permite afirmar,
está en docs/golden_set_sampling.md. Resumen de lo que hay que tener presente al
leer este archivo:

- La muestra es deliberadamente NO representativa: sobre-representa los párrafos
  con IA porque a ~0,4% de prevalencia una muestra aleatoria de 10.000 daría unos
  40 positivos. Cada fila guarda `inclusion_weight` para poder reponderar; sin
  eso, cualquier estimación a nivel de corpus queda sesgada.
- Las etiquetas existen para romper la circularidad de evaluar un prefiltro
  léxico+semántico con una etiqueta léxica.
- Etiquetar es caro, así que nada se pierde: partes atómicas cada N filas, los
  errores de API se guardan en la fila en vez de abortar, y una interrupción hace
  flush antes de salir.

Uso:
    python scripts/enrichment/golden_set.py sample          # elige los párrafos
    python scripts/enrichment/golden_set.py label           # los etiqueta (aditivo)
    python scripts/enrichment/golden_set.py label --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
ANALYSIS_PANEL = L.ANALYSIS_PANEL

DEFAULT_DIR = REPO_ROOT / "data" / "interim" / "golden_set"
SAMPLE_PATH = DEFAULT_DIR / "golden_set_sample.parquet"
LABEL_GLOB = "golden_set_labels__session=*.parquet"

PARAGRAPH_KEY = ("country_code", "form", "accession_number", "item_key", "paragraph_index")
DEFAULT_JUDGE_MODEL = "qwen/qwen3.7-flash"
PROMPT_VERSION = "v1"
SAMPLING_VERSION = "v1"

CATEGORIES = ("ai_use", "ai_exploration", "ai_capability", "ai_outcome",
              "ai_risk", "ai_governance", "ai_strategy")

# Etapa 1: sectores donde se espera más presencia de IA. Son nombres de la
# taxonomía de configs/us/config.yaml (12 sectores sobre códigos SIC), resuelta
# por scripts/us/sector_map.py — no una lista aparte, para que retocar sectores
# en un solo lugar valga para todo el pipeline.
TECH_SECTORS = ("software_it", "computing_hardware", "instruments_devices")

# Diccionario CONGELADO para estratificar, deliberadamente separado de
# configs/ai_prefilter.yaml. Retocar los términos del prefiltro no debe cambiar
# qué párrafos componen el conjunto con que se evalúa ese mismo prefiltro.
# El sesgo residual (IA descrita sin estas palabras) lo cubre la etapa 3
# aleatoria, que es el único estrato libre de supuestos.
SAMPLING_KEYWORDS = {
    "strong": ("artificial intelligence", "machine learning", "generative ai", "genai",
               "deep learning", "large language model", "neural network", "chatgpt",
               "openai", "copilot"),
    "weak": ("algorithm", "predictive", "automation", "automated decision",
             "computer vision", "natural language processing"),
}


# --------------------------------------------------------------------------
# Salida estructurada
# --------------------------------------------------------------------------

class ParagraphLabel(BaseModel):
    """Etiqueta de un párrafo. Los campos existen para medir el prefiltro:
    `is_ai_disclosure` es la etiqueta binaria contra la que se calcula la curva
    precisión-recall, `relevance` separa la mención de pasada del disclosure
    sustantivo (que es lo que decide si el umbral debe ser agresivo o
    conservador), `categories` se compara contra `best_semantic_anchor`, y
    `evidence_quote` permite auditar si el juez alucinó."""

    is_ai_disclosure: bool = Field(
        description="True sólo si el párrafo dice algo sustantivo sobre el uso, "
                    "desarrollo, riesgo, gobernanza o estrategia de IA de la empresa.")
    relevance: Literal["none", "incidental", "substantive"] = Field(
        description="'none': no habla de IA. 'incidental': la menciona al pasar, en una "
                    "lista de tecnologías o como término de la industria, sin decir nada "
                    "de la empresa. 'substantive': afirma algo concreto sobre la IA de "
                    "esta empresa.")
    categories: list[Literal["ai_use", "ai_exploration", "ai_capability", "ai_outcome",
                             "ai_risk", "ai_governance", "ai_strategy"]] = Field(
        default_factory=list,
        description="Todas las que apliquen; vacío si is_ai_disclosure es False.")
    evidence_quote: str | None = Field(
        default=None,
        description="Fragmento LITERAL del párrafo que justifica la decisión, copiado "
                    "carácter por carácter. null si is_ai_disclosure es False.")
    mentions_ai_explicitly: bool = Field(
        description="True si el párrafo usa un término explícito de IA (artificial "
                    "intelligence, machine learning, LLM, etc.). Independiente de "
                    "is_ai_disclosure: un párrafo puede describir IA sin nombrarla.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza en la decisión.")
    reasoning: str = Field(description="Una o dos oraciones justificando la decisión.")


SYSTEM_PROMPT = """\
Eres un analista de reportes financieros que construye un conjunto de referencia \
para investigación académica sobre divulgación corporativa de inteligencia \
artificial. Etiquetas párrafos individuales extraídos de formularios 10-K y 10-Q \
presentados ante la SEC (secciones: 1 Business, 1A Risk Factors, 7 y 2 MD&A).

Tu criterio define la verdad contra la que se mide un prefiltro automático, así \
que la consistencia importa más que la generosidad.

QUÉ CUENTA COMO DIVULGACIÓN DE IA (is_ai_disclosure = true):
El párrafo afirma algo concreto sobre inteligencia artificial, machine learning, \
deep learning, modelos de lenguaje, IA generativa, redes neuronales o visión \
computacional EN RELACIÓN A ESTA EMPRESA: que la usa, la desarrolla, invierte en \
ella, obtiene resultados de ella, enfrenta riesgos por ella, la gobierna, o la \
considera estratégica. También cuenta si describe esa sustancia sin usar la \
palabra "IA" — por ejemplo, un modelo predictivo entrenado sobre datos propios \
que toma decisiones automatizadas.

QUÉ NO CUENTA (is_ai_disclosure = false):
- "Algoritmo", "modelo", "automatización" o "predictivo" en sentidos ajenos a la \
IA: modelos de valuación, modelos actuariales, flujos de caja descontados, \
automatización de facturación, algoritmos de ruteo convencionales.
- Menciones de IA que no dicen nada de esta empresa: describir una tendencia de \
la industria, nombrar IA en una lista genérica de tecnologías emergentes, o el \
encabezado de una sección sin contenido.
- Dependencia de proveedores, software o desarrolladores externos sin que la IA \
sea el objeto de esa dependencia.

RELEVANCE:
- "none": no hay IA.
- "incidental": la IA aparece pero el párrafo no afirma nada sustantivo sobre la \
empresa (mención de pasada, enumeración, encabezado).
- "substantive": el párrafo afirma algo concreto sobre la IA de la empresa.
is_ai_disclosure debe ser true si y sólo si relevance es "substantive".

CATEGORÍAS (todas las que apliquen):
- ai_use: usa o despliega IA en operaciones, productos o servicios.
- ai_exploration: explora, evalúa, pilotea o experimenta con IA.
- ai_capability: desarrolla modelos propios, depende de modelos de terceros, o \
invierte en infraestructura, datos o talento de IA.
- ai_outcome: resultados atribuidos a la IA (productividad, costos, ingresos, \
experiencia del cliente).
- ai_risk: riesgos por IA (ciberseguridad, privacidad, regulación, propiedad \
intelectual, sesgo, errores, competencia, desplazamiento laboral).
- ai_governance: políticas, supervisión, controles o revisión humana de la IA.
- ai_strategy: la IA descrita como estratégica o transformadora para la empresa.

EVIDENCE_QUOTE debe ser texto copiado literalmente del párrafo, sin parafrasear. \
Si no puedes citar, is_ai_disclosure es false.

Los párrafos vienen de texto extraído automáticamente: pueden estar truncados, \
ser encabezados sueltos o restos de tablas. Eso no los hace divulgación de IA.\
"""


def build_prompt(row: dict) -> str:
    return (
        f"País: {row['country_code']} | Formulario: {row['form']} | "
        f"Sección: {row['item_key']} | Tipo: {row['content_type']}\n"
        f"---\n{row['paragraph_text']}\n---\n"
        f"Etiqueta este párrafo."
    )


# --------------------------------------------------------------------------
# Muestreo
# --------------------------------------------------------------------------

def _sector_expr(column: str = "sic") -> pl.Expr:
    """Código SIC -> sector usando la MISMA regla que
    scripts/raw_ingestion/us/sector_map.py, en vez de una copia que se
    desincronice: grupo mayor = dos primeros dígitos del SIC rellenado a 4."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "raw_ingestion" / "us"))
    import sector_map
    rules = sector_map.sic_rule_map(sector_map.load_config())
    major = pl.col(column).cast(pl.String).str.pad_start(4, "0").str.slice(0, 2)
    return major.replace_strict(rules, default="sector_unknown", return_dtype=pl.String)


def _keyword_tier_expr(column: str = "paragraph_text") -> pl.Expr:
    """strong / weak / none según el diccionario CONGELADO de este módulo
    (substring literal sobre el texto en minúsculas)."""
    lower = pl.col(column).str.to_lowercase()

    def any_of(terms):
        return pl.any_horizontal([lower.str.contains(t, literal=True) for t in terms]).fill_null(False)
    return (pl.when(any_of(SAMPLING_KEYWORDS["strong"])).then(pl.lit("strong"))
            .when(any_of(SAMPLING_KEYWORDS["weak"])).then(pl.lit("weak"))
            .otherwise(pl.lit("none")))


def _seeded_order(values: pl.Series, salt: str) -> pl.Series:
    """Orden pseudoaleatorio reproducible: BLAKE2b de 8 bytes de `valor|salt`.
    No depende de la versión de ninguna librería."""
    return pl.Series([int.from_bytes(hashlib.blake2b(f"{v}|{salt}".encode(), digest_size=8).digest(),
                                     "big", signed=False) for v in values.to_list()], dtype=pl.UInt64)


def build_sample(
    total: int = 10_000,
    tech_n: int = 4_000,
    variation_n: int = 4_500,
    random_n: int = 1_500,
    tech_sectors: tuple[str, ...] = TECH_SECTORS,
    seed: int = 42,
):
    """Muestreo en tres etapas — docs/golden_set_sampling.md §5.

    NO lee embeddings ni scores del prefiltro: el conjunto con que se evalúa el
    prefiltro no puede estar definido por la salida del prefiltro (§2).

    El orden aleatorio de cada etapa sale de `_seeded_order` sobre el
    `text_hash` del párrafo y la semilla.
    """
    import pandas as pd

    keys = list(PARAGRAPH_KEY)
    # bronze.firm_universe filtered by the analysis-panel membership constant,
    # not silver.firm_universe/filing_manifest*: enrichment reads bronze, never
    # silver (silver is built AFTER enrichment; see docs/tasks/stage_violations_audit.md
    # E-L8 -- no cycle exists today because the silver universe tables don't
    # depend on LLM outputs, but reading them from here is still the wrong
    # direction to depend in).
    panel_firms = (L.scan("bronze.firm_universe")
                   .filter(pl.col("membership_groups").list.contains(ANALYSIS_PANEL)))
    doc = pl.concat([L.scan(name).select("country_code", "accession_number", "cik", "ticker", "filing_date")
                     for name in ("bronze.filing_manifest", "bronze.filing_manifest_10q")]
                    ).join(panel_firms.select("ticker"), on="ticker", how="semi").drop("ticker")
    firms = panel_firms.select("country_code", "cik", "sic")
    base = (L.scan("bronze.paragraphs")
            .select(*keys, "content_type", "paragraph_text", "text_hash")
            .join(doc, on=["country_code", "accession_number"], how="left")
            .join(firms, on=["cik", "country_code"], how="left")
            .select(*keys, "content_type",
                    pl.col("filing_date").cast(pl.Date, strict=False).dt.year().cast(pl.String)
                    .fill_null("unknown").alias("filing_year"),
                    _sector_expr("sic").alias("sector"),
                    _keyword_tier_expr("paragraph_text").alias("keyword_tier"),
                    pl.col("text_hash").alias("content_key")))
    # El boilerplate se repite entre filings: pagar dos veces la misma
    # etiqueta infla el acuerdo además del costo.
    pool = (base.sort(keys)
            .with_columns(pl.len().over("content_key").alias("duplicate_count"))
            .unique("content_key", keep="first", maintain_order=True)
            .collect(engine="streaming"))
    populations = pool.height

    def rank(frame: pl.DataFrame, salt: str) -> pl.DataFrame:
        return frame.with_columns(_seeded_order(frame["content_key"], salt).alias("_order"))

    in_tech = pl.col("sector").is_in(list(tech_sectors))
    # Etapa 1 — densidad: sectores con más presencia esperada de IA,
    # repartido por sección y nivel de palabra clave para que no se
    # concentre todo en Risk Factors.
    stage1 = (rank(pool.filter(in_tech), str(seed))
              .sort("_order")
              .with_columns(pl.int_range(1, pl.len() + 1).over("item_key", "keyword_tier").alias("rn"))
              .filter(pl.col("rn") <= max(1, tech_n // 12))
              .sort("item_key", "keyword_tier", "rn").head(tech_n)
              .drop("_order").to_pandas())
    stage1["stage"] = "stage1_tech_oversample"
    stage1["stratum"] = stage1["sector"] + "|" + stage1["item_key"] + "|" + stage1["keyword_tier"]

    # Etapa 2 — cobertura: round-robin de celda rara a común sobre los 7 ejes.
    axes = ["country_code", "form", "item_key", "content_type", "filing_year", "sector", "keyword_tier"]
    rest = pool.filter(~in_tech)
    cells = rest.group_by(axes).agg(pl.len().alias("cell_size"))
    per_cell = max(1, math.ceil(variation_n / cells.height)) if cells.height else 1
    quota = cells.with_columns(pl.min_horizontal("cell_size", pl.lit(per_cell)).alias("take"))
    stage2 = (rank(rest, str(seed)).join(quota, on=axes)
              .sort("_order")
              .with_columns(pl.int_range(1, pl.len() + 1).over(axes).alias("rn"))
              .filter(pl.col("rn") <= pl.col("take"))
              .sort(["cell_size", *axes, "rn"]).head(variation_n)
              .drop("_order", "take", "rn").to_pandas())
    stage2["stage"] = "stage2_max_variation"
    stage2["stratum"] = stage2[["country_code", "form", "item_key", "content_type",
                                "filing_year", "sector", "keyword_tier"]].agg("|".join, axis=1)

    # Etapa 3 — aleatoria pura: el único estrato sin supuestos, y el que da
    # un estimador insesgado de prevalencia con el que reponderar las otras.
    chosen = set(map(tuple, pd.concat([stage1, stage2])[keys].values))
    stage3 = (rank(pool, f"{seed}|random").sort("_order").head(random_n * 2)
              .drop("_order").with_columns(pl.lit(None, pl.Int64).alias("cell_size")).to_pandas())
    stage3 = stage3[~stage3[keys].apply(tuple, axis=1).isin(chosen)].head(random_n)
    stage3["stage"] = "stage3_random"
    stage3["stratum"] = "random"

    sample = pd.concat([stage1.assign(cell_size=pd.NA), stage2, stage3], ignore_index=True)
    sample = sample.drop(columns=[c for c in ("rn",) if c in sample.columns])

    # Peso de inclusión = población del estrato / muestreados, con la
    # población medida EN EL MARCO DEL QUE ESA ETAPA SORTEÓ. Cada etapa usa
    # una definición de estrato distinta, así que cada una necesita su
    # propio conteo: buscar la población de la etapa 1 en un índice armado
    # con las claves de 7 partes de la etapa 2 no acierta nunca y cae a un
    # fallback, que es como las 3.950 filas del sobremuestreo tech
    # terminaron pesando 36 párrafos de corpus en lugar de 424.051.
    axes1 = ["sector", "item_key", "keyword_tier"]
    axes2 = axes
    pob1 = pool.filter(in_tech).group_by(axes1).agg(pl.len().alias("pob"))
    pob2 = pool.filter(~in_tech).group_by(axes2).agg(pl.len().alias("pob"))
    p1 = {tuple(r[a] for a in axes1): r["pob"] for r in pob1.iter_rows(named=True)}
    p2 = {tuple(r[a] for a in axes2): r["pob"] for r in pob2.iter_rows(named=True)}
    taken = sample.groupby(["stage", "stratum"])["stratum"].transform("size")

    def weight(row, size: int) -> float:
        size = max(int(size), 1)
        if row.stage == "stage3_random":
            return populations / size
        if row.stage == "stage1_tech_oversample":
            return p1.get(tuple(getattr(row, a) for a in axes1), 1) / size
        return p2.get(tuple(getattr(row, a) for a in axes2), 1) / size

    sample["inclusion_weight"] = [
        weight(row, size) for row, size in zip(sample.itertuples(), taken)
    ]
    sample["sampling_version"] = SAMPLING_VERSION
    sample["sampled_at"] = datetime.now(timezone.utc).isoformat()
    return pa.Table.from_pandas(sample.drop(columns=["cell_size", "content_key"]),
                                preserve_index=False)


# --------------------------------------------------------------------------
# Partes: verificación y escritura atómica
# --------------------------------------------------------------------------

LABEL_SCHEMA = pa.schema([
    ("country_code", pa.string()), ("form", pa.string()),
    ("accession_number", pa.string()), ("item_key", pa.string()),
    ("paragraph_index", pa.int64()), ("text_hash", pa.uint64()),
    ("stage", pa.string()), ("stratum", pa.string()),
    ("inclusion_weight", pa.float64()), ("keyword_tier", pa.string()),
    ("sector", pa.string()), ("filing_year", pa.string()),
    ("sampling_version", pa.string()), ("judge_model", pa.string()),
    ("prompt_version", pa.string()), ("session_id", pa.string()),
    ("labeled_at", pa.string()), ("is_ai_disclosure", pa.bool_()),
    ("relevance", pa.string()), ("categories", pa.list_(pa.string())),
    ("evidence_quote", pa.string()), ("evidence_verbatim", pa.bool_()),
    ("mentions_ai_explicitly", pa.bool_()), ("confidence", pa.float64()),
    ("reasoning", pa.string()), ("error", pa.string()),
])


def label_parts(directory: Path) -> list[Path]:
    return sorted(directory.glob(LABEL_GLOB))


def verify_parts(directory: Path, judge_model: str, prompt_version: str) -> dict:
    """Una parte cuenta como trabajo hecho sólo si abre, trae la llave, y viene
    del mismo (modelo, prompt). Otro modelo es otra población de etiquetas: se
    reporta aparte, nunca se borra ni se mezcla."""
    usable, unreadable, mismatched = [], [], []
    for part in label_parts(directory):
        try:
            columns = set(pl.read_parquet_schema(part))
            row = pl.scan_parquet(part).select(
                pl.len(),
                *[(pl.col(c).cast(pl.String).drop_nulls().first() if c in columns else pl.lit(None, pl.String))
                  for c in ("judge_model", "prompt_version")]).collect().row(0)
        except (pl.exceptions.PolarsError, OSError) as error:
            unreadable.append({"path": str(part), "error": str(error).splitlines()[0]})
            continue
        missing = [c for c in PARAGRAPH_KEY if c not in columns]
        if missing or row[1] != judge_model or row[2] != prompt_version:
            mismatched.append({"path": str(part), "rows": row[0], "judge_model": row[1],
                               "prompt_version": row[2], "missing_columns": missing})
            continue
        usable.append(part)
    return {"usable": usable, "unreadable": unreadable, "mismatched": mismatched}


def commit_part(rows: list[dict], directory: Path, session_id: str, index: int) -> Path:
    final = directory / f"golden_set_labels__session={session_id}__part={index:05d}.parquet"
    staging = final.with_suffix(".parquet.partial")
    # Esquema explícito, no inferido. Con inferencia, una parte sin errores tipa
    # `error` como NULL y una con errores como VARCHAR, y después union_by_name no
    # las puede unir ("Unimplemented type for cast VARCHAR -> NULL"). Lo mismo
    # pasa con los campos de etiqueta si una parte entera falló. text_hash va
    # uint64 —8 bytes sin signo se pasan de int64— igual que en ai_embed.py.
    pq.write_table(pa.Table.from_pylist(rows, schema=LABEL_SCHEMA), staging, compression="zstd")
    staging.replace(final)
    return final


# --------------------------------------------------------------------------
# Etiquetado
# --------------------------------------------------------------------------

async def label_rows(
    rows: list[dict],
    directory: Path,
    session_id: str,
    judge_model: str,
    concurrency: int,
    part_rows: int,
    progress_every: int,
    start_index: int,
) -> tuple[list[Path], dict]:
    """Etiqueta `rows`, escribiendo partes cada `part_rows`.

    Nada se pierde: un fallo de API queda registrado en la fila y la corrida
    sigue; una interrupción hace flush de lo acumulado antes de salir.
    """
    from pydantic_ai import Agent
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider

    # OpenRouterModel, not the generic OpenAIChatModel+OpenRouterProvider —
    # only OpenRouterModel knows how to turn pydantic-ai's CachePoint /
    # `openrouter_cache_instructions` setting into a real `cache_control`
    # breakpoint. Even then, pydantic-ai hardcodes
    # `openrouter_supports_cache_control = provider in ('anthropic', 'google')`
    # (see pydantic_ai.providers.openrouter), which silently drops caching
    # for every other vendor — including Alibaba/Qwen, even though
    # OpenRouter's own model metadata for "qwen/qwen3.7-flash" DOES list
    # `input_cache_read`/`input_cache_write` pricing, i.e. Alibaba really
    # supports it. Forcing the flag via a profile override and verifying
    # LIVE (2026-09-04) that it actually engages was necessary — cache
    # support isn't guaranteed just because a model has cache pricing
    # listed. Confirmed: first call `cache_write_tokens=1488`, every repeat
    # call with the same system prompt `cache_read_tokens=1488`.
    #
    # If DEFAULT_JUDGE_MODEL ever changes, re-verify this the same way
    # before assuming the override is still safe — a model OpenRouter/the
    # vendor genuinely doesn't support caching for may just ignore
    # `cache_control` (silently no-op) or may error; check the model's
    # `input_cache_read`/`input_cache_write` fields at
    # https://openrouter.ai/api/v1/models first.
    model = OpenRouterModel(
        judge_model,
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
        profile=OpenRouterModelProfile(openrouter_supports_cache_control=True),
        settings={"openrouter_cache_instructions": True},
    )
    agent = Agent(model, output_type=ParagraphLabel, system_prompt=SYSTEM_PROMPT, retries=2)

    written: list[Path] = []
    buffered: list[dict] = []
    done = failed = 0
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    interrupted = asyncio.Event()

    def flush() -> None:
        nonlocal buffered
        if buffered:
            written.append(commit_part(buffered, directory, session_id, start_index + len(written)))
            buffered = []

    async def label_one(row: dict) -> dict:
        nonlocal done, failed
        record = {key: row[key] for key in PARAGRAPH_KEY}
        record.update({
            # Mismo BLAKE2b de 8 bytes que ai_embed.py, para que la etiqueta se
            # pueda cruzar con el vector y detectar si el texto cambió después.
            "text_hash": int.from_bytes(
                hashlib.blake2b(row["paragraph_text"].encode("utf-8"), digest_size=8).digest(),
                "big", signed=False),
            "stage": row["stage"], "stratum": row["stratum"],
            "inclusion_weight": float(row["inclusion_weight"]),
            "keyword_tier": row["keyword_tier"], "sector": row["sector"],
            "filing_year": row["filing_year"], "sampling_version": row["sampling_version"],
            "judge_model": judge_model, "prompt_version": PROMPT_VERSION,
            "session_id": session_id, "labeled_at": datetime.now(timezone.utc).isoformat(),
        })
        async with semaphore:
            if interrupted.is_set():
                raise asyncio.CancelledError
            try:
                result = await agent.run(build_prompt(row))
                label = result.output
                quote = label.evidence_quote
                record.update({
                    "is_ai_disclosure": label.is_ai_disclosure, "relevance": label.relevance,
                    "categories": list(label.categories), "evidence_quote": quote,
                    # Auditoría de alucinación: la cita tiene que estar en el párrafo.
                    # None cuando no hay cita (los negativos no la llevan), para no
                    # confundir "no aplica" con "el juez la inventó".
                    "evidence_verbatim": (quote.strip() in row["paragraph_text"]) if quote else None,
                    "mentions_ai_explicitly": label.mentions_ai_explicitly,
                    "confidence": float(label.confidence), "reasoning": label.reasoning,
                    "error": None,
                })
                done += 1
            except asyncio.CancelledError:
                raise
            except Exception as error:  # la corrida no se cae por un ítem
                record.update({
                    "is_ai_disclosure": None, "relevance": None, "categories": [],
                    "evidence_quote": None, "evidence_verbatim": None,
                    "mentions_ai_explicitly": None, "confidence": None, "reasoning": None,
                    "error": f"{type(error).__name__}: {str(error).splitlines()[0][:300]}",
                })
                failed += 1
        return record

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, interrupted.set)
        except (NotImplementedError, RuntimeError):
            pass

    tasks = [asyncio.create_task(label_one(row)) for row in rows]
    try:
        for completed in asyncio.as_completed(tasks):
            try:
                buffered.append(await completed)
            except asyncio.CancelledError:
                continue
            total = done + failed
            if progress_every and total % progress_every == 0:
                rate = total / max(time.perf_counter() - started, 1e-9)
                eta = (len(rows) - total) / rate if rate else 0
                print(f"[golden-set] {total:,}/{len(rows):,} | {done:,} ok | {failed:,} error | "
                      f"{rate:.1f} it/s | ETA {int(eta // 60):02d}:{int(eta % 60):02d}", flush=True)
            if len(buffered) >= part_rows:
                flush()
                print(f"  parte escrita ({len(written)} en esta sesión)", flush=True)
    finally:
        for task in tasks:
            task.cancel()
        flush()   # lo que alcanzó a etiquetarse queda en disco pase lo que pase

    return written, {"labeled": done, "failed": failed,
                     "interrupted": interrupted.is_set(),
                     "seconds": time.perf_counter() - started}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_sample(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = build_sample(args.total, args.tech_n, args.variation_n,
                         args.random_n, TECH_SECTORS, args.seed)
    pq.write_table(table, args.sample_path, compression="zstd")
    df = table.to_pandas()
    print(f"Muestra -> {args.sample_path}  ({len(df):,} párrafos)\n")
    print(df.groupby(["stage", "keyword_tier"]).size().to_string())
    print("\npor sector:")
    print(df.groupby("sector").size().sort_values(ascending=False).to_string())
    print("\npor forma/sección/tipo:")
    print(df.groupby(["form", "item_key", "content_type"]).size().to_string())
    print(f"\naños: {sorted(df['filing_year'].unique())}")


def cmd_label(args) -> None:
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Falta OPENROUTER_API_KEY en .env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.sample_path.exists():
        sys.exit(f"No existe {args.sample_path}; corré primero: golden_set.py sample")

    audit = verify_parts(args.output_dir, args.judge_model, PROMPT_VERSION)
    for bad in audit["unreadable"]:
        print(f"  parte ILEGIBLE ignorada: {bad['path']} ({bad['error']})", flush=True)
    for bad in audit["mismatched"]:
        print(f"  parte de otro judge_model/prompt (igual cuenta como cobertura, "
              f"ver [[golden-set-use-all-saved]]): {Path(bad['path']).name} "
              f"(modelo={bad['judge_model']}, prompt={bad['prompt_version']}, filas={bad['rows']})",
              flush=True)

    keys = list(PARAGRAPH_KEY)
    sample = pl.read_parquet(args.sample_path)
    unreadable_paths = {bad["path"] for bad in audit["unreadable"]}
    all_parts = sorted(
        p for p in args.output_dir.glob(LABEL_GLOB) if str(p) not in unreadable_paths
    )
    if all_parts:
        # `error IS NULL`: una fila que falló en la API está escrita en la
        # parte para no perder el intento, pero NO es cobertura — si contara,
        # la corrida siguiente la saltearía y el párrafo quedaría sin etiqueta
        # para siempre. Así un corte por créditos o por red se reintenta solo.
        #
        # TODAS las partes legibles cuentan como cobertura, no solo las que
        # matchean judge_model/prompt_version de esta corrida (eso es lo que
        # `audit["usable"]` filtra, y sirve para OTRA cosa: decidir qué es
        # seguro reusar/comparar dentro de esta config). Un párrafo ya
        # etiquetado por un judge anterior (ej. gemini-3.8-flash antes de
        # migrar a OpenRouter) sigue siendo cobertura real — decisión del
        # proyecto: usar todo lo que se guarde, sin filtrar por tipo/judge
        # (ver memoria golden-set-use-all-saved). Filtrar acá por
        # judge_model haría que cambiar de modelo re-etiquete TODO el
        # golden set de nuevo, gastando de más.
        # `--coverage-judge current` restringe la cobertura a las filas que
        # etiquetó ESTE juez. Es lo que permite RE-etiquetar con un modelo
        # distinto sin borrar nada: las filas del juez viejo siguen en
        # disco (población propia, identificable por `judge_model`) y esta
        # corrida las trata como pendientes.
        #
        # Existe porque el golden set terminó siendo una MEZCLA de dos
        # jueces correlacionada con el estrato — gemini-3.8-flash etiquetó
        # stage1 completo y parte de stage2; qwen3.7-flash el resto de
        # stage2 y TODO stage3_random. Dentro del mismo estrato y el mismo
        # keyword_tier los dos no miden lo mismo (tier fuerte: 100,0% vs
        # 81,8% de positivos), así que el target del prefiltro era una
        # mezcla de dos reglas de decisión, y la que ancla la
        # reponderación al corpus (stage3) era la de un solo juez.
        labeled = pl.concat([pl.scan_parquet(p) for p in all_parts], how="diagonal_relaxed") \
            .filter(pl.col("error").is_null())
        if args.coverage_judge != "any":
            labeled = labeled.filter(pl.col("judge_model") == args.judge_model)
        labeled = labeled.select(keys).unique().collect()
    else:
        labeled = sample.select(keys).clear()
    already = labeled.height
    paragraphs = (L.scan("bronze.paragraphs")
                  .filter(pl.col("accession_number").is_in(sample["accession_number"].unique().implode()))
                  .select(*keys, "paragraph_text"))
    pending_frame = (sample.lazy()
                     .join(labeled.lazy(), on=keys, how="anti", nulls_equal=True)
                     .join(paragraphs, on=keys)
                     .collect())
    # Orden aleatorio sembrado, NO por estrato: si la corrida se corta
    # (créditos, red), lo etiquetado tiene que seguir siendo una
    # submuestra aleatoria del diseño y no los estratos alfabéticamente
    # primeros. Ordenar por estrato dejó stage3_random en cero etiquetas
    # cuando se agotaron los créditos, y con eso la reponderación
    # quedó inservible.
    order = _seeded_order(pl.Series([f"{a}{i}" for a, i in zip(pending_frame["accession_number"],
                                                              pending_frame["paragraph_index"])]),
                          "20260902")
    pending_frame = pending_frame.with_columns(order.alias("_order")).sort("_order", *keys).drop("_order")
    if args.limit:
        pending_frame = pending_frame.head(int(args.limit))
    pending = pending_frame.to_dicts()

    total = con_total = len(pending)
    print(f"Modelo: {args.judge_model} | prompt {PROMPT_VERSION} | "
          f"{len(audit['usable'])} parte(s) previas, {already:,} ya etiquetados", flush=True)
    print(f"Pendientes en esta sesión: {con_total:,}", flush=True)
    if not total:
        print("Nada pendiente.")
        return

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written, stats = asyncio.run(label_rows(
        pending, args.output_dir, session_id, args.judge_model, args.concurrency,
        args.part_rows, args.progress_every, start_index=0))

    manifest = {
        "session_id": session_id, "judge_model": args.judge_model,
        "prompt_version": PROMPT_VERSION, "sampling_version": SAMPLING_VERSION,
        "requested": total, "already_labeled": already,
        "reused_parts": [str(p) for p in audit["usable"]],
        "unreadable_parts": audit["unreadable"], "mismatched_parts": audit["mismatched"],
        "parts_written": [str(p) for p in written],
        **stats, "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    path = args.output_dir / f"golden_set_manifest__session={session_id}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\n{stats['labeled']:,} etiquetados, {stats['failed']:,} con error"
          f"{' (INTERRUMPIDO — lo hecho quedó en disco)' if stats['interrupted'] else ''}")
    print(f"Partes: {len(written)} | Manifiesto -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--sample-path", type=Path, default=SAMPLE_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    sampler = sub.add_parser("sample", help="elegir los párrafos (muestreo en dos etapas)")
    sampler.add_argument("--total", type=int, default=10_000)
    sampler.add_argument("--tech-n", type=int, default=4_000,
                         help="Etapa 1: sobremuestreo de sectores con más IA esperada")
    sampler.add_argument("--variation-n", type=int, default=4_500,
                         help="Etapa 2: máxima variación sobre los 7 ejes")
    sampler.add_argument("--random-n", type=int, default=1_500,
                         help="Etapa 3: aleatoria pura, el único estrato sin supuestos")
    sampler.add_argument("--seed", type=int, default=42)
    sampler.set_defaults(func=cmd_sample)

    labeler = sub.add_parser("label", help="etiquetar con el juez (aditivo)")
    labeler.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    labeler.add_argument("--coverage-judge", choices=("any", "current"), default="any",
                         help="Qué cuenta como ya etiquetado. 'any' (default): cualquier "
                              "juez — no re-etiquetar lo que ya tiene etiqueta de otro "
                              "modelo. 'current': sólo lo que etiquetó --judge-model, de "
                              "modo que un cambio de juez re-etiquete el set completo y "
                              "quede una sola población de etiquetas (las viejas no se "
                              "borran, quedan identificadas por judge_model).")
    labeler.add_argument("--limit", type=int, default=0, help="Etiquetar sólo N pendientes")
    labeler.add_argument("--concurrency", type=int, default=8)
    labeler.add_argument("--part-rows", type=int, default=250,
                         help="Filas por parte; una interrupción pierde a lo sumo esto")
    labeler.add_argument("--progress-every", type=int, default=50)
    labeler.set_defaults(func=cmd_label)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
