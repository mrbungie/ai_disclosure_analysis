# Plan: infraestructura de selección de variante (rule_based / llm_full)

**Estado:** aprobado, pendiente de implementación. Este documento es la referencia
única del diseño — no ejecutar nada, no generar comparaciones entre variantes,
hasta que se implemente lo descrito acá.

## Objetivo

Permitir que dos variantes de clasificación de disclosure de IA coexistan sin
pisarse — **rule_based** (BoW / `--bow-only`, la corrida actual) y **llm_full**
(script 08 con `pydantic_ai` corrido sobre el 100% del corpus, sin fallback a
proxy) — y poder elegir cuál está activa en cada momento, de forma explícita y
visible, tanto en consola como en el dashboard de Streamlit.

No es un mecanismo de comparación automática: es la infraestructura de
selección para cuando se decida correr una variante, la otra, o quedarse solo
con `rule_based`.

## Qué es compartido (no se duplica)

Todo lo de EDGAR y preprocesamiento hasta el chunking es upstream y compartido
entre ambas variantes — se corre una sola vez, independiente de la variante activa:

- `00_build_firm_universe.py` → `firm_universe.parquet`
- `01_build_filing_manifest.py` → `filing_manifest.parquet`
- `02_select_download_batch.py`
- `03_download_selected_filings.py` → `filings_html/`
- `04_extract_sections.py` → `filing_sections.parquet`
- `05_prefilter_ai_mentions.py`
- `06_chunk_candidates.py` → `ai_candidate_chunks.parquet`

Ninguno de estos scripts recibe flag de variante ni duplica output.
`ai_candidate_chunks.parquet` es el punto de entrada único y compartido.

También quedan compartidos, sin flag de variante:

- `07_extract_bow_features.py` → `ai_disclosure_bow_features.parquet` (deterministico, no cambia entre variantes)
- `08_run_llm_classifier.py` → `ai_disclosure_mentions.parquet` (misma Agent/schema `pydantic_ai`; la diferencia entre variantes es **cuánto corriste este script**, no un flag — ver sección de cobertura más abajo)

## Dónde empieza la bifurcación

Sobre el mismo `ai_candidate_chunks.parquet` compartido, dos caminos posibles desde script 09 en adelante:

- **A) `rule_based`** → hoy equivale a script 09 con `--bow-only` (sin tocar esa lógica)
- **B) `llm_full`** → script 09 en una rama nueva: usa `ai_disclosure_mentions.parquet` **sin** fallback a proxy BoW; chunks sin label LLM quedan marcados explícitamente como missing, no se completan silenciosamente con regex

Todo lo que sigue (09 combine → 10 firm-year panel → 11 clustering → 12 event
study → 13 factor analysis → 14 governance sensitivity) depende de la variante
elegida y lee/escribe de forma aislada por variante desde ese punto en adelante.

## 1. Config central: `configs/config.json`

Nueva key top-level:

```json
"variants": {
  "active": "rule_based",
  "choices": ["rule_based", "llm_full"],
  "output_root": "data/processed"
}
```

`active` es la única fuente de verdad, compartida entre consola y dashboard —
el selector de la UI persiste ahí inmediatamente al cambiarlo (no es un filtro
de sesión aislado).

## 2. Módulo nuevo: `scripts/variant_utils.py`

Sin prefijo numérico → importable por los scripts numerados y por `app.py`
(a diferencia de `11_cluster_archetypes.py`, que no se puede importar por
empezar con dígito). Scripts 00–08 no lo importan, no reciben `--variant`
(no aplica).

Contenido:

- `VALID_VARIANTS = ("rule_based", "llm_full")`
- `add_variant_arg(parser)` → agrega `--variant {rule_based,llm_full}`, default `None` ("usar el de config").
- `resolve_variant(cli_value, config) -> str` → prioridad: flag CLI explícito > `config["variants"]["active"]` > error si no hay ninguno. Imprime siempre, antes de cualquier otro output del script:
  ```
  [variant] active variant: rule_based (source: config default)
  [variant] active variant: llm_full (source: --variant flag)
  ```
- `variant_dir(variant, output_root="data/processed") -> Path` → `data/processed/variant_{variant}/`
- `variant_path(variant, filename_stem, ext, subdir=None) -> Path` → `.../variant_{variant}/[subdir/]{stem}__{variant}.{ext}`
- `require_variant(variant, allowed, script_name)` → corta la ejecución (mensaje explicativo + exit) si `variant` no está en `allowed`. Usado por val_01/val_02.

## 3. Estructura de directorios de salida

```
data/processed/
├── variant_rule_based/
│   ├── ai_scored_chunks__rule_based.parquet            (09)
│   ├── firm_year_features__rule_based.parquet          (10)
│   ├── clusters/firm_year_clusters__rule_based.parquet (11)
│   ├── panels/event_study_panel__rule_based.parquet    (12)
│   ├── factors/factor_loadings__rule_based.csv,
│   │        chunk_factor_scores__rule_based.parquet,
│   │        firm_year_factor_scores__rule_based.parquet (13)
│   └── validation/llm_labeled_sample__rule_based.parquet,
│                validation_report__rule_based.csv        (val_01/02 — única variante permitida)
└── variant_llm_full/
    ├── ai_scored_chunks__llm_full.parquet
    ├── coverage_summary__llm_full.txt                    (09, ver sección de cobertura)
    ├── firm_year_features__llm_full.parquet
    ├── clusters/firm_year_clusters__llm_full.parquet
    ├── panels/event_study_panel__llm_full.parquet
    └── factors/...                                       (sin carpeta validation/ — no aplica)
```

Reportes texto de 11/12/14 quedan sufijados en el `reports/` flat existente:
`reports/cluster_threshold_justification__rule_based.txt`,
`reports/did_power_check__llm_full.txt`, etc.

`data/interim/candidate_chunks/` (`ai_candidate_chunks.parquet`,
`ai_disclosure_bow_features.parquet`, `ai_disclosure_mentions.parquet`) no se
toca — sigue compartido, sin sufijo de variante.

## 4. Cambios por script

| Script | Cambio |
|---|---|
| 00–06 | Ninguno. |
| 07 (BoW) | Ninguno. |
| 08 (LLM classifier) | Ninguno en el código. La diferencia entre variantes es cuánto corriste este script (parcial vs. 100% del corpus) — ver cobertura abajo. |
| 09 (combine) | Agrega `--variant`. `rule_based` = comportamiento actual de `--bow-only`, sin tocar esa lógica, solo cambia el path de salida a `variant_rule_based/`. `llm_full` = rama nueva (ver detalle abajo). Error explícito si se pasa `--bow-only` junto con `--variant llm_full` (incompatibles). |
| 10 (panel) | Agrega `--variant`. Lee `ai_scored_chunks__{variant}.parquet` del dir de esa variante, escribe `firm_year_features__{variant}.parquet` ahí. Sin cambios a la lógica de agregación. |
| 11 (clustering) | Agrega `--variant` (además de `--method`, ya existente). Lee/escribe del dir de la variante; reportes de threshold justification/sensitivity (agregados en v2) también sufijados por variante. |
| 12 (event study) | Agrega `--variant`. Lee de `clusters/` y `features/` de esa variante, escribe `panels/event_study_panel__{variant}.parquet` y `did_power_check__{variant}.txt` ahí. Sin cambios de lógica. |
| 13 (factor analysis) | Agrega `--variant`. Nota documentada en el script: corre sobre `ai_disclosure_bow_features.parquet` (compartido), así que los loadings van a ser matemáticamente iguales entre variantes — el flag solo determina de qué `ai_scored_chunks__{variant}.parquet` saca metadata y a qué carpeta escribe. |
| 14 (governance sensitivity) | Agrega `--variant`. Lee `firm_year_features__{variant}.parquet`. Documenta que su comparación BoW-only vs LLM-only D5 es más informativa corrida sobre `llm_full` (donde `share_llm_is_governance` tiene señal real) que sobre `rule_based` (donde queda ~0 porque 09 ignoró el LLM). No genera ningún reporte comparando variantes entre sí — solo lo documenta como caveat de lectura. |
| val_01 (sample+label) | Agrega `--variant`, default = config. Si resuelve a `llm_full` → rechazo explícito vía `require_variant`, con el mensaje metodológico (ver sección de validación). Si es `rule_based`, funciona igual que hoy pero lee/escribe en `variant_rule_based/validation/`. |
| val_02 (validate) | Mismo guard + mismo cambio de paths. |

### Detalle — rama `llm_full` de script 09

- Exige que `ai_disclosure_mentions.parquet` exista y tenga filas (si no, error claro pidiendo correr script 08 primero).
- No hace `COALESCE(is_substantive, substantive_proxy)` ni equivalentes para `is_promotional`, `is_risk_related`, `is_governance_related`, `is_ai_related`: para chunks sin fila en `mentions_df`, el campo queda `NULL` y se agrega columna explícita `llm_label_missing = TRUE` en esa fila.
- **Chequeo de cobertura agregado (no solo por fila):** antes de escribir el output, calcula
  `coverage_pct = (chunks con fila real en mentions_df) / (total chunks) * 100`
  y:
  - lo imprime en consola de forma imposible de pasar por alto:
    `[variant] llm_full coverage: 62.3% — 1,598 / 4,204 chunks missing LLM label`
  - lo escribe a `data/processed/variant_llm_full/coverage_summary__llm_full.txt` (persistente, no solo stdout)
  - lo loggea vía `pipeline_logger` como `WARNING` si `coverage_pct < 95` (umbral configurable), sin bloquear la ejecución — el `llm_label_missing` por fila ya permite decidir después cómo tratar esos chunks, pero el agregado debe ser tan visible como la columna por fila para evitar tratar una corrida parcial como si fuera "full" sin darse cuenta.

## 5. Consola: visibilidad obligatoria

- Cada script variant-aware imprime `[variant] active variant: ...` antes de cualquier otro output, sin forma de correrlo sin verla.
- Cada `pipeline_logger.log_event(...)` posterior en esos scripts incluye `details={"variant": variant, ...}`.

## 6. Dashboard (`app.py`)

- **Sidebar**: nuevo selector (`st.radio` o `st.selectbox`) *"🔀 Active Variant"*, debajo de "📊 Pipeline Status". Al cambiarlo, se llama `save_config(config)` de inmediato (persistencia confirmada) + `st.toast` de confirmación. Visible en todas las tabs, pero solo afecta a las que dependen de la variante:
  - **Sin cambios**: Configuration, Universe Management, Discovery & Manifests, Project Architecture.
  - **Candidate Chunks**: este dataset es compartido (`ai_candidate_chunks.parquet`) — se agrega una nota explícita en la UI aclarando que no filtra por variante, en vez de simular un filtro que no existe.
  - **Post-Pipeline**: pasa a leer/escribir/mostrar rutas bajo `variant_dir(active_variant)`. El toggle existente "Skip LLM classifier (BoW-only mode)" se deriva de la variante activa (bloqueado, no independiente): `rule_based` → forzado ON con tooltip explicando por qué; `llm_full` → forzado OFF, con validación de que `ai_disclosure_mentions.parquet` tenga cobertura antes de habilitar el botón de correr 09. Los botones "Run Step"/"Run All" pasan `--variant {active}` a cada script. **Confirmado: no se agrega ningún trigger automático** — cada step sigue requiriendo click manual, igual que hoy; el selector de variante en el sidebar solo persiste config y cambia qué se muestra, nunca ejecuta nada por sí solo.
  - **Pipeline Logs**: suma un filtro "Variant" (extraído de `details.variant` del JSONL) junto a los filtros existentes de Level/Step.
- Badge fijo `🔀 Variant: **rule_based**` visible en el header de cada tab afectada.

## 7. Validación — guard explícito (solo aplica a `rule_based`)

`require_variant(variant, allowed=("rule_based",), script_name="val_01")` corta
val_01/val_02 si se pasa `--variant llm_full`, con este mensaje:

> val_01/val_02 solo validan la variante rule_based por diseño: usar un
> LLM-judge para validar las mismas categorías que ya clasificó otro LLM
> (llm_full) es circular. Pasá --variant rule_based. llm_full no tiene
> validación de este tipo disponible por diseño, no por omisión.

El mismo texto se agrega a `docs/thesis_plan.md` (Validation Plan) y al README.

## 8. Migración de datos existentes (mover, no recalcular) — último paso

Orden de trabajo:

1. Crear `variant_utils.py` y modificar los 8 scripts + `app.py` para leer/escribir de las rutas nuevas.
2. Smoke check: `py_compile` de los scripts tocados + grep de paths viejos hardcodeados, para confirmar que ningún script ni el dashboard sigue apuntando a `data/interim/candidate_chunks/ai_scored_chunks.parquet` o `data/processed/{features,clusters,panels,validation}` sin sufijo de variante.
3. Recién ahí, como último paso, mover los archivos existentes:

```
data/interim/candidate_chunks/ai_scored_chunks.parquet
  → data/processed/variant_rule_based/ai_scored_chunks__rule_based.parquet

data/processed/validation/llm_labeled_sample.parquet
  → data/processed/variant_rule_based/validation/llm_labeled_sample__rule_based.parquet

data/processed/validation/validation_report.csv
  → data/processed/variant_rule_based/validation/validation_report__rule_based.csv
```

(`ai_candidate_chunks.parquet`, `ai_disclosure_bow_features.parquet`,
`ai_disclosure_mentions.parquet` no se mueven — son compartidos.
`firm_year_features/clusters/event_study_panel` no existen todavía en disco
— no hay nada que mover ahí.)

No se hace la migración antes de que el código ya lea de las rutas nuevas, para
evitar que el dashboard quede apuntando a rutas viejas a mitad de camino.

## 9. Qué NO se hace en esta implementación

- No se corre ningún script.
- No se genera ningún reporte comparativo entre variantes.
- No se toca ni duplica nada de 00–06 ni `ai_candidate_chunks.parquet`.
- No se borra nada — solo se mueven los 3 archivos listados en la sección 8, y solo al final.

## 10. Archivos a tocar

- `configs/config.json` (nueva key `variants`)
- `scripts/variant_utils.py` (nuevo)
- `scripts/09_score_with_llm_booleans.py`, `10_build_features.py`, `11_cluster_archetypes.py`, `12_event_study_panel.py`, `13_factor_analysis.py`, `14_governance_sensitivity.py`, `val_01_sample_and_label.py`, `val_02_validate.py`
- `app.py` (sidebar selector + tabs Post-Pipeline/Logs/Candidate Chunks)
- `docs/thesis_plan.md`, `README.md` (documentar el guard de validación y el flag)
- `Makefile` (targets existentes pasan `--variant` opcional vía `ARGS`, sin cambios estructurales)
- Movimiento de 3 archivos de datos locales (sección 8, último paso)
