# Plan de migración: analytics y thesis.qmd de v1 a v2

**Estado (2026-09-13): EJECUTADO.** Todo lo que `thesis.qmd` lee corre sobre
v2 real, verificado corriendo cada script contra la data del corpus (no solo
leído). Resumen al final del documento (§3).

Documento de planificación, no ejecutado. Generado tras correr el pipeline v2
completo sobre el corpus real (ver `docs/judge_model_selection.md` y el
commit `e2b50b2`). Cubre todo lo que queda apuntando al schema viejo
(`gold_ai_frames` / `data/interim/ai_activities` v1, ahora en
`data/deprecated/`) y necesita reescribirse para usar la salida real de
`scripts/common/ai_classify.py` / `ai_activities_from_frames.py` v2.

**Por qué esto es un documento y no un commit**: toca ~15 scripts de
`scripts/analytics/`, `ui-validator/build_sample.py`, y decenas de puntos en
`thesis_document/thesis.qmd` (prácticamente todo el capítulo empírico). Hay
al menos 4 decisiones de diseño reales (no mecánicas) que no puedo tomar
solo — ver §1. Sin esas decisiones, no tiene sentido tocar código.

## 0. Resumen de lo que cambió (v1 → v2)

**Frames (`gold_ai_frames` / `ai_classify.py`):**

| Campo v1 | Campo v2 | Nota |
|---|---|---|
| `subject`: firm / suppliers_or_partners / customers / competitors_or_industry | `subject`: firm / partners / customers / industry / regulators | valores renombrados + `regulators` nuevo |
| `temporal`: realized / planned / expected / hypothetical (4) | `temporal`: realized / forward / hypothetical (3) | `planned`+`expected` colapsan en `forward` |
| `domain`: internal / customer_facing / unspecified | **eliminado** | sin campo, sin reemplazo directo |
| `concepts`: lista, nombres largos (`risk_cybersecurity`, `gov_board_oversight`, `pilot_or_testing`, `use_stage_unspecified`...) | `concepts`: lista, nombres cortos (`risk_cyber`, `gov_board`, `pilot`...) | renombrados, y **algunos sin equivalente** — ver §1.4 |
| `specificity_business_process`/`_product_or_system`/`_vendor_or_partner`/`_quantified_metric`/`_date_or_timeline` (5 booleanos) | `specificity`: lista de strings (`process`/`product`/`vendor`/`metric`/`timeline`) | booleanos separados → lista |
| `rhetoric_promotional`/`_strategic_importance` (2 booleanos) | `rhetoric`: lista de strings (`promotional`/`strategic`/`hedged`) | booleanos separados → lista, + `hedged` nuevo |
| — (no existía) | `valence`: positive/negative/mixed/not_applicable | campo nuevo |
| `evidence_sentence_ids` + `sentence_indices` (traducción a índice real de BD) | `sentence_ids` (posiciones, sin traducción) | **sin traducción a índice real** — ningún consumidor de analytics la usaba, sin impacto |

**Actividades (`data/interim/ai_activities` / `ai_activities_from_frames.py`):**

| Campo v1 | Campo v2 | Nota |
|---|---|---|
| `action`: incluye `invest_infrastructure`, `buy_or_license`, `hire_or_train`, `measure_outcome`, `govern_or_control`, `pilot_or_explore` | `action`: `invest`, `procure`, `staff`, `measure`, `govern` — **`pilot_or_explore` eliminado, sin reemplazo** | ver §1.2 |
| `ai_source` | `source` | rename de columna, mismos valores |
| `named_entities` con roles `own_product_or_brand`/`external_provider`/`external_model`/`acquired_company`/`distribution_channel`/`competitor_or_reference` | `entities` con roles `own_brand`/`provider`/`model`/`acquired`/`channel`/`benchmark` | rename de columna + valores |
| `evidence_strength`: `named_product_or_process`/`metric`/`vendor`/`generic` | `evidence_type`: `named`/`metric`/`vendor`/`generic` | rename de columna + un valor |
| — (no existía) | `metrics`: lista de {raw,value,value_max,unit,measures} | campo nuevo |
| — (no existía) | `frame_id`: liga a qué frame de pass-1 realiza | campo nuevo (linaje) |

## 1. Decisiones de diseño que necesito ANTES de escribir código

### 1.1 `activity_profiles.py`: ¿mantener nombres de columna de salida?

`activity_profiles.py` es el ÚNICO punto de entrada al `ai_activities` crudo;
todo lo demás (`washing_score.py`, `incremental_signal.py`,
`build_call_beta_panel.py`, y ~15 lugares de `thesis.qmd`) lee SUS parquets
derivados (`firm_activities.parquet`, `firm_year_activities.parquet`, etc.),
no el crudo.

**Decisión (2026-09-12): Opción B** — `activity_profiles.py` renombra sus
columnas de salida a los nombres v2 (`source`, `entities`, `evidence_type`,
manteniendo `action`). Todo lo que lee esos parquets (`washing_score.py`,
`incremental_signal.py`, `build_call_beta_panel.py`, `activity_grounding.py`,
`thesis.qmd`) se actualiza a los nombres nuevos. Mayor superficie de cambio
que la Opción A, pero evita arrastrar nombres v1 indefinidamente sobre datos
que ya son v2.

### 1.2 `pilot_or_explore` (action v1) no tiene equivalente en v2

En v2, "explorando/pilotando" pasó a ser un valor de `stage`
(`exploring`/`piloting`), no de `action` — la acción real siempre es una de
las 11 restantes (deploy/develop/integrate/procure/partner/invest/staff/
scale/measure/govern/restrict). `activity_profiles.py` y el `get_action()`
de `thesis.qmd` (L1307-1341) tienen una rama entera para
`action=="pilot_or_explore"`.

**Decisión (2026-09-12)**: reemplazar esa rama por
`stage.isin(["exploring","piloting"])` sobre CUALQUIER acción, no una acción
separada. Es un cambio de significado, no un rename — antes "pilotar" era
mutuamente excluyente de "desplegar"; ahora un `deploy` en etapa `piloting`
cuenta como pilotaje. Se acepta el solapamiento porque refleja mejor cómo las
firmas describen pilotos en la práctica (probando X mientras despliegan Y).

### 1.3 `domain` (internal/customer_facing) eliminado, sin reemplazo

Usado como dimensión de agrupación explícita en `earnings_calls_analysis.py`
(L133, itera `for column in ("temporal","subject","ai_type","domain")`) y en
`build_strategy_dimensions.py` (leído, sin uso downstream claro detectado).
También es una de las "ocho dimensiones" descritas en `tbl-frame-schema` de
`thesis.qmd` (~L380-476) — con `domain` fuera, la prosa de "eight-dimension
schema" queda inconsistente (son siete).

**Decisión (2026-09-12): opción (b)** — reconstruir `domain` como proxy desde
el `target` de pass-2 (`employees`/`internal_process` ≈ internal;
`customers`/`developers` ≈ customer_facing). Implicaciones a documentar en
`thesis.qmd` cuando se llegue al capítulo empírico:

- `domain` deja de ser un campo de frame (pass-1) puro; pasa a depender de
  que el frame haya disparado al menos una actividad en pass-2. Frames sin
  actividad asociada no tienen `domain` bajo este esquema.
- Cobertura parcial: solo frames con actividad, no todos los frames del
  schema de ocho dimensiones original.
- `earnings_calls_analysis.py` (L133) y `build_strategy_dimensions.py`
  necesitan el join frame↔activity (vía `frame_id`, campo nuevo de v2, ver
  §0) antes de poder derivar `domain`, no solo leer la tabla de frames.

### 1.4 Conceptos sin equivalente 1:1

- `ai_investment`/`ai_infrastructure` (usados en `ai_intensity.py`) → v2 los
  tiene como `investment`/`infrastructure`. Mecánico, solo rename.
- `risk_legal_liability`, `gov_ethical_framework`, `gov_committee`,
  `gov_compliance`, `gov_model_risk` (usados en
  `check_archetype_document_channels.py`) → no tienen equivalente 1:1 en el
  enum `Concept` de v2, pero sí mapean por agrupación a categorías más
  gruesas. **Decisión (2026-09-12)**:

  | v1 | → v2 | razón |
  |---|---|---|
  | `gov_committee` | `gov_board` | comité de gobernanza de IA reporta al board |
  | `gov_ethical_framework` | `gov_policy` | framework ético = política formalizada |
  | `gov_compliance` | `gov_policy` | cumplimiento normativo se instrumenta como política |
  | `gov_model_risk` | `gov_technical` | model risk management es gobernanza técnica del modelo |
  | `risk_legal_liability` | `risk_regulatory` | exposición legal por incumplimiento normativo |

  `check_archetype_document_channels.py` (L65, L80) actualiza su `gov_rate`
  para usar directamente `list_contains(f.concepts, 'gov_board')` etc. — sin
  necesidad de OR sobre nombres viejos, dado que la reclasificación v2 ya
  produce los conceptos nuevos.

## 2. Orden de regeneración (topológico)

```
1. build_duckdb.py
   → crear vista nueva (gold_ai_frames_v2, o reemplazar gold_ai_frames)
     apuntando a data/interim/ai_classify/ con schema v2
        │
        ├─→ 2a. RENAME MECÁNICO (bajo esfuerzo, sin decisiones pendientes):
        │       ai_intensity.py, build_geo_provenance.py,
        │       build_earnings_call_speaker_roles.py, frames_source.py
        │       (decidir si frames_source.py se colapsa, es código duplicado
        │       de la vista de build_duckdb.py)
        │
        ├─→ 2b. REDISEÑO DE LÓGICA (specificity/rhetoric booleanos→listas,
        │       requiere §1.3 resuelto):
        │       channel_gap_analysis.py, build_strategy_dimensions.py,
        │       earnings_calls_analysis.py,
        │       check_archetype_document_channels.py (requiere además §1.4)
        │
        │       build_firm_clusters.py NO se migra (2026-09-13): calcula los
        │       arquetipos de voz A/B/C/D y los clusters de comportamiento
        │       0-3 del viejo Capítulo 4, método que build_strategy_dimensions.py
        │       ya reemplaza y que thesis.qmd ya no cita (usa
        │       firm_strategy_dimensions.parquet). Movido a scripts/deprecated/,
        │       junto con sus 7 consumidores directos que también importaban de
        │       él y no aparecen citados por nombre en thesis.qmd:
        │       behavior_block_eval.py, washing_hierarchical.py,
        │       segments_no_intensity_robustness.py, build_voice_behavior_grid.py,
        │       voice_behavior_factors.py, build_segments.py,
        │       cluster_diagnostics.py.
        │
        │       HALLAZGO real durante esta migración: build_firm_panels.py
        │       (productor de firm_year_master_v2.parquet, sí vivo y citado
        │       en thesis.qmd decenas de veces) leía
        │       firm_year_archetype_behaviors.parquet/voice_x_behavior.parquet
        │       -- los outputs de build_firm_clusters.py. Eso significaba que
        │       la columna `archetype` de firm_year_master_v2.parquet era un
        │       constructo DISTINTO del `archetype` que usa el resto de la
        │       tesis (el k=3 de build_strategy_dimensions.py) bajo el MISMO
        │       nombre -- thesis.qmd ya tenía un comentario defensivo
        │       ("the year-level letter codes in the master panel ... are not
        │       used here") para evitar leer el campo equivocado. Corregido:
        │       build_firm_panels.py ahora lee firm_year_strategy_dimensions.
        │       parquet / firm_strategy_dimensions.parquet directamente: un
        │       solo `archetype`, en todas partes. build_strategy_economic_profiles.py
        │       y report_crosscheck_stats.py no dependían de build_firm_clusters.py
        │       y no necesitaron cambios.
        │
        └─→ 2c. ui-validator/build_sample.py (aislado, no bloquea nada más)

3. activity_profiles.py -- único punto de entrada a ai_activities crudo
   (requiere §1.1 y §1.2 resueltos)
   produce: firm_activities.parquet, firm_activity_profiles.parquet,
            firm_year_activities.parquet, channel_activity_cells.parquet,
            activity_profiles.json
        │
        ├─→ 4a. washing_score.py, incremental_signal.py,
        │       build_call_beta_panel.py -- SIN CAMBIOS si §1.1 = Opción A
        │
        └─→ 4b. activity_grounding.py -- SÍ necesita rename de
                evidence_strength/stage aunque se elija Opción A, porque lee
                el parquet CRUDO además del agregado

5. thesis_document/thesis.qmd -- al final, después de 1-4.
   17 referencias directas a gold_ai_frames, ~15+ referencias directas a
   firm_activities.parquet, y decenas de tablas/figuras cuya fuente pasa por
   los parquets derivados de arriba (no necesitan tocarse si el schema de
   SALIDA de esos parquets no cambia de nombre de columna).
```

## 3. Validación después de migrar (antes de tocar `thesis.qmd`)

No basta con que el código corra sin error — hay que confirmar que los
números tienen sentido frente a v1, dado que la población de párrafos
también cambió (fix de segmentación, +292/-65 en `is_ai_prefiltered`, ver
`docs/judge_model_selection.md`). Antes de re-renderizar la tesis:

1. Correr cada script de `scripts/analytics/` reescrito y comparar conteos
   agregados (n_frames, n_activities, distribución de `subject`/`temporal`)
   contra los números que ya están en `thesis.qmd` hoy (v1) — un cambio de
   ±5-10% es esperable (más corpus + mejor extracción), un cambio de 50%+ en
   cualquier dirección amerita mirar por qué antes de seguir.
2. Guardar el resultado en una rama separada o al menos comparar `git diff`
   de las cifras clave del capítulo empírico antes de dar por buena la
   migración — este documento no incluye ese paso porque depende de que el
   código nuevo ya exista.

## 4. Lo que NO está en el alcance de este plan

- Los números de validación (F1 del golden set, Cohen's κ inter-judge,
  precision/recall out-of-domain) en `thesis_document/thesis.qmd` siguen
  describiendo correctamente la corrida v1 — no se tocan aquí. Quedan
  desactualizados respecto al pipeline v2 hasta que se corra una validación
  nueva (golden set + inter-judge) sobre v2, que es un trabajo aparte, más
  caro, no cubierto por esta migración de analytics.
- Aprovechar los campos NUEVOS de v2 (`valence`, `metrics` estructuradas,
  `frame_id` de linaje, `hedged` en rhetoric) para análisis que v1 no podía
  hacer — eso es una mejora, no una migración de paridad. Se puede evaluar
  después de que la paridad funcional esté lograda.
- El fix de speaker/analista vs. ejecutivo en earnings calls (investigado en
  la sesión anterior, delegado a otra sesión aparte) — es ortogonal a esta
  migración.

## 5. Resumen de ejecución (2026-09-13)

**Regla que terminó guiando todo lo demás**: no alcanza con que un script no
crashee sobre datos v2 — hay que rastrear, desde `thesis.qmd` hacia atrás,
qué construye cada número que la tesis realmente lee. Corriendo `make` o
grepeando solo `thesis.qmd` por nombre de archivo se llega a conclusiones
falsas en ambas direcciones (ver más abajo). El criterio final fue: ¿qué lee
`thesis.qmd`, y de qué construct viene ese dato, hoy?

**Migrados y verificados corriendo contra v2 real** (schema, no solo
lectura): `build_duckdb.py` (`gold_ai_frames`/`gold_ai_activities` con
`domain` proxy), `ai_intensity.py`, `frames_source.py`, `channel_gap_analysis.py`,
`channel_gap_words_robustness.py`, `build_strategy_dimensions.py` (+
`hedging_posture` como 8va dimensión), `earnings_calls_analysis.py`,
`check_archetype_document_channels.py`, `build_firm_panels.py`,
`activity_profiles.py`, `activity_grounding.py`, `incremental_signal.py`,
`washing_score.py`, `firm_year_aggregation_robustness.py`.

**El hallazgo real de esta sesión — un solo `archetype`, no dos**:
`build_firm_panels.py` (productor de `firm_year_master_v2.parquet`, leído
por `thesis.qmd` decenas de veces) leía los outputs de
`build_firm_clusters.py` (el A/B/C/D del viejo Capítulo 4). Eso significaba
que la columna `archetype` de `firm_year_master_v2.parquet` era un
constructo DISTINTO del `archetype` que usa el resto de la tesis (el k=3 de
`build_strategy_dimensions.py`) bajo el MISMO nombre — `thesis.qmd` ya tenía
un comentario defensivo para esquivarlo. Corregido en la fuente:
`build_firm_panels.py` ahora lee `firm_year_strategy_dimensions.parquet` /
`firm_strategy_dimensions.parquet` directamente. El mismo patrón se repitió
y se corrigió en `activity_profiles.py` (componía por "segmento" del viejo
K-means de Capítulo 3, `build_segments.py`; ahora compone por `archetype`) y
en `incremental_signal.py` (el modelo M3 decía "dummies de segmento
(arquetipos)" en su propio comentario — se tomó en serio el paréntesis: M3
ahora arma sus dummies desde `archetype`, no desde `segmento`).

**Deprecados a `scripts/deprecated/`** (verificado que ningún dato que
`thesis.qmd` lee depende de ellos, directa o transitivamente, una vez
corregido lo anterior): `build_firm_clusters.py`, `build_segments.py`,
`build_voice_behavior_grid.py`, `economic_profiles.py`,
`behavior_block_eval.py`, `washing_hierarchical.py`,
`segments_no_intensity_robustness.py`, `voice_behavior_factors.py`,
`cluster_diagnostics.py`. Todos calculaban caracterizaciones de empresa por
clustering propio (voz×comportamiento, K-means de Capítulo 3, o el A/B/C/D
de Capítulo 4) — método ya reemplazado por el archetype de
`build_strategy_dimensions.py` en todo lo que la tesis efectivamente cita.
`activity_grounding.py` perdió en el proceso sus funciones `grid_grounding`
y `washing_grounding` (cruzaban contra la grilla voz×comportamiento y un
score de washing beta-binomial superseded; ninguna de las dos escribía nada
que `thesis.qmd` leyera).

**Dos correcciones de error real encontradas en el camino, no de schema**:
`frames_source.py` colgaba >15 minutos (memory pressure) porque
`current_population` era una `VIEW` recalculada dos veces — pasó a `TABLE`
materializada una vez. `activity_profiles.py` tenía `f["piloting_or_exploring"]
= a["action"] == "pilot_or_explore"` — siempre `False` contra v2 (esa acción
no existe más), silenciosamente. Corregido a `a["stage"].isin(["exploring",
"piloting"])` por la decisión de §1.2.

**Advertencia sobre el Makefile**: `make analytics-text` todavía invoca
`build_firm_clusters.py`/`build_segments.py`/`build_voice_behavior_grid.py`
por nombre de archivo — quedaría roto si alguien corre `make` en vez de los
scripts individuales. No se tocó el Makefile en esta sesión porque el
criterio de qué migrar/deprecar fue "qué lee `thesis.qmd`", no "qué invoca
`make"; el Makefile queda desactualizado como deuda pendiente, documentada
acá para que no sorprenda.
