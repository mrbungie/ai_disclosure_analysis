# Pipeline de texto: de párrafo crudo a candidato de IA

Qué pasa entre que un filing/earnings call entra al corpus y que un párrafo
queda marcado como candidato para el juez LLM. Cubre solo las etapas
determinísticas (embeddings + prefiltro) — el paso final (`ai_classify.py`,
el juez LLM) queda fuera de alcance de este documento porque tiene costo
real y corre bajo decisión explícita, no automática.

## Las cinco etapas

```
fetch/extract          ->  paragraphs, unique_paragraphs  (build_duckdb.py)
        |
        v
1. ai_embed.py          ->  UN vector por párrafo único (texto completo)
        |
        v
2. ai_prefilter.py       ->  score léxico + semántico por párrafo
        |
        v
3. ai_prefilter_sentences.py -> UN vector por ORACIÓN que menciona IA
        |
        v
4. ai_prefilter_deploy.py / ai_prefilter_apply_frozen.py
                          ->  modelo (gradient boosting, 39 señales) ->
                              is_ai_prefiltered = True/False
        |
        v
5. ai_classify.py (LLM, fuera de alcance acá) -> AIFrames
```

## Etapa 1 — `ai_embed.py`: un vector por párrafo

Cada párrafo ÚNICO (deduplicado por `text_hash` en `unique_paragraphs`, no
por instancia — el mismo texto de boilerplate legal repetido en 50 filings
se embebe una sola vez) recibe un vector de 1024 dimensiones con BGE-M3,
en fp16. Responde una sola pregunta: **¿de qué habla este párrafo, en
general?**

- Escala: ~8M párrafos únicos en el corpus completo (US+CL+IT); ~5.6M ya
  embebidos al 2026-09-10.
- Resumible por `text_hash`: un párrafo nuevo que resulta ser texto
  idéntico a uno ya embebido no se reembebe — el índice liviano
  (`paragraph_embeddings_index.parquet`) lo reconoce.
- Corre SOLO en GPU en la nube — nunca local (ver
  `embeddings-run-in-cloud` en memoria de sesión). ~35 min por cada 3.3M
  párrafos en una RTX 5090.

## Etapa 2 — `ai_prefilter.py`: score por párrafo

Con el vector de la etapa 1, calcula qué tan cerca está cada párrafo de un
conjunto de anchors positivos (frases ejemplo de divulgación real de IA) vs.
términos léxicos fuertes/débiles (`configs/ai_prefilter.yaml`). Esto es
barato — un matmul de 8M×1024 contra 1024×20 corre en segundos, no minutos,
por eso está separado de la etapa 1 (los vectores se calculan una vez, se
reusan cada vez que se afina un anchor).

## Etapa 3 — `ai_prefilter_sentences.py`: un vector por oración con término de IA

**El límite estructural de la etapa 1:** en una matriz de habilidades del
directorio de 14,350 caracteres, "inteligencia artificial" es una celda
entre cientos — el vector del párrafo completo promedia sobre todo ese
ruido y sale pareciéndose a "tabla de habilidades genérica", no a
"divulgación de IA". Medido: de 66,972 textos que pasan la compuerta léxica
hay 226,139 oraciones, y solo 30,792 contienen un término de IA — el 86%
de lo que el embedding del párrafo promedia es ruido para esta decisión
específica.

Esta etapa NO reembebe el párrafo. Extrae solo las oraciones que
literalmente contienen un término de IA y les saca su propio vector, con
el mismo modelo y los mismos anchors que la etapa 1/2. Agrega por
`text_hash`: máximo y promedio del margen semántico entre las oraciones de
IA del texto, máximo por categoría, y cuántas oraciones de IA tiene.

**Por qué no es "hacer lo mismo dos veces":** son conjuntos de texto de
tamaño totalmente distinto — 8M párrafos completos (etapa 1) vs. ~182K
oraciones que ya pasaron el filtro léxico (etapa 3, ~3% del volumen de la
etapa 1) — respondiendo preguntas distintas ("¿de qué habla el párrafo?"
vs. "¿de qué habla específicamente la parte que menciona IA?"). Las
columnas de la etapa 3 entran al modelo final como señales ADICIONALES,
nunca reemplazan a las de la etapa 1.

**Limitación conocida, sin arreglar a propósito (2026-09-10):** a
diferencia de `ai_embed.py`, `load_sentences()` en este script no filtra
por lo ya calculado — cada corrida reembebe las 182K oraciones completas,
no solo las nuevas. Es plata de GPU tirada en cada corrida (~7 min), no
solo la primera vez. Queda anotado para agregarle el mismo resume-por-
`text_hash` que ya tiene `ai_embed.py`, pendiente de decisión explícita
antes de tocarlo.

## Etapa 4 — modelo congelado: 39 señales -> `is_ai_prefiltered`

`ai_prefilter_deploy.py` entrena (via `nested_cv()`) un gradient boosting
sobre las labels doradas + las señales de las etapas 1-3, combinadas por
`sentence_join()` (`LEFT JOIN` de los scores de oración por `text_hash`).
El modelo resultante se congela (`docs/FREEZE.md`) en una fecha fija —
`ai_prefilter_apply_frozen.py` es la variante que aplica ese modelo YA
congelado a texto nuevo (embeddings + scores de las etapas 1-3 recién
calculados) SIN reentrenar, para que agregar texto nuevo al corpus no
cambie silenciosamente la superficie de decisión del modelo.

Salida: `data/interim/prefilter_predictions_unique/` — un `is_ai_prefiltered`
por `text_hash`, la población que después pasa (o no) al juez LLM.

## Etapa 5 — `ai_classify.py` (fuera de alcance de este documento)

El juez LLM extrae `AIFrame`s sobre la población `is_ai_prefiltered=True`.
Tiene costo real por llamada — no corre automático, requiere decisión
explícita separada de las etapas 1-4. Ver `docs/classification_model.md`
para el diseño completo de esa etapa.

## Ver también

- `docs/prefilter_evaluation.md` — evaluación y métricas del modelo de
  prefiltro (§8.8 el funnel de deduplicación, §9 el problema de "contexto
  en el embedding" que motivó la etapa 3).
- `docs/FREEZE.md` — qué significa que un modelo esté congelado y cuándo
  se puede reentrenar.
- `docs/analytics/temporal_alignment_rules.md` /
  `docs/analytics/analysis_spines.md` — las reglas de alineamiento
  temporal y los spines de análisis para todo lo que va DESPUÉS de que un
  párrafo tiene sus AIFrames (regresiones, paneles).
