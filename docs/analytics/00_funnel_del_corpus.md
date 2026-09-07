# Funnel del corpus: de documentos a frames

EE.UU., prefiltro vigente (v2, árboles, umbral 0,17, `run=20260906T160624Z`),
clasificación completa, y segunda pasada de actividades (`09`) completa. Todo sale de `duckdb/thesis.duckdb` (`paragraphs`,
`unique_paragraphs`, `gold_ai_frames`) y del parquet de predicciones del
prefiltro; la consulta está al final.

Es el funnel de **inferencia**: por dónde pasa cada texto del corpus hasta
convertirse en frame y en actividad. Cada fila dice qué filtro la produce:
**determinístico** (una regla sobre los datos: largo, hash, fecha, un campo de
una etiqueta ya existente) o **modelo** (prefiltro de árboles, juez LLM). Los
textos que se etiquetaron para entrenar o evaluar esos modelos NO están en el
funnel: van en la tabla de la sección siguiente. Cuatro bloques, de arriba
hacia abajo; la **unidad** cambia entre bloques y está dicha en cada fila:
documentos, párrafos como instancias, textos únicos (un mismo párrafo repetido
en varios documentos cuenta una vez), empresas.

**A. Corpus** (lo que hay) — todo determinístico

| etapa | filtro | unidad | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| documentos | universo S&P 500 × EDGAR / transcripciones | documentos | 2.873 | 8.099 | 2.646 | 34.416 | 7.947 | 55.981 |
| párrafos | segmentación del HTML / turnos de la call | instancias | 1.544.496 | 1.736.542 | 3.679.553 | 1.623.906 | 554.281 | 9.138.778 |
| párrafos puntuables | regla: >3 caracteres y algún alfanumérico | instancias | 1.422.746 | 1.593.351 | 3.254.803 | 1.529.705 | 553.942 | 8.354.547 |
| textos únicos puntuables | hash del texto (BLAKE2b) | textos únicos | 855.682 | 849.415 | 2.273.716 | 406.756 | 483.235 | 4.868.804 |

**B. Prefiltro** (¿menciona IA?) — modelo

| etapa | filtro | unidad | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| textos únicos positivos | **modelo**: gradient boosting sobre 39 señales (léxico, similitud a anchors, forma del texto, oraciones), umbral 0,17 | textos únicos | 11.293 | 2.933 | 6.383 | 384 | 9.584 | 30.577 |
| % de los textos únicos puntuables | | | 1,3% | 0,3% | 0,3% | 0,1% | 2,0% | 0,6% |
| instancias positivas | determinístico: los mismos textos, en todos sus documentos | instancias | 12.487 | 3.928 | 6.536 | 405 | 9.593 | 32.949 |

**C. Frames** (¿qué afirma sobre IA?) — modelo

| etapa | filtro | unidad | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| textos únicos clasificados | determinístico: todos los positivos de B, una llamada cada uno | textos únicos | 11.294 | 2.933 | 6.384 | 384 | 9.584 | 30.579 |
| textos únicos con ≥1 frame | **modelo**: juez LLM `qwen3.7-flash`, prompt v1, cero frames = descarte | textos únicos | 10.532 | 2.587 | 5.216 | 232 | 7.902 | 26.469 |
| % de los clasificados (el resto son falsos positivos de B) | | | 93% | 88% | 82% | 60% | 82% | 87% |
| frames únicos | **modelo**: el juez decide cuántos frames tiene el texto | frames | 16.245 | 3.409 | 7.168 | 295 | 16.249 | 43.366 |
| frames en documentos | determinístico: join por hash a todas las instancias | instancias | 17.758 | 4.590 | 7.284 | 313 | 16.270 | 46.215 |
| documentos con ≥1 frame | determinístico | documentos | 1.681 | 1.431 | 1.090 | 206 | 2.709 | 7.117 |
| % de los documentos del formulario | | | 59% | 18% | 41% | 0,6% | 34% | 13% |

**D. Actividades** (¿qué dice que hace?) — regla sobre etiquetas del juez, luego modelo

| etapa | filtro | unidad | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| textos únicos con frame conductual de la empresa | determinístico sobre las etiquetas de C: `subject=firm` y algún concepto de adopción, capacidad, inversión, talento o resultado | textos únicos | 5.927 | 1.829 | 2.923 | 134 | 6.803 | 17.420 |
| % de los textos con frame | | | 56% | 71% | 56% | 58% | 86% | 66% |
| actividades únicas | **modelo**: segunda pasada del mismo juez, prompt v2, con la empresa emisora en el prompt; regla posterior: sin oración de evidencia no hay actividad | actividades | 8.556 | 2.408 | 2.670 | 163 | 14.020 | 27.566 |
| actividades por texto conductual | | | 1,4 | 1,3 | 0,9 | 1,2 | 2,1 | 1,6 |
| actividades en documentos | determinístico: join por hash | instancias | 9.224 | 3.238 | 2.708 | 165 | 14.033 | 29.368 |
| documentos con ≥1 actividad | determinístico | documentos | 1.298 | 961 | 586 | 88 | 2.444 | 5.377 |
| empresas con ≥1 actividad | determinístico: ticker del documento | empresas | 409 | 159 | 269 | 55 | 383 | 471 (de 510) |

Todo lo que viene después de D (intensidades por 1.000 párrafos, segmentos,
grilla, regresiones, brecha entre canales, score de washing) es
determinístico sobre estas etiquetas: ningún script de `scripts/analytics/`
llama a un modelo.

Las columnas por formulario de textos únicos suman más que el total porque un
texto puede aparecer en dos formularios y se cuenta en cada uno. El 27.583 de
`09` cuenta cada actividad una vez por empresa (17 textos están en dos
empresas); el 27.566 de acá la cuenta una vez.

## Textos etiquetados para entrenar y evaluar los modelos

Fuera del funnel. Son muestras del mismo corpus, etiquetadas por un juez LLM
(no por humanos) para ajustar el prefiltro, medir su error y medir el acuerdo
entre jueces. Los frames y las actividades no tienen entrenamiento: son
extracción zero-shot con esquema, y su único chequeo pendiente es la
validación humana.

| conjunto | formularios | textos únicos | positivos (juez) | etiqueta | uso | resultado |
|---|---|---:|---:|---|---|---|
| golden set 10-K/10-Q | 10-K 6.991 · 10-Q 2.909 | 9.900 | 1.852 | `qwen3.7-flash` (todos); `gemini-3.8-flash` en 6.038 | **entrenar** el prefiltro (GroupKFold por filing, pesos por estrato) y elegir el umbral; estratos: sobremuestreo tech 3.950, máxima variación 4.450, aleatorio 1.500 (validación) | F1 ponderado 0,96 en 10-K; precisión 0,98 en 10-K/10-Q |
| acuerdo entre jueces | 10-K/10-Q | 6.038 pareados | — | gemini vs qwen | medir si el juez es reproducible | κ 0,87 (menciona IA), 0,86 (divulgación), 0,81 (relevancia en 3 niveles) |
| ajuste DEF 14A / 8-K | DEF 14A 1.999 · 8-K 394 | 2.393 | 408 | qwen | **entrenar** v2 fuera del dominio 10-K; disjunto de la validación | — |
| validación DEF 14A / 8-K | DEF 14A 1.000 · 8-K 500 | 1.500 | 381 | qwen | **holdout**: nunca entra al ajuste ni al umbral; estratos término fuerte / débil / ninguno | F1 0,82 (DEF 14A), 0,65 (8-K); precisión 0,73, recall 0,92 ponderados |
| validación earnings calls | calls | 800 | 376 (menciona IA) | qwen | **holdout**: el prefiltro no se entrenó ni ajustó con calls; estratos término fuerte 400 / débil 200 / ninguno 200 (`prefilter_form_validation.py`, etiquetas en `golden_set_forms/calls/`) | precisión 0,59, recall 0,98, F1 0,73 ponderados al corpus; por estrato: fuerte 0,85 / débil 0,58 |
| anchors y léxico | — | 17 anchors positivos, 10 negativos; 18 términos fuertes, 12 débiles, ~60 entidades | — | escritos a mano (`configs/ai_prefilter.yaml`) | señales del prefiltro (similitud coseno, compuerta léxica) | — |
| esquema de actividades (desarrollo) | mixto | 30 párrafos al azar × 3 versiones del prompt | — | juez, revisados a mano en sesión | diseñar `ai_source` y los roles de entidad; no entran a ningún número | — |
| **validación humana** (pendiente) | todos | 300 párrafos con frames (434 frames) · 300 párrafos del prefiltro por estrato · 120 párrafos con actividades (267 actividades) | — | **humano** (`ui-validator/`) | κ humano–juez en promocional y temporal; precisión/recall del prefiltro reponderados, incluidas calls; existencia y precisión por campo de las actividades | pendiente |

Lo que esta tabla deja claro: **todas las etiquetas que sostienen los
análisis son de un LLM**, el único acuerdo medido es entre dos LLM, y el
prefiltro mide distinto por canal: precisión 0,98 en 10-K/10-Q, 0,73 en
proxy/8-K y 0,59 en calls, con recall alto en todos. El juez de frames filtra
después esos falsos positivos (devuelve cero frames), pero la comparación
entre canales arrastra error distinto. Por eso la muestra humana estratifica
por formulario.

El corpus completo (con Italia y Chile) tiene 10.216.373 párrafos y 5.616.337
textos únicos puntuables; sólo EE.UU. tiene embeddings, scoring y frames.

## Cómo leerlo

- **Deduplicación.** 8,35 millones de instancias puntuables son 4,87 millones
  de textos únicos. Embeddings, scoring del prefiltro y clasificación del juez
  ocurren una vez por texto único (`prefilter_evaluation.md` §8.8). Cobertura
  completa: 30.579 clasificados sobre 30.577 positivos (la diferencia son
  textos que cambiaron de umbral entre corridas y ya estaban clasificados).
- **El juez descarta el 13% de lo que el prefiltro marca**: 26.469 de 30.579
  textos positivos reciben algún frame. Los otros 4.110 son falsos positivos
  del prefiltro que la extracción filtra devolviendo cero frames.
- **Expansión a instancias.** `gold_ai_frames` une los frames de cada texto
  único a TODOS los documentos donde aparece ese texto (`FROM paragraphs p JOIN
  latest_frames f ON text_hash`), así que 43.366 frames únicos son 46.215 en
  documentos: factor 1,066. Sólo 1.726 textos positivos aparecen en dos o más
  documentos. Los párrafos de IA se reescriben; el boilerplate repetido pesa
  poco.
- **Únicos vs. instancias, dónde va cada uno.** Costo y cobertura se miden en
  únicos. Todo lo que es intensidad por documento (`ai_intensity.py`: frames
  por 1.000 párrafos, con cero cuando el documento no habla de IA) usa
  instancias en numerador y denominador, del mismo documento. El score de
  washing de `09` cuenta cada texto una vez por empresa (`--unit unique`) para
  no inflar la evidencia con repeticiones dentro de la misma empresa.
- El total del prefiltro sin repetir por formulario es 30.280 textos únicos
  (30.577 sumando formularios).
- **Actividades.** De los 26.469 textos con frame, 17.420 tienen al menos un
  frame conductual de la propia empresa (despliegue, piloto, capacidad,
  infraestructura, talento, inversión o resultado) y son los que pasan a la
  segunda extracción (`ai_activities_from_frames.py`). Salen 27.566
  actividades únicas (1,6 por texto) que son 29.368 en documentos. Las calls
  aportan la mitad de las actividades (14.020) con el 39% de los textos
  conductuales: son el canal denso también en conducta. Los textos
  conductuales por formulario suman más que 17.420 porque un texto puede
  aparecer en dos formularios.
- **8-K**: 34.416 documentos, 206 con algún frame. Ruido casi puro.
- **Calls**: 7.947 documentos con la décima parte de los párrafos de un 10-K
  y 16.270 frames, casi tantos como el 10-K. Es el canal denso
  (`06_brecha_entre_canales.md`).

## Consulta

```sql
WITH docs AS (
  SELECT form, count(DISTINCT accession_number) documentos, count(*) parrafos,
         count(*) FILTER (WHERE is_scorable) puntuables,
         count(DISTINCT text_hash) FILTER (WHERE is_scorable) textos_unicos
  FROM paragraphs WHERE country_code='us' GROUP BY 1),
pref AS (
  SELECT p.form,
         count(*) FILTER (WHERE u.is_ai_prefiltered) instancias_positivas,
         count(DISTINCT p.text_hash) FILTER (WHERE u.is_ai_prefiltered) unicos_positivos
  FROM paragraphs p
  LEFT JOIN read_parquet('data/interim/prefilter_predictions_unique/prefilter_predictions__run=20260906T160624Z.parquet') u
         ON u.text_hash = p.text_hash AND u.country_code = 'us'
  WHERE p.country_code='us' AND p.is_scorable GROUP BY 1),
cls AS (
  SELECT form, count(DISTINCT text_hash) unicos_clasificados,
         count(DISTINCT text_hash) FILTER (WHERE has_frame) unicos_con_frame,
         count(DISTINCT (text_hash, frame_index)) FILTER (WHERE has_frame) frames_unicos,
         count(*) FILTER (WHERE has_frame) frames_instancias,
         count(DISTINCT accession_number) FILTER (WHERE has_frame) documentos_con_frame
  FROM gold_ai_frames WHERE country_code='us' GROUP BY 1)
SELECT * FROM docs LEFT JOIN pref USING (form) LEFT JOIN cls USING (form) ORDER BY parrafos DESC;
```
