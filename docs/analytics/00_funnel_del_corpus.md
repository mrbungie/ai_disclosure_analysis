# Funnel del corpus: de documentos a frames

EE.UU., prefiltro vigente (v2, árboles, umbral 0,17, `run=20260906T160624Z`),
clasificación completa, y segunda pasada de actividades (`09`) completa. Todo sale de `duckdb/thesis.duckdb` (`paragraphs`,
`unique_paragraphs`, `gold_ai_frames`) y del parquet de predicciones del
prefiltro; la consulta está al final.

Cuatro bloques, de arriba hacia abajo. La **unidad** cambia entre bloques y
está dicha en cada fila: documentos, párrafos como instancias, textos únicos
(un mismo párrafo repetido en varios documentos cuenta una vez), empresas.
Costo y cobertura se miden en textos únicos; todo lo que es intensidad por
documento (`ai_intensity.py`) usa instancias.

**A. Corpus** (lo que hay)

| etapa | unidad | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| documentos | documentos | 2.873 | 8.099 | 2.646 | 34.416 | 7.947 | 55.981 |
| párrafos | instancias | 1.544.496 | 1.736.542 | 3.679.553 | 1.623.906 | 554.281 | 9.138.778 |
| párrafos puntuables (largo mínimo, no tabla vacía) | instancias | 1.422.746 | 1.593.351 | 3.254.803 | 1.529.705 | 553.942 | 8.354.547 |
| textos únicos puntuables | textos únicos | 855.682 | 849.415 | 2.273.716 | 406.756 | 483.235 | 4.868.804 |

**B. Prefiltro** (¿menciona IA?) — v2, umbral 0,17, sobre textos únicos

| etapa | unidad | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| textos únicos positivos | textos únicos | 11.293 | 2.933 | 6.383 | 384 | 9.584 | 30.577 |
| % de los textos únicos puntuables | | 1,3% | 0,3% | 0,3% | 0,1% | 2,0% | 0,6% |
| instancias positivas (los mismos textos, en documentos) | instancias | 12.487 | 3.928 | 6.536 | 405 | 9.593 | 32.949 |

**C. Frames** (¿qué afirma sobre IA?) — juez `qwen3.7-flash`, una llamada por texto único positivo

| etapa | unidad | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| textos únicos clasificados | textos únicos | 11.294 | 2.933 | 6.384 | 384 | 9.584 | 30.579 |
| textos únicos con ≥1 frame | textos únicos | 10.532 | 2.587 | 5.216 | 232 | 7.902 | 26.469 |
| % de los clasificados (el resto son falsos positivos del prefiltro) | | 93% | 88% | 82% | 60% | 82% | 87% |
| frames únicos | frames | 16.245 | 3.409 | 7.168 | 295 | 16.249 | 43.366 |
| frames en documentos | instancias | 17.758 | 4.590 | 7.284 | 313 | 16.270 | 46.215 |
| documentos con ≥1 frame | documentos | 1.681 | 1.431 | 1.090 | 206 | 2.709 | 7.117 |
| % de los documentos del formulario | | 59% | 18% | 41% | 0,6% | 34% | 13% |

**D. Actividades** (¿qué dice que hace?) — segunda pasada, sólo textos con un frame conductual de la propia empresa

| etapa | unidad | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| textos únicos con frame conductual de la empresa | textos únicos | 5.927 | 1.829 | 2.923 | 134 | 6.803 | 17.420 |
| % de los textos con frame | | 56% | 71% | 56% | 58% | 86% | 66% |
| actividades únicas | actividades | 8.556 | 2.408 | 2.670 | 163 | 14.020 | 27.566 |
| actividades por texto conductual | | 1,4 | 1,3 | 0,9 | 1,2 | 2,1 | 1,6 |
| actividades en documentos | instancias | 9.224 | 3.238 | 2.708 | 165 | 14.033 | 29.368 |
| documentos con ≥1 actividad | documentos | 1.298 | 961 | 586 | 88 | 2.444 | 5.377 |
| empresas con ≥1 actividad | empresas | 409 | 159 | 269 | 55 | 383 | 471 (de 510) |

Las columnas por formulario de textos únicos suman más que el total porque un
texto puede aparecer en dos formularios y se cuenta en cada uno. El 27.583 de
`09` cuenta cada actividad una vez por empresa (17 textos están en dos
empresas); el 27.566 de acá la cuenta una vez.

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
