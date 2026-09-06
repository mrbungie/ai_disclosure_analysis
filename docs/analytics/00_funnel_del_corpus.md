# Funnel del corpus: de documentos a frames

EE.UU., prefiltro vigente (v2, árboles, umbral 0,17, `run=20260906T160624Z`),
clasificación completa. Todo sale de `duckdb/thesis.duckdb` (`paragraphs`,
`unique_paragraphs`, `gold_ai_frames`) y del parquet de predicciones del
prefiltro; la consulta está al final.

| etapa | 10-K | 10-Q | DEF 14A | 8-K | Calls | Total |
|---|---:|---:|---:|---:|---:|---:|
| documentos | 2.873 | 8.099 | 2.646 | 34.416 | 7.947 | 55.981 |
| párrafos (instancias) | 1.544.496 | 1.736.542 | 3.679.553 | 1.623.906 | 554.281 | 9.138.778 |
| párrafos puntuables | 1.422.746 | 1.593.351 | 3.254.803 | 1.529.705 | 553.942 | 8.354.547 |
| textos únicos puntuables | 855.682 | 849.415 | 2.273.716 | 406.756 | 483.235 | 4.868.804 |
| textos únicos positivos del prefiltro | 11.293 | 2.933 | 6.383 | 384 | 9.584 | 30.577 |
| instancias positivas | 12.487 | 3.928 | 6.536 | 405 | 9.593 | 32.949 |
| textos únicos clasificados por el juez | 11.294 | 2.933 | 6.384 | 384 | 9.584 | 30.579 |
| textos únicos con ≥1 frame | 10.532 | 2.587 | 5.216 | 232 | 7.902 | 26.469 |
| frames únicos | 16.245 | 3.409 | 7.168 | 295 | 16.249 | 43.366 |
| frames como instancias (en documentos) | 17.758 | 4.590 | 7.284 | 313 | 16.270 | 46.215 |
| documentos con ≥1 frame | 1.681 | 1.431 | 1.090 | 206 | 2.709 | 7.117 |

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
- La suma de textos únicos positivos por formulario (30.577) es mayor que el
  total del prefiltro (30.280) porque un texto puede aparecer en dos
  formularios y se cuenta en cada uno.
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
