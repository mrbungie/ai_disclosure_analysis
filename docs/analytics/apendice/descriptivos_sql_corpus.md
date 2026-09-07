> **Apéndice.** Descriptivos del corpus por consultas SQL: conteos por año, sector, temporalidad, sujeto, riesgos, dominio, tipo de IA y entidades. Complementa `00_funnel_del_corpus.md` y `01_evolucion_2021_2025.md`; no entra al cuerpo.

# Analytics preliminares de divulgación de IA (EE.UU.)

10 preguntas cortas, resueltas con SQL simple sobre `gold_ai_frames` /
`gold_ai_entity_mentions` (`duckdb/thesis.duckdb`) — ver
`docs/prefilter_evaluation.md` §8.8-§8.13 para cómo se construyeron esas
tablas. Todas las consultas están en la sección final para reproducirlas.

Alcance: **EE.UU.** (Italia y Chile tienen párrafos en el corpus pero no
entran a estas consultas — filtro `country_code='us'`). Población:
**18.280 textos únicos** de formularios SEC marcados IA-relevantes con al
menos un frame y **29.945 frames** semánticos extraídos vía
`ai_classify.py` (qwen/qwen3.7-flash), sobre un corpus prefiltrado de
4.799.469 textos únicos (30.280 marcados por el prefiltro v2, umbral 0,17,
`run=20260906T160624Z`). Las earnings calls (9.584 textos, 16.270 frames)
están en `gold_ai_frames` pero se reportan aparte: son otro canal, con
otro error de medición (`07_shocks_sec_deepseek.md`, `06_brecha_entre_canales.md`).

Composición de la población por formulario:

| Formulario | Textos únicos con frame | Frames |
|---|---:|---:|
| 10-K | 11.294 | 17.758 |
| DEF 14A | 6.384 | 7.284 |
| 10-Q | 2.933 | 4.590 |
| 8-K | 384 | 313 |
| *Earnings call (aparte)* | *9.584* | *16.270* |

## Advertencia de comparabilidad entre formularios

> El prefiltro mide distinto en cada formulario (`docs/prefilter_evaluation.md`
> §8.16): recall 0,96 en el holdout de DEF 14A / 8-K con precisión 0,73, contra
> 0,98 / 0,98 en 10-K/10-Q; en earnings calls precisión 0,59 con recall 0,98. Es decir:
> **el instrumento no mide igual en los canales**, y toda comparación entre
> ellos —incluida la #8, la más citada de este documento— mezcla diferencia de
> discurso con diferencia de error de medición.


Casi cualquier conteo agregado sobre estos cuatro formularios mezcla dos
cosas distintas: **intensidad de divulgación** y **volumen documental**.
Una empresa presenta un 10-K al año y una DEF 14A al año, pero muchos 8-K.
Por eso las preguntas sensibles a eso (#1, #8, #9) se reportan
desglosadas por `form`, no pooled — el agregado existe, pero no se lee
como "las empresas hablan más de IA".

## Resultados

| # | Pregunta | Hallazgo |
|---|---|---|
| 1 | ¿Cómo evoluciona la cantidad de párrafos con mención de IA por año de filing? | Agregado (10-K + DEF 14A + 8-K): 745 (2021) → 872 (2022) → 1.217 (2023) → 3.110 (2024) → 4.853 (2025) → 6.225 (2026, parcial). El salto 2023→2024 es ×2,6 en textos y ×1,9 en filings con IA (10-K: 188 → 352), o sea que es más divulgación por documento y también más documentos. Desglose en la consulta 1b: 10-K 536 → 3.995, DEF 14A 204 → 2.153, 8-K 5 → 85. |
| 2 | ¿Qué sectores (SIC 2 dígitos) tienen mayor prevalencia de empresas con divulgación de IA? | SIC 73 (software/servicios de cómputo): 66/66 empresas (100%). SIC 63 (seguros): 23/23 (100%). SIC 49 (utilities): 33/34 (97%). SIC 35 (maquinaria/computadores): 28/29. SIC 67 (holdings): 26/27. SIC 38 (instrumentos): 39/42. SIC 28 (químicos/farma): 37/40. SIC 36 (electrónica): 28/30. La prevalencia es casi universal en el S&P 500: el contraste está en cuánto y cómo, no en si. |
| 3 | ¿Qué proporción de las afirmaciones sobre IA son sobre algo YA ocurrido vs. planeado/esperado/hipotético? | 72,1% `realized` (21.598), 12,5% `planned` (3.731), 8,2% `expected` (2.470), 7,2% `hypothetical` (2.146). En earnings calls: 69,4% / 17,9% / 12,0% / **0,8%** hipotético — en la call casi nada se enmarca como hipotético. |
| 4 | ¿De quién se habla cuando se habla de IA — la propia empresa, sus clientes, o la competencia? | 85,3% `firm` (25.557), 9,1% `competitors_or_industry` (2.723), 4,9% `customers` (1.480), 0,6% `suppliers_or_partners` (185). En calls sube `customers` a 10,3%. |
| 5 | ¿Qué empresas tienen más frames de IA extraídos (más "vocales" sobre IA en sus filings)? | MSFT (605), NVDA (553), GOOGL (504), ADBE (494), INTC (480), AMZN (430), CRM (352), SNOW (320), HPE (320), WDAY (305), META (303), AMD (273), PANW (270), CTSH (266), EFX (266). Sólo formularios; las calls cambian el ranking (`earnings_calls_analysis.py`: NVDA, MSFT, META, ADBE, GOOGL). |
| 6 | ¿Cuáles son los riesgos de IA más mencionados? | Ciberseguridad (3.113), regulatorio/legal (2.786), competitivo/disrupción (1.725), confiabilidad/precisión (1.617), dependencia operacional (1.216), privacidad (1.167), propiedad intelectual (928), sesgo/equidad (660), fuerza laboral (415). |
| 7 | ¿La IA se usa más internamente o de cara al cliente? | 46,6% interno (13.955), 37,7% customer-facing (11.288), 15,7% sin especificar (4.702) en formularios. En earnings calls se invierte: 55,6% customer-facing, 35,1% interno — en la call se habla del producto, en el filing del proceso. |
| 8 | ¿Qué fracción de las afirmaciones sobre IA usa lenguaje promocional/superlativo? | 9,5% agregado en formularios (2.838 de 29.945) — pero el agregado esconde el hallazgo real: **DEF 14A 14,7%** (1.069/7.284) vs. **10-K 7,1%** (1.262/17.758), con 10-Q 10,5% y 8-K 7,3%. Y las **earnings calls 21,0%** (3.422/16.270): tres veces el 10-K. Leer con la advertencia de comparabilidad de arriba. |
| 9 | ¿Cómo evoluciona generativa vs. predictiva/ML clásica en el tiempo? | 2021: 178 gen / 196 predictiva (predictiva domina). 2023: 469 / 230 (cruce). 2026: 2.849 gen / 213 predictiva. La generativa pasa de minoría a ~93% de las menciones con tipo declarado; la predictiva se mantiene plana en ~200-260 por año. |
| 10 | ¿Qué entidades de IA específicas se nombran más, y de dónde son? | Con frame de IA confirmado: OpenAI (105), Copilot (100), Gemini (56), Anthropic (35), ChatGPT (30), watsonx (28), Vertex AI (16), Meta AI (9), GitHub Copilot (9). No estadounidenses: Cohere (5). `claude` queda en 6 (ver la sección de falsos positivos). |

## Falsos positivos de entidades introducidos por DEF 14A

`gold_ai_entity_mentions` es regex literal sin ninguna validación
semántica (por diseño — ver `ai_entity_mentions.py`). Eso funcionaba
mientras el corpus fuera 10-K/10-Q. Con proxy statements deja de
funcionar para los términos que además son **nombres de persona**:

- `claude` pasó de 30 a 218 textos únicos. 191 de esos están en DEF 14A,
  y son biografías de directores y tablas de compensación donde "Claude"
  es un nombre de pila. Uno de los textos es, literalmente, la palabra
  `Claude` sola.
- El juez LLM los rechaza correctamente: de esos 191 textos, sólo **5**
  recibieron un frame de IA. En 8-K, 21 textos con `claude` y **cero**
  frames.

Consecuencias, y qué se hizo:

- **Las preguntas #1 a #9 no están afectadas.** Todas filtran por
  `has_frame`, así que los falsos positivos nunca entran: el juez ya los
  descartó.
- **La #10 sí lo estaba**, porque leía `gold_ai_entity_mentions` directo.
  Su consulta ahora exige que el texto tenga un frame de IA confirmado.
  Con ese filtro `claude` cae de 218 a 6, que es el número honesto.
- El costo real fue de ~209 llamadas LLM desperdiciadas: el override
  `NAMED_AI_ENTITIES` de `ai_prefilter_classify.py` fuerza la inclusión de
  cualquier párrafo que nombre una entidad de la lista, y 209 de los 304
  rescates son de DEF 14A.
- **Pendiente, no resuelto acá:** la premisa del override ("una mención de
  estas es esencialmente nunca ruido") es falsa en proxy statements. La
  lista ya excluye `watson`, `bard`, `grok` y `sora` por ambigüedad;
  `claude` es exactamente la misma clase de palabra y sobrevivió sólo
  porque el 10-K y el 10-Q no tienen biografías de directorio. Corregirlo
  requiere decidir si se saca de la lista, se condiciona por formulario, o
  se acepta el costo sabiendo que el juez lo filtra.

## Consultas SQL

```sql
-- 1. Evolución temporal (agregada)
SELECT extract(year from fm.filing_date) AS anio, COUNT(DISTINCT f.text_hash) AS textos_unicos_ia
FROM gold_ai_frames f
JOIN filing_manifest fm USING (country_code, accession_number)
WHERE f.country_code='us' AND f.has_frame
GROUP BY 1 ORDER BY 1;

-- 1b. Evolución temporal DESGLOSADA por formulario (la lectura correcta:
--     `filings` expone si un salto es más divulgación o más documentos)
SELECT extract(year from fm.filing_date) AS anio, f.form,
       COUNT(DISTINCT f.text_hash) AS textos_ia,
       COUNT(DISTINCT fm.accession_number) AS filings
FROM gold_ai_frames f
JOIN filing_manifest fm USING (country_code, accession_number)
WHERE f.country_code='us' AND f.has_frame
GROUP BY 1,2 ORDER BY 1,2;

-- 2. Prevalencia por sector (SIC 2 dígitos)
WITH ai_firms AS (
    SELECT DISTINCT fm.cik
    FROM gold_ai_frames f JOIN filing_manifest fm USING (country_code, accession_number)
    WHERE f.country_code='us' AND f.has_frame
), all_firms AS (
    SELECT cik, LEFT(sic,2) AS sic2 FROM firm_universe WHERE country_code='us' AND sic IS NOT NULL
), totals AS (
    SELECT sic2, COUNT(DISTINCT cik) AS empresas_total FROM all_firms GROUP BY 1
)
SELECT af.sic2, COUNT(DISTINCT af.cik) AS empresas_con_ia, t.empresas_total
FROM all_firms af
JOIN ai_firms ON ai_firms.cik = af.cik
JOIN totals t ON t.sic2 = af.sic2
GROUP BY 1, t.empresas_total ORDER BY empresas_con_ia DESC LIMIT 10;

-- 3. Temporalidad
SELECT temporal, COUNT(*) n
FROM gold_ai_frames WHERE country_code='us' AND has_frame
GROUP BY 1 ORDER BY 2 DESC;

-- 4. Sujeto
SELECT subject, COUNT(*) n
FROM gold_ai_frames WHERE country_code='us' AND has_frame
GROUP BY 1 ORDER BY 2 DESC;

-- 5. Top empresas por frames
SELECT fm.ticker, COUNT(*) n_frames
FROM gold_ai_frames f JOIN filing_manifest fm USING (country_code, accession_number)
WHERE f.country_code='us' AND f.has_frame
GROUP BY 1 ORDER BY 2 DESC LIMIT 15;

-- 6. Conceptos de riesgo
SELECT concept, COUNT(*) n
FROM (SELECT UNNEST(concepts) AS concept FROM gold_ai_frames WHERE country_code='us' AND has_frame)
WHERE concept LIKE 'risk_%'
GROUP BY 1 ORDER BY 2 DESC;

-- 7. Dominio
SELECT domain, COUNT(*) n
FROM gold_ai_frames WHERE country_code='us' AND has_frame
GROUP BY 1 ORDER BY 2 DESC;

-- 8. Lenguaje promocional (agregado)
SELECT rhetoric_promotional, COUNT(*) n, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),1) pct
FROM gold_ai_frames WHERE country_code='us' AND has_frame
GROUP BY 1;

-- 8b. Lenguaje promocional POR FORMULARIO — el hallazgo real de #8
SELECT form,
       COUNT(*) FILTER (WHERE rhetoric_promotional) AS promocionales,
       COUNT(*) AS frames,
       ROUND(100.0*COUNT(*) FILTER (WHERE rhetoric_promotional)/COUNT(*),1) AS pct
FROM gold_ai_frames WHERE country_code='us' AND has_frame
GROUP BY 1 ORDER BY 3 DESC;

-- 9. Tipo de IA por año
SELECT extract(year from fm.filing_date) AS anio, f.ai_type, COUNT(*) n
FROM gold_ai_frames f JOIN filing_manifest fm USING (country_code, accession_number)
WHERE f.country_code='us' AND f.has_frame AND f.ai_type != 'unspecified'
GROUP BY 1,2 ORDER BY 1,2;

-- 10. Entidades por geografía — EXIGE frame de IA confirmado por el juez.
--     Sin ese EXISTS, DEF 14A infla `claude` de 6 a 218 con nombres de
--     directores (ver la sección de falsos positivos más arriba).
SELECT m.geo, m.term, COUNT(DISTINCT m.text_hash) n
FROM gold_ai_entity_mentions m
WHERE m.country_code='us'
  AND EXISTS (SELECT 1 FROM gold_ai_frames f WHERE f.text_hash = m.text_hash AND f.has_frame)
GROUP BY 1,2 ORDER BY 3 DESC LIMIT 15;
```

## Notas metodológicas

- Todas las consultas filtran `country_code='us'` explícitamente — quitar
  ese filtro las hace correr sobre cualquier país que entre a
  `gold_ai_frames`/`gold_ai_entity_mentions` más adelante, sin cambiar
  nada más (ver docs/prefilter_evaluation.md §8.12).
- `#2` cuenta EMPRESAS (por `cik`), no párrafos — una empresa cuenta una
  vez si tiene al menos un frame, independiente de cuántos.
- `#1` y `#9` usan `filing_date` de `filing_manifest`, no la fecha de
  clasificación — el año real del filing, no cuándo corrió el LLM.
- **El 10-Q no aparece en `#1`, `#2`, `#5` ni `#9`.** Esas consultas hacen
  JOIN contra `filing_manifest`, que por decisión de alcance del proyecto
  NO incluye el 10-Q (instrumento separado, con su propio
  `filing_manifest_10q` — ver README). Los 2.381 textos únicos de 10-Q de
  la población quedan fuera de esas cuatro consultas y dentro de las
  otras seis.
- Ninguna de estas cifras está ponderada por `inclusion_weight` — son
  conteos directos sobre la población ya filtrada por el prefiltro, no
  estimaciones de prevalencia del corpus completo (para eso ver
  docs/prefilter_evaluation.md §8.9's estimador de `stage3_random`).
