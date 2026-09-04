# Analytics preliminares de divulgación de IA (EE.UU.)

10 preguntas cortas, resueltas con SQL simple sobre `gold_ai_frames` /
`gold_ai_entity_mentions` (`duckdb/thesis.duckdb`) — ver
`docs/prefilter_evaluation.md` §8.8-§8.13 para cómo se construyeron esas
tablas. Todas las consultas están en la sección final para reproducirlas.

Alcance: **solo EE.UU.** (Chile todavía no tiene embeddings/scoring —
ver §8.9). Población: 13.442 textos únicos marcados IA-relevantes
(15.945 instancias), 22.622 frames semánticos extraídos vía
`ai_classify.py` (qwen/qwen3.7-flash).

## Resultados

| # | Pregunta | Hallazgo |
|---|---|---|
| 1 | ¿Cómo evoluciona la cantidad de párrafos con mención de IA por año de filing? | Crecimiento sostenido: 512 (2021) → 578 (2022) → 868 (2023) → 2.184 (2024) → 3.316 (2025) → 4.018 (2026, año en curso, parcial). El salto 2023→2024 (+152%) coincide con la ola de adopción de IA generativa post-ChatGPT. |
| 2 | ¿Qué sectores (SIC 2 dígitos) tienen mayor prevalencia de empresas con divulgación de IA? | SIC 73 (Servicios de cómputo/software): 66/66 empresas (100%) mencionan IA. SIC 38 (Instrumentos): 38/42 (90%). SIC 28 (Químicos/farma): 34/40 (85%). El sobremuestreo tech del golden set (docs/golden_set_sampling.md) anticipaba exactamente esto. |
| 3 | ¿Qué proporción de las afirmaciones sobre IA son sobre algo YA ocurrido vs. planeado/esperado/hipotético? | 67% `realized` (15.353), 13% `planned` (2.877), 10% `expected` (2.232), 9% `hypothetical` (2.160) — la mayoría de las menciones de IA describen uso/inversión ya en curso, no promesas futuras. |
| 4 | ¿De quién se habla cuando se habla de IA — la propia empresa, sus clientes, o la competencia? | 82% `firm` (18.728 frames), 11% `competitors_or_industry` (2.494), 5,5% `customers` (1.265), 0,6% `suppliers_or_partners` (135) — la gran mayoría es autodescripción, no presión competitiva ni demanda de clientes. |
| 5 | ¿Qué empresas tienen más frames de IA extraídos (más "vocales" sobre IA en sus filings)? | NVDA (484), MSFT (422), INTC (385), ADBE (361), SNOW (273), GOOGL (253), HPE (242), CRM (239), AMD (225), META (222) — dominado por semiconductores/infraestructura de cómputo y software empresarial, consistente con el sesgo sectorial esperado. |
| 6 | ¿Cuáles son los riesgos de IA más mencionados? | Ciberseguridad (2.793), regulatorio/legal (2.711), competitivo/disrupción (1.713), confiabilidad/precisión (1.580), dependencia operacional (1.189), privacidad (1.073), propiedad intelectual (917), sesgo/equidad (586, el menos mencionado con diferencia). |
| 7 | ¿La IA se usa más internamente o de cara al cliente? | Casi empatado: 44% interno (9.296), 43% customer-facing (9.188), 19% sin especificar (4.138) — no hay un sesgo claro hacia uso interno (eficiencia) vs. producto (crecimiento). |
| 8 | ¿Qué fracción de las afirmaciones sobre IA usa lenguaje promocional/superlativo? | Solo 8% (1.820 de 22.622 frames) — la gran mayoría del texto es descriptivo/neutro, no marketing evidente. Relevante como línea base antes de cualquier análisis de AI-washing más fino. |
| 9 | ¿Cómo evoluciona generativa vs. predictiva/ML clásica en el tiempo? | 2021: 121 gen / 145 predictiva (predictiva domina). 2024: 1.034 gen / 201 predictiva. 2026: 1.856 gen / 165 predictiva — la IA generativa pasó de ser minoría a ~92% de las menciones con tipo especificado, cruce claro entre 2023-2024. |
| 10 | ¿Qué entidades de IA específicas se nombran más, y de dónde son? | Copilot (78), OpenAI (59), Gemini (45), Anthropic (30), watsonx (22), ChatGPT (15) — todas EE.UU. Únicas menciones no-EE.UU. en todo el corpus: Stable Diffusion/Europa (3) y DeepSeek/China (2). El "AI-washing" nombrado, cuando ocurre, es casi exclusivamente sobre proveedores estadounidenses. |

## Consultas SQL

```sql
-- 1. Evolución temporal
SELECT extract(year from fm.filing_date) AS anio, COUNT(DISTINCT f.text_hash) AS textos_unicos_ia
FROM gold_ai_frames f
JOIN filing_manifest fm USING (country_code, accession_number)
WHERE f.country_code='us' AND f.has_frame
GROUP BY 1 ORDER BY 1;

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

-- 8. Lenguaje promocional
SELECT rhetoric_promotional, COUNT(*) n, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),1) pct
FROM gold_ai_frames WHERE country_code='us' AND has_frame
GROUP BY 1;

-- 9. Tipo de IA por año
SELECT extract(year from fm.filing_date) AS anio, f.ai_type, COUNT(*) n
FROM gold_ai_frames f JOIN filing_manifest fm USING (country_code, accession_number)
WHERE f.country_code='us' AND f.has_frame AND f.ai_type != 'unspecified'
GROUP BY 1,2 ORDER BY 1,2;

-- 10. Entidades por geografía
SELECT geo, term, COUNT(DISTINCT text_hash) n
FROM gold_ai_entity_mentions WHERE country_code='us'
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
- Ninguna de estas cifras está ponderada por `inclusion_weight` — son
  conteos directos sobre la población ya filtrada por el prefiltro, no
  estimaciones de prevalencia del corpus completo (para eso ver
  docs/prefilter_evaluation.md §8.9's estimador de `stage3_random`).
