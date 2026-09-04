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

---

# Arquetipos de comportamiento (EE.UU.)

Primera exploración de la pregunta central de la tesis
(`docs/thesis_proposal.md`): ¿en qué arquetipos distintos se agrupan las
empresas según CÓMO divulgan IA, no solo cuánto? Se agregó cada empresa
(ticker) a partir de sus frames de `gold_ai_frames` en 9 métricas, y se
agruparon con K-means.

## Método

- **Población**: 421 empresas de EE.UU. con ≥5 frames (filtro de volumen
  mínimo para que el perfil agregado no sea ruido de 1-2 menciones).
- **Métricas por empresa** (promedio sobre sus frames):
  - `specificity_index`: promedio de las 5 banderas de especificidad
    (proceso de negocio, producto/sistema, proveedor, métrica
    cuantificada, fecha/cronograma) — "qué tan concreto" es el discurso.
  - `quantified_rate`: fracción de frames con una métrica numérica
    explícita (subconjunto de lo anterior, aislado porque es el
    indicador más directo de "sustancia" vs. "promesa vacía").
  - `promotional_rate` / `strategic_rate`: señales retóricas
    (`rhetoric_promotional`, `rhetoric_strategic_importance`).
  - `realized_share` / `hypothetical_share`: fracción de frames en cada
    extremo de `temporal` (ya ocurrido vs. condicional/especulativo).
  - `risk_share` / `gov_share`: fracción de frames con al menos un
    concepto `risk_*` / `gov_*`.
  - `firm_subject_share`: fracción de frames con `subject='firm'` (habla
    de sí misma, no de clientes/competencia).
- **Clustering**: K-means sobre las 9 métricas estandarizadas
  (media 0, varianza 1). k=4 elegido por curva de codo (inercia baja
  monótonamente sin quiebre marcado — 4 es el punto donde los clusters
  siguen siendo interpretables sin fragmentarse en grupos triviales).

## Los 4 arquetipos

| Arquetipo | n empresas | Especificidad | % cuantificado | % promocional | % ya realizado | % riesgo | Empresas típicas |
|---|---|---|---|---|---|---|---|
| **A. Listadores de riesgo cautelosos** | 101 | 0,05 (mín.) | 0,5% | 0,7% (mín.) | 39% (mín.) | 77% (máx.) | BAX, IFF, IT, WELL, A, PRU, FTV, GPN |
| **B. Adoptantes genéricos** | 200 | 0,09 | 0,8% | 1,0% | 61% | 60% | MA, NWS, ADSK, AXP, BKNG, MSCI, GEN, RHI |
| **C. Cuantificadores concretos** | 7 | 0,27 (máx.) | 31% (máx.) | 6,6% | 66% | 24% (mín.) | NRG, AES, APH, APTV, ALLE, DUK, TSN |
| **D. Líderes vocales de IA** | 113 | 0,16 | 3,3% | 10% (máx.) | 71% (máx.) | 31% | NVDA, MSFT, INTC, ADBE, SNOW, GOOGL, HPE, CRM |

**A — Listadores de riesgo cautelosos** (101 empresas, salud/seguros/industrial
diversificado). Mencionan IA casi exclusivamente como riesgo genérico
futuro (77% de sus frames son de riesgo, 35% hipotético) — el patrón
clásico de "boilerplate de risk factors" que enumera IA junto a otras
amenazas tecnológicas sin describir uso propio. Especificidad y
promoción prácticamente en cero. Es el grupo con MENOR riesgo de
AI-washing simplemente porque casi no hace ninguna afirmación positiva
que verificar.

**B — Adoptantes genéricos** (200 empresas, el grupo más grande — la
"empresa promedio"). Hablan de IA como algo ya en curso (61% realizado)
pero de forma llana: baja especificidad, casi nada cuantificado, casi
nada promocional. Ni sustancia fuerte ni bombo — el punto medio del
espectro que la tesis quiere segmentar.

**C — Cuantificadores concretos** (solo 7 empresas — NRG, AES, Amphenol,
Aptiv, Allegion, Duke Energy, Tyson — energía/industrial). Grupo chico
pero muy distinto: la especificidad y sobre todo la cuantificación
(31%, ~40x el resto) son con diferencia las más altas, y el foco en
riesgo/gobernanza el más bajo. No hablan de "adoptar IA" — hablan de IA
como **motor de demanda externa** con cifras concretas (ej. demanda de
centros de datos impulsada por IA), un patrón ya visto en el análisis de
falsos negativos de `docs/prefilter_evaluation.md` §8.9. Metodológicamente
interesante: es sustancia real, pero sobre el efecto de la IA en el
NEGOCIO de un tercero, no sobre capacidad de IA propia — no encaja
limpiamente en el eje "washing vs. creíble" tal como está planteado.

**D — Líderes vocales de IA** (113 empresas, ~10x más frames por empresa
que cualquier otro grupo — NVDA, MSFT, INTC, ADBE, SNOW, GOOGL, HPE,
CRM). El grupo más interesante para la pregunta de la tesis: especificidad
alta (segunda más alta) Y retórica promocional más alta (10%, el doble
que el resto) AL MISMO TIEMPO — no es un trade-off. Son las empresas que
más hablan de IA, con más sustancia real, pero también con el lenguaje
más superlativo. La pregunta de "¿es AI-washing?" no se resuelve por
volumen ni por tono solos — este grupo tiene ambos altos, lo que sugiere
que dentro de él mismo hay variación real (algunos frames sustanciando
las afirmaciones promocionales, otros no) que ninguna métrica agregada a
nivel empresa puede separar. Candidato natural para el siguiente nivel de
análisis: mirar frame por frame DENTRO de este cluster, no solo el
promedio por empresa.

## Notas metodológicas (arquetipos)

- Exploratorio, no definitivo: k=4 es una elección razonable por la curva
  de codo, no la única. Vale la pena repetir con k=3,5,6 y ver si C
  (7 empresas) se sostiene como grupo propio o se disuelve — un cluster
  tan chico es sensible a la semilla y al k elegido.
- Ninguna métrica está ponderada por tamaño de filing ni por cuántas
  veces se repite el mismo párrafo entre años (`duplicate_count`) — una
  empresa que reusa el mismo texto de un año a otro cuenta cada aparición
  como una observación independiente. Corregir esto es un paso pendiente
  antes de cualquier resultado publicable.
- No incluye Chile (todavía sin scoring, §8.9) ni pondera por
  `inclusion_weight` — son arquetipos sobre la población ya filtrada por
  el prefiltro (13.442 textos), no sobre el corpus completo.
