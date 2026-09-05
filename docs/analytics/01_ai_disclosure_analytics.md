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

## Evolución temporal — y por qué el agregado engaña

Pregunta directa: ¿los arquetipos cambian en el tiempo? Sí, pero el
resultado agregado es casi lo opuesto de lo esperable a primera vista, y
la explicación real es composicional, no de comportamiento individual.

### La tendencia agregada (todos los frames, por año de filing)

| año | frames | % promocional | % cuantificado | especificidad | % hipotético |
|---|---|---|---|---|---|
| 2021 | 755 | 12,7% | 6,6% | 0,206 | 4,8% |
| 2022 | 873 | 14,1% | 5,5% | 0,201 | 4,6% |
| 2023 | 1.327 | 11,0% | 4,3% | 0,168 | 6,0% |
| 2024 | 3.397 | 7,9% | 2,8% | 0,130 | 11,6% |
| 2025 | 5.254 | 6,1% | 2,6% | 0,130 | 11,5% |

A primera vista esto sugiere lo contrario de "más AI-washing": la
retórica promocional CAE a la mitad (12,7%→6,1%) y la especificidad
también cae (0,206→0,130), mientras el volumen se multiplica por 7. Sería
tentador leer esto como "las empresas se volvieron más cautelosas
después del escrutinio de la SEC en 2024" — pero la caída ya viene desde
2021, sin quiebre visible justo en 2024, así que esa lectura causal no
se sostiene con este solo dato.

### La explicación real: es composición, no comportamiento individual

Asignando cada empresa a un arquetipo usando SOLO los frames de ESE año
(no el pool completo), y proyectando sobre los mismos 4 centroides:

| año | empresas con perfil ese año | % líderes vocales (D) | % adoptantes genéricos (B) | % riesgo cauteloso (A) |
|---|---|---|---|---|
| 2022 | 87 | **63%** (55) | 23% (20) | 7% (6) |
| 2024 | 247 | 27% (66) | 42% (103) | 28% (70) |
| 2025 | 337 | 23% (77) | 48% (163) | 27% (92) |

En 2022, la mayoría de las empresas con volumen suficiente para tener un
perfil YA eran "líderes vocales" — el grupo que habla mucho, con
especificidad y promoción altas. Para 2024-2025, el número de empresas
con volumen suficiente casi se cuadruplicó (87→337), pero ese crecimiento
es casi todo de empresas "genéricas" y "cautelosas" que recién empiezan a
mencionar IA a escala — probablemente presión regulatoria/de mercado
para tener AL MENOS un párrafo de riesgo sobre IA, no adopción real nueva.
**El grupo de líderes vocales, mirado por separado, cae mucho menos**
(15,1%→17,3%→13,7%→12,7%→10,4% de retórica promocional en su propio
arquetipo, año a año — ver tabla completa abajo) que el agregado
(12,7%→6,1%). La "caída" en el promedio general es sobre todo dilución:
muchas más empresas de baja intensidad entrando al denominador, no las
mismas empresas volviéndose menos promocionales.

<details>
<summary>Tendencia por arquetipo (pool completo, no solo el año de asignación)</summary>

| arquetipo | año | frames | % promocional | % cuantificado |
|---|---|---|---|---|
| A. Riesgo cauteloso | 2021→2025 | 7→585 | 0%→1,5% | 0%→0,9% |
| B. Adoptante genérico | 2021→2025 | 155→1.900 | 4,5%→1,3% | 0,7%→1,4% |
| C. Cuantificador concreto | 2021→2025 | 2→44 | 0%→4,6% | 100%→22,7% (n muy chico en 2021-2022, no confiable) |
| D. Líder vocal | 2021→2025 | 591→2.725 | 15,1%→10,4% | 8,0%→3,6% |

</details>

**Lectura para las preguntas extendidas de la tesis**:
- *SEC 2024*: no hay un quiebre discreto visible en 2024 en ningún
  arquetipo — la caída es gradual desde 2021 en todos. Si el escrutinio
  de la SEC tuvo efecto, no se ve como un salto en este corte anual; haría
  falta granularidad trimestral y una ventana más angosta alrededor del
  anuncio para no confundirlo con la tendencia composicional de fondo.
- *DeepSeek (2025)*: el volumen de menciones de China en
  `gold_ai_entity_mentions` sigue siendo mínimo (2 textos, tabla #10 de
  arriba) — no hay señal todavía de que el "shock DeepSeek" haya cambiado
  cómo las empresas de EE.UU. describen su posicionamiento competitivo.
  Puede ser demasiado reciente para el corpus actual (2026 parcial).
- El hallazgo más sólido no es "menos washing con el tiempo" sino **"el
  universo de empresas que hablan de IA se amplió mucho más rápido de lo
  que cambió el discurso de las que ya hablaban"** — un resultado en sí
  mismo relevante para el framework de benchmarking que propone la tesis.

### Notas metodológicas (evolución)

- 2025 (y 2026 en la tabla temporal de más arriba) son años fiscales
  parciales para muchas empresas — la comparación año a año no es
  limpia hasta que termine el año calendario.
- El arquetipo por año usa un umbral de volumen más bajo (≥3 frames en
  el año) que el arquetipo "de vida" (≥5 frames pooled) para no perder
  casi todas las empresas en los años tempranos — los conteos de
  empresas por año NO son directamente comparables al conteo total de
  421 de la sección anterior.
- Proyectar perfiles anuales sobre los centroides ya entrenados (en vez
  de re-clusterizar cada año) es deliberado: permite comparar "qué tan
  cerca de cada arquetipo original está esta empresa este año", pero
  asume que los 4 arquetipos de referencia siguen siendo la partición
  correcta en años tempranos con mucha menos data — otra razón para no
  sobre-interpretar 2021-2022 (n=87 empresas).

## Comportamientos: general y por arquetipo

Pregunta: más allá de CUÁNTO y CÓMO (retórica) se divulga IA, ¿QUÉ
comportamiento describe cada frame — despliegue ya en producción, piloto,
exploración, inversión, infraestructura, talento — y a QUIÉN nombran
cuando lo hacen (proveedores/productos de `gold_ai_entity_mentions`)? Y
¿ese patrón es distinto por arquetipo?

"Comportamiento" aquí = los `concepts` de `gold_ai_frames` que describen
etapa de uso o resultado (no los de riesgo/gobernanza, ya cubiertos en la
pregunta #6 de la tabla inicial): `deployed`, `pilot_or_testing`,
`exploring`, `ai_investment`, `ai_infrastructure`, `ai_talent`,
`proprietary_ai`, `third_party_ai`, `expansion_or_scaling`, y los
outcomes `productivity_outcome` / `revenue_outcome` / `cost_outcome` /
`customer_outcome`. Un frame puede tener 0, 1 o varios de estos
concepts a la vez.

### Comportamiento general (22.622 frames, EE.UU.)

| Concept | n | % de frames |
|---|---|---|
| `deployed` | 5.873 | 32,7% |
| `productivity_outcome` | 1.759 | 9,8% |
| `expansion_or_scaling` | 1.599 | 8,9% |
| `revenue_outcome` | 859 | 4,8% |
| `ai_investment` | 658 | 3,7% |
| `third_party_ai` | 605 | 3,4% |
| `cost_outcome` | 600 | 3,3% |
| `ai_infrastructure` | 503 | 2,8% |
| `proprietary_ai` | 405 | 2,3% |
| `customer_outcome` | 367 | 2,0% |
| `ai_talent` | 303 | 1,7% |
| `exploring` | 211 | 1,2% |
| `pilot_or_testing` | 167 | 0,9% |

Un tercio de los frames (33%) describe IA ya desplegada — consistente
con el 67% `realized` de la pregunta #3 de la tabla inicial (`deployed`
es más estricto: exige lenguaje de despliegue explícito, no solo tiempo
verbal pasado). Piloto/exploración combinados son apenas 2,1% — casi
nadie en el corpus describe la fase temprana de adopción; se salta
directo de "lo estamos evaluando" a "ya está en producción" en el
discurso público, lo que en sí mismo es sugerente de selección
(las empresas no reportan lo que todavía no funciona).

### Comportamiento por arquetipo (% de frames del arquetipo con el concept)

| Concept | A (cauteloso) | B (genérico) | C (cuantificador) | D (líder vocal) |
|---|---|---|---|---|
| `deployed` | 12,5% | 21,0% | 29,0% | **43,6%** |
| `pilot_or_testing` | 0,5% | 1,0% | 0,7% | 1,0% |
| `exploring` | 2,8% | 1,3% | 0,0% | 0,8% |
| `ai_investment` | 1,9% | 2,8% | **11,6%** | 4,3% |
| `ai_infrastructure` | 0,8% | 1,8% | 2,2% | **3,8%** |
| `ai_talent` | 1,0% | 1,2% | 0,7% | **2,1%** |
| `proprietary_ai` | 0,4% | 1,5% | 1,4% | **3,1%** |
| `third_party_ai` | 2,4% | 3,0% | **5,1%** | 3,8% |
| `expansion_or_scaling` | 5,3% | 7,7% | **14,5%** | 10,3% |
| `productivity_outcome` | 2,3% | 5,2% | 5,8% | **14,0%** |
| `revenue_outcome` | 1,6% | 3,5% | **21,7%** | 5,9% |
| `cost_outcome` | 2,7% | 3,2% | 5,8% | **5,9%** |
| `customer_outcome` | 0,7% | 1,3% | 0,7% | **2,8%** |

| Domain | A | B | C | D |
|---|---|---|---|---|
| `internal` | 57,0% | 53,2% | 29,7% | 33,2% |
| `customer_facing` | 15,3% | 24,2% | 34,1% | **51,4%** |
| `unspecified` | 27,6% | 22,6% | 36,2% | 15,4% |

- **A (cauteloso)** apenas despliega nada (12,5%, el mínimo) y cuando
  actúa es sobre todo interno (57%) — coherente con el perfil ya
  descrito de boilerplate de riesgo sin sustancia operativa.
- **C (cuantificador)** es el único arquetipo donde `revenue_outcome`
  (21,7%) supera con holgura a `productivity_outcome` (5,8%) —
  confirma con datos de comportamiento la lectura ya hecha por retórica:
  C no habla de "adoptar IA" internamente, habla de IA como motor de
  **demanda** de terceros (de ahí también su pico en `ai_investment`
  11,6% y `expansion_or_scaling` 14,5% — capex de infraestructura
  impulsado por otros, no productos propios).
- **D (líder vocal)** domina en casi todos los concepts de sustancia
  real y no solo en retórica: más `deployed` (43,6%), más
  `ai_infrastructure`, `ai_talent`, `proprietary_ai` y todos los
  outcomes salvo `revenue_outcome`. También es, con diferencia, el más
  `customer_facing` (51,4% vs. 15-34% del resto) — habla de IA como
  producto que vende, no solo como herramienta interna. Esto ancla con
  números duros la pregunta pendiente de la sección anterior: D no es
  solo el grupo "promocional", es también el grupo con más evidencia
  operativa real — la variación que importa para AI-washing está DENTRO
  de D, no en si D es sustancia o bombo en promedio.

### Menciones de entidades por arquetipo

Uniendo `gold_ai_entity_mentions` a `filing_manifest` (para heredar
`ticker` → arquetipo) igual que en la pregunta #10 de la tabla inicial.
Nota: esta unión pierde ~1/3 de las menciones nombradas respecto al
conteo global de la pregunta #10 (78 menciones de "Copilot" en el
corpus completo vs. 52 aquí) porque 29 `accession_number` de
`gold_ai_entity_mentions` no tienen fila en `filing_manifest` y por
tanto no pueden atribuirse a un ticker/arquetipo — ver notas
metodológicas abajo.

| Arquetipo | n empresas | Firmas que nombran ≥1 entidad | Menciones totales | Top entidades nombradas |
|---|---|---|---|---|
| A. Riesgo cauteloso | 103 | 1 (1,0%) | 1 | gemini (1) |
| B. Adoptantes genéricos | 197 | 9 (4,6%) | 22 | gemini (9), chatgpt (7), gpt-4 (3), anthropic (1), copilot (1), openai (1) |
| C. Cuantificadores concretos | 7 | 0 (0%) | 0 | — |
| D. Líderes vocales de IA | 114 | 19 (16,7%) | 155 | copilot (51), openai (25), gemini (23), anthropic (9), vertex ai (8), chatgpt (7) |

El nombrar un proveedor/producto de IA específico está casi enteramente
concentrado en D: 155 de las 178 menciones atribuibles (87%), y casi
1 de cada 6 empresas de D nombra al menos una entidad, contra 1 de cada
100 en A y ninguna en C. Esto es exactamente lo que predeciría el
resultado de la sección anterior — D es el arquetipo con más
especificidad Y más retórica promocional a la vez — y añade una tercera
pieza: cuando una empresa SÍ nombra un proveedor concreto (evidencia
verificable, en contraste con "usamos IA" genérico), casi siempre es una
empresa de D. C, pese a ser el arquetipo más cuantificado en general
(31% de frames con métrica numérica), no nombra ni una sola entidad de
IA — su cuantificación es sobre demanda/capex, no sobre qué modelo o
proveedor usan.

### Notas metodológicas (comportamientos)

- La asignación de arquetipo usada en esta sección es una
  **reconstrucción**, no la tabla original: mismo método (9 métricas
  estandarizadas, K-means k=4, población ≥5 frames, n=421) pero corrida
  de nuevo con `random_state=42` porque la corrida original no dejó
  fijada la semilla en el momento de generar la tabla de arquetipos
  arriba. Los tamaños de cluster resultantes (A=103, B=197, C=7, D=114)
  son muy cercanos pero NO idénticos a los documentados arriba
  (A=101, B=200, C=7, D=113) — incluso el cluster chico C mantiene
  exactamente 7 empresas, lo que sugiere que es un grupo estable frente
  a la semilla, no artefacto de una corrida particular.
- `deployed` es un concept que el LLM asigna de forma independiente de
  `temporal='realized'` — un frame puede ser `realized` sin `deployed`
  (p. ej. "ya evaluamos IA el año pasado" es tiempo pasado pero no
  necesariamente despliegue) — así que 32,7% (`deployed`) < 67%
  (`realized`, pregunta #3) es esperable, no contradictorio.
- El % de firmas que nombran una entidad NO está ponderado por cuántos
  frames tiene cada empresa — una empresa de D con 3 frames y otra con
  400 cuentan igual en el numerador "firmas que nombran ≥1 entidad".
- Igual que el resto del documento: solo EE.UU., sin ponderar por
  `inclusion_weight`, población ya filtrada por el prefiltro (no
  estimación de prevalencia del corpus completo).

## Panel empresa-año: transiciones de arquetipo y estrategias

Todo lo anterior es transversal (una empresa = un punto). Esta sección
baja el análisis a **empresa × año** para responder dos preguntas que el
corte transversal no puede: ¿el arquetipo de una empresa es un rasgo
fijo o cambia en el tiempo?, y dentro de "cómo divulgan IA", ¿qué
**hacen** — construyen producto o lo usan puertas adentro, construyen IA
propia o integran de terceros, y qué resultado (`outcome`) enfatizan?

### Construcción del panel

- Se recalculan las 9 métricas del clustering (§ "Método" arriba) por
  `(ticker, año de filing)` en vez de pooladas por empresa.
- Los años-empresa se **proyectan** contra los mismos 4 centroides ya
  entrenados sobre el pool completo (no se re-clusteriza por año) —
  así "arquetipo 2023 de esta empresa" es comparable entre años.
- Umbral de volumen: **≥3 frames en el año** (más bajo que el ≥5 pooled,
  igual que en la sección de evolución temporal, para no perder los
  años tempranos con menos data).
- Guardado en `data/processed/clusters/firm_year_archetype_behaviors.parquet`
  — 1.229 filas (429 empresas × hasta 6 años, 2021-2026), 34 columnas:
  arquetipo + distancia al centroide, las 9 métricas base, 14
  `behavior_share_*` (concepts de comportamiento de la sección
  anterior), 3 `domain_share_*`, y `n_entity_mentions`/`entities_named`.
- Es un archivo **derivado**, no una nueva extracción LLM — se
  reconstruye desde `gold_ai_frames`/`gold_ai_entity_mentions` ya
  existentes, sin costo adicional de API.

```python
# Construcción completa del panel (resumen; script completo similar
# al usado para la tabla de arquetipos pooled más arriba)
import duckdb, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

con = duckdb.connect('duckdb/thesis.duckdb', read_only=True)
frames = con.execute('''
    SELECT fm.ticker, fm.cik, extract(year from fm.filing_date)::INT AS year, f.*
    FROM gold_ai_frames f JOIN filing_manifest fm USING (country_code, accession_number)
    WHERE f.country_code='us' AND f.has_frame
''').fetchdf()
# ... deriva specificity_index, quantified_rate, promotional_rate, strategic_rate,
#     realized_share, hypothetical_share, risk_share, gov_share, firm_subject_share
#     igual que en la sección de arquetipos pooled ...

# 1. entrena centroides sobre el pool (>=5 frames/empresa, todos los años)
scaler = StandardScaler().fit(pop[FEAT_COLS])
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaler.transform(pop[FEAT_COLS]))

# 2. agrega por (ticker, year) con umbral >=3 frames/año y proyecta sobre esos centroides
fy_pop = fy[fy['n_frames'] >= 3]
dists = km.transform(scaler.transform(fy_pop[FEAT_COLS]))
fy_pop['cluster'] = dists.argmin(axis=1)          # arquetipo asignado
fy_pop['archetype_dist'] = dists.min(axis=1)       # confianza de la asignación

# 3. behavior_share_* = frecuencia de cada concept de comportamiento, por (ticker, year)
exp = frames[['ticker','year','concepts']].explode('concepts')
beh = pd.crosstab([exp['ticker'], exp['year']], exp['concepts'])
beh_share = beh.div(frames.groupby(['ticker','year']).size(), axis=0)

# panel final = merge de arquetipo + behavior_share_* + domain_share_* + entity mentions
panel.to_parquet('data/processed/clusters/firm_year_archetype_behaviors.parquet')
```

### Persistencia y transiciones de arquetipo

De 429 empresas, 335 (78%) aparecen en ≥2 años del panel — suficiente
para mirar transiciones año-a-año (800 pares consecutivos).

```python
p = pd.read_parquet('data/processed/clusters/firm_year_archetype_behaviors.parquet')
p = p.sort_values(['ticker','year'])
p['prev_archetype'] = p.groupby('ticker')['archetype'].shift(1)
transitions = p.dropna(subset=['prev_archetype'])
tm = pd.crosstab(transitions['prev_archetype'], transitions['archetype'],
                  normalize='index') * 100
```

| De \ A | A cauteloso | B genérico | C cuantificador | D vocal |
|---|---|---|---|---|
| **A cauteloso** | 53,4% | 39,7% | 1,1% | 5,7% |
| **B genérico** | 15,3% | 70,1% | 1,3% | 13,4% |
| **C cuantificador** | 13,9% | 11,1% | 27,8% | 47,2% |
| **D vocal** | 4,0% | 17,0% | 2,9% | **76,1%** |

Persistencia agregada: 66,6% de los pares año-a-año se quedan en el
mismo arquetipo. Pero muy desigual:
- **D es el más "pegajoso" con diferencia** (76% se queda) — no es un
  evento de un año, las empresas que llegan a D tienden a quedarse.
- **A es el menos estable** (53% se queda; 40% sube a B) — el
  boilerplate de riesgo genérico funciona más como peldaño hacia
  "adoptante genérico" que como estado permanente.
- **C es casi un cluster de tránsito** (solo 28% se queda; cuando sale,
  47% termina en D) — refuerza la nota de la sección de arquetipos de
  que C (n=7) es sensible a semilla/año, no un grupo estructural firme.

**El núcleo de D es estable, no volátil**: de las 145 firma-años en D,
87 (60%) son empresas que aparecen en D 2+ años — el "núcleo duro"
(NVDA, MSFT, GOOGL, INTC, ADBE, SNOW nunca salen de D en los 6 años del
panel) — y 58 (40%) son apariciones de un solo año, probablemente picos
puntuales de retórica/especificidad sin sostenerlos.

Un grupo chico de empresas circula por 3 arquetipos distintos en el
período (BKR, NFLX, DLR, SWK, J, DE, KLAC, LDOS, LUV, BKNG) — ejemplo,
BKR: `C`(2021) → `D`(2022-24) → `B`(2025) → `A`(2026). Son los casos
más interesantes para lectura cualitativa frame-por-frame: el arquetipo
no es un rasgo fijo de la empresa, se mueve con el ciclo de hype.

### Estrategias: qué hacen las empresas, no solo cómo lo cuentan

Usando `domain_share_*` y `behavior_share_*` del último año observado
por empresa (n=429 "instantáneas" de estrategia actual):

```python
latest = p.sort_values('year').groupby('ticker').tail(1)
latest['product_focus'] = latest['domain_share_customer_facing'] > latest['domain_share_internal']
pd.crosstab(latest['archetype'], latest['product_focus'], normalize='index')
```

**1. Producto vs. uso interno — casi enteramente explicado por el
arquetipo.** Solo 86 de 429 empresas (20%) son "product-focused"
(más frames `customer_facing` que `internal`):

| Arquetipo | % product-focused |
|---|---|
| A cauteloso | 3% |
| B genérico | 11% |
| C cuantificador | 40% |
| D vocal | **56%** |

**2. Build vs. buy — incluso D compra más de lo que construye, en
promedio.** `proprietary_ai` vs. `third_party_ai`, share medio por
arquetipo:

| Arquetipo | IA propia | IA de terceros |
|---|---|---|
| A | 0,3% | 1,9% |
| B | 1,0% | 2,7% |
| C | **4,4%** | 2,9% |
| D | 3,5% | 4,0% |

D es el único donde "propia" se acerca a "terceros", pero no le gana en
promedio: incluso los líderes vocales integran más de lo que
construyen. Constructores puros 2024-26 (`build_minus_buy` más alto):
CVS, VTR, APTV, TRMB, TT, TSLA, SLB, QCOM. Integradores puros
(más negativo): CE, WAB, CARR, PVH, AFL, VLO, DOW.

```python
recent = p[p['year'] >= 2024]
recent['build_minus_buy'] = recent['behavior_share_proprietary_ai'] - recent['behavior_share_third_party_ai']
recent.groupby('ticker')['build_minus_buy'].mean().sort_values(ascending=False)
```

**3. Qué resultado enfatizan al hablar de IA — otra vez separado por
arquetipo, no por tamaño ni sector:**

```python
outcome_cols = ['behavior_share_productivity_outcome','behavior_share_revenue_outcome',
                'behavior_share_cost_outcome','behavior_share_customer_outcome']
latest['dominant_outcome'] = latest[outcome_cols].idxmax(axis=1)
latest.loc[latest[outcome_cols].sum(axis=1) == 0, 'dominant_outcome'] = 'none'
pd.crosstab(latest['archetype'], latest['dominant_outcome'], normalize='index')
```

| Arquetipo | Productividad | Revenue | Costo | Cliente | Sin outcome claro |
|---|---|---|---|---|---|
| A | 17% | 9% | 15% | 1% | **58%** |
| B | 36% | 14% | 12% | 2% | 37% |
| C | 30% | **30%** | 20% | 0% | 20% |
| D | **69%** | 18% | 3% | 3% | 8% |

D casi siempre enmarca IA como productividad/eficiencia propia (69%),
casi nunca como ahorro de costos (3%) — vende "hacer más", no "gastar
menos". C es el único arquetipo donde revenue empata con productividad,
consistente con su historia ya establecida de "IA como demanda
externa" (§ Comportamiento por arquetipo). A no tiene narrativa: más de
la mitad de sus menciones no aterrizan en ningún resultado, solo listan
el riesgo.

**4. La intensidad de "deployed" cae con el tiempo incluso DENTRO de
D**, no solo en el agregado general (que ya sabíamos que caía por
dilución compositiva — § Evolución temporal):

```python
p.groupby(['year','archetype'])['behavior_share_deployed'].mean().unstack()
```

| Año | D vocal (`behavior_share_deployed`) |
|---|---|
| 2021 | 58% |
| 2022 | 61% |
| 2023 | 58% |
| 2024 | 45% |
| 2025 | 42% |
| 2026 | 39% |

Esto es más fuerte que la dilución compositiva: incluso mirando SOLO
las empresas que siguen siendo D, la proporción de frames de despliegue
concreto cae según crece el volumen total de menciones de IA — hablan
más de IA en general, pero proporcionalmente menos de "ya está
funcionando en producción".

**5. El discurso de infraestructura de IA ya no es solo de tech.** Top
10 por `behavior_share_ai_infrastructure` 2024-26 (≥5 frames/año
promedio): EXPD (logística/freight), DOV (industrial), MRNA (farma),
TGT (retail), DLR (data centers), BKR (energía), TPR, AMZN, KEYS, ODFL.
La narrativa de "invertimos en infraestructura de IA" se difundió fuera
del núcleo de semiconductores/software hacia logística, industrial y
farma — coherente con el hallazgo de C (§ Comportamiento por
arquetipo) de que gran parte de la conversación de infraestructura es
sobre demanda de terceros, no capacidad propia.

```python
recent.groupby('ticker').agg(
    mean_infra=('behavior_share_ai_infrastructure','mean'),
    n=('n_frames','mean'),
).query('n >= 5').sort_values('mean_infra', ascending=False).head(10)
```

### Notas metodológicas (panel empresa-año)

- `archetype_dist` (distancia al centroide asignado) no se usa para
  filtrar en ninguna de las tablas de arriba — vale la pena repetir el
  análisis de transiciones excluyendo asignaciones de baja confianza
  (distancia alta) antes de tratar la matriz de transición como
  definitiva.
- "Constructor puro" / "integrador puro" son EXTREMOS del ranking
  `build_minus_buy`, no una clasificación binaria de toda la
  población — la mayoría de empresas está cerca de 0 (ni construyen ni
  integran de forma verbalmente distintiva).
- El `dominant_outcome` de una empresa con 0 frames de cualquier
  outcome se marca `'none'` explícitamente (no se excluye) — por eso
  A tiene 58% en `'none'`: no es dato faltante, es el hallazgo (A no
  narra resultados).
- Igual que el resto del documento: solo EE.UU., sin ponderar por
  `inclusion_weight`, población filtrada por el prefiltro.

## SEC 2024 y DeepSeek: ¿cambió la TENDENCIA, no el nivel?

Las preguntas extendidas de la tesis (`docs/thesis_proposal.md`) piden
comparar empresas con divulgación pre-2024 vaga vs. específica,
antes/después del escrutinio SEC, y ver si DeepSeek afectó el framing.
Ya se había descartado la lectura ingenua de "evento discreto" (§
Evolución temporal: no hay quiebre visible en el corte anual). Este
análisis prueba la versión correcta de la pregunta: ¿cambió la
**pendiente** de la tendencia — no el nivel — alrededor de esas fechas,
y ese cambio es distinto entre empresas que ya divulgaban de forma vaga
vs. las que ya eran específicas?

### Diseño

- **Serie temporal**: `filing_manifest_10q` (2.898 filings 10-Q, no
  `filing_manifest` de 10-K) — trimestral, no anual, para tener
  resolución suficiente alrededor de fechas puntuales (marzo 2024, enero
  2025). Consistente con el resto del proyecto: 10-K y 10-Q **nunca se
  poolean** — el panel firma-año de la sección anterior usa solo 10-K;
  esta sección usa 10-Q exclusivamente como "serie de shock" separada.
- **Grupos (Treatment/Control)**: clasificados usando SOLO frames de
  10-K **pre-2024** (2021-2023), para evitar circularidad con la propia
  serie de 10-Q que se está testeando. `vagueness = z(promotional_rate)
  - z(specificity_index)` por empresa; split en la mediana. 163 empresas
  con ≥3 frames 10-K pre-2024 (mínimo para clasificar) → 81
  `Treat_vague` (promocional 13,0%, especificidad 0,113) vs. 82
  `Control_specific` (promocional 3,6%, especificidad 0,243).
- **Regresión segmentada** (no un solo salto): `y_t = β0 + β1·t + β2·post
  + β3·(t·post)`, por trimestre y por grupo, ponderada por volumen de
  frames ese trimestre. `β3` = cambio de pendiente en el corte. El
  "DiD de tendencia" es `β3(Treat) - β3(Control)` — ¿el quiebre de
  pendiente es distinto entre grupos, no solo si cada grupo tiene un
  quiebre?
- **Cortes**: SEC → 2024-04-01 (el escrutinio se hizo público el 18 de
  marzo 2024; el trimestre 2024-01 mezcla filings pre y post-anuncio,
  se descarta como corte limpio). DeepSeek → 2025-01-01 (R1 se lanzó el
  20 de enero 2025; trimestre ambiguo pero la mayoría de filings de ese
  trimestre caen después del lanzamiento).

```python
import duckdb, pandas as pd, numpy as np
con = duckdb.connect('duckdb/thesis.duckdb', read_only=True)

# grupo Treatment/Control, SOLO 10-K pre-2024
pre = con.execute('''
    SELECT fm.ticker, f.* FROM gold_ai_frames f
    JOIN filing_manifest fm USING (country_code, accession_number)
    WHERE f.country_code='us' AND f.has_frame AND f.form='10-K'
      AND fm.filing_date < DATE '2024-01-01'
''').fetchdf()
# ... specificity_index, promotional_rate por empresa; vagueness = z(promo) - z(spec)
# ... split en la mediana -> group_map: ticker -> 'Treat_vague' | 'Control_specific'

# serie de shock, SOLO 10-Q, todos los años
q = con.execute('''
    SELECT fm.ticker, fm.filing_date, f.* FROM gold_ai_frames f
    JOIN filing_manifest_10q fm USING (country_code, accession_number)
    WHERE f.country_code='us' AND f.has_frame
''').fetchdf()
q['group'] = q['ticker'].map(group_map)
panel = q.dropna(subset=['group']).groupby(['quarter','group']).agg(...)

def segmented_reg(sub, cutoff_t, ycol, wcol='n'):
    # OLS ponderado: y = b0 + b1*t + b2*post + b3*t*post
    ...  # ver script completo en el historial de la sesión
```

### Resultados: SEC 2024 (corte en 2024-Q2)

| Métrica | Grupo | Pendiente pre | Cambio de pendiente | t |
|---|---|---|---|---|
| `specificity_index` | Control (específico) | −0,0044/trim | **+0,0094/trim** | **2,32** |
| `specificity_index` | Treat (vago) | +0,0048/trim | −0,0019/trim | −0,85 |
| | **DiD (Treat−Control)** | | **−0,0114** | **−2,44** |
| `risk_share` | Control (específico) | **+0,0145/trim** | **−0,0150/trim** | **−2,94** |
| `risk_share` | Treat (vago) | +0,0022/trim | +0,0003/trim | 0,07 |
| | **DiD (Treat−Control)** | | **+0,0153** | **2,30** |
| `promotional_rate` | ambos grupos | — | sin cambio significativo | <1,4 |
| `hypothetical_share` | ambos grupos | — | ambos bajan, sin diferencia entre grupos | ~1,2 (DiD) |

### Resultados: DeepSeek (corte en 2025-Q1)

| Métrica | Grupo | Pendiente pre | Cambio de pendiente | t |
|---|---|---|---|---|
| `specificity_index` | Control (específico) | −0,0079/trim | **+0,0170/trim** | **3,57** |
| `specificity_index` | Treat (vago) | +0,0012/trim | +0,0038/trim | 1,23 |
| | **DiD (Treat−Control)** | | **−0,0132** | **−2,34** |
| `risk_share` | Control (específico) | +0,0127/trim | −0,0092/trim | −1,61 |
| `risk_share` | Treat (vago) | +0,0032/trim | +0,0054/trim | 1,18 |
| | **DiD (Treat−Control)** | | +0,0146 | 1,99 (límite) |
| `hypothetical_share` | ambos grupos | — | ambos bajan fuerte, sin diferencia entre grupos | ~0,0 (DiD) |
| `promotional_rate` | ambos grupos | — | sin cambio significativo | <1,2 |

### Lectura

**Sí hay quiebre de tendencia detectable en ambas fechas — pero está
en el grupo "equivocado".** La hipótesis de la propuesta (empresas
vagas se vuelven más específicas después del escrutinio) no se
sostiene: el grupo `Treat_vague` no muestra cambio significativo de
pendiente en especificidad ni en riesgo en ninguno de los dos cortes.
El que sí reacciona, con fuerza y en ambos eventos, es
`Control_specific` — las empresas que YA eran concretas antes de 2024:

- Su `specificity_index` **acelera bruscamente** justo después de
  marzo 2024 Y otra vez después de enero 2025 (pendiente pasa de
  negativa/plana a claramente positiva, t≈2,3-3,6 en ambos cortes) —
  se vuelven MÁS concretas todavía, no al revés.
- Su `risk_share`, que venía subiendo con fuerza desde 2021
  (+0,0145/trimestre), **se frena en seco justo en el corte SEC**
  (t=−2,94) — dejan de agregar boilerplate de riesgo al mismo ritmo.
- El grupo vago, en cambio, apenas se mueve en estas dos dimensiones.
  Sí baja su `hypothetical_share` en ambos cortes, pero el grupo
  específico baja lo mismo — no es una respuesta diferencial al evento,
  parece ser una tendencia de fondo compartida (menos lenguaje
  condicional con el tiempo, ya visto en la tabla temporal agregada).

**Interpretación tentativa**: el escrutinio regulatorio y el shock
DeepSeek no parecen haber "correjido" a las empresas más vagas — a las
36-48 meses de datos disponibles, esas empresas simplemente no
reaccionan de forma medible en estas métricas. Quien sí reacciona es el
grupo que ya tenía más que perder por precisión (más específico → más
verificable → más expuesto si una afirmación concreta resulta falsa):
ante el escrutinio, redoblan la especificidad (defensa por evidencia) y
frenan la acumulación de riesgo genérico (quizás porque ya tienen
suficiente lenguaje de riesgo acumulado, o porque el riesgo específico
reemplaza al genérico — no distinguible con `risk_share` solo). Esto
conecta con el hallazgo de la sección de arquetipos: D (líder vocal, el
más cercano a "específico + promocional a la vez") es también, por
construcción, más parecido a `Control_specific` que a `Treat_vague` —
son quienes tienen más frames de sustancia real, y son quienes se
mueven cuando el escrutinio aumenta.

### Notas metodológicas y limitaciones (importante leer antes de citar)

- **Dos cortes muy cercanos en el tiempo** (2024-Q2 y 2025-Q1, solo 3
  trimestres de separación) — la "pendiente pre" de DeepSeek incluye el
  período post-SEC, así que el pre-trend de DeepSeek YA refleja el
  quiebre de SEC. Estos son dos modelos de dos tramos ajustados por
  separado, no un modelo conjunto de 3 regímenes — un próximo paso
  obligatorio antes de reportar esto como resultado firme es un modelo
  único con dos quiebres simultáneos (o un evento-study propiamente
  dicho) para no confundir el efecto SEC con el efecto DeepSeek.
- **Regresión sobre medias trimestrales por grupo, no panel a nivel
  empresa** — no hay efectos fijos de empresa ni errores estándar
  clusterizados por firma; los t-stats son indicativos de magnitud de
  señal, no inferencia causal rigurosa. Antes de citar esto como
  resultado de tesis, correr el mismo diseño como panel firma-trimestre
  con SE clusterizados.
- **163 de ~421+ empresas clasificables** (≥3 frames 10-K pre-2024) —
  el split de vagueness es más ruidoso en la cola de empresas con pocos
  frames tempranos; repetir con un umbral más alto como check de
  robustez.
- El corte de DeepSeek (2025-Q1) es el más débil de los dos: el
  trimestre mezcla filings de antes y después del 20 de enero, y el
  volumen 10-Q post-corte todavía es parcial (datos hasta 2026-Q3).
- Igual que el resto del documento: solo EE.UU., sin ponderar por
  `inclusion_weight`.

## Cartas de comentario de la SEC: ¿hay evidencia directa de escrutinio por AI-washing?

Motivación: si el "efecto SEC 2024" de la sección anterior es real, debería
dejar rastro directo en `data/raw/sec_letters/` — 5,057 cartas de
correspondencia (UPLOAD/CORRESP) sobre 371/517 empresas (72%), cubriendo
todo tipo de correspondencia (`SEC_COMMENT`, `COMPANY_RESPONSE`,
`COMPANY_LETTER`, `REVIEW_COMPLETE`, `ACCELERATION_REQUEST`, `NO_REVIEW`).

**Búsqueda 1 — solo términos de IA** ("artificial intelligence", "generative
AI", "machine learning", "\bAI\b"): 23 cartas en 13 empresas. Al leer las
23 completas, son casi todas preguntas rutinarias de reporte de segmentos
(ASC 280-10-50) o reconocimiento de ingresos en arreglos multi-elemento,
donde el nombre del segmento/producto de la empresa simplemente contiene
"AI" (p.ej. "Cloud and AI Titans" de Arista, "HPC & AI" de HPE, "NVIDIA AI
cloud services"). Ninguna de esas tres (ANET jul-ago 2024, HPE 2023, NVDA
2023) cuestiona si las afirmaciones de IA de la empresa son exageradas o
falsas — es contabilidad, no escrutinio de disclosure.

**Búsqueda 2 — término de IA + frase de escepticismo típica de la SEC**
("tell us[,] and revise... to discuss", "provide support for", "quantify
the impact/benefit", "substantiate your claim", etc.): 15 candidatos. 14
son falsos positivos al leerlos (ANET/HPE/NVDA de la búsqueda 1 repetidos;
DIS 2024-02-27 es sobre materiales de solicitación de **Blackwells
Capital**, un activista en una pelea por proxy, no disclosure de Disney;
META 2022-11-21 tiene "AI" y "quantify" en párrafos distintos del mismo
comentario, sin relación; PFE y CCI son "AI" como substring de otra
palabra).

**El único caso real: Welltower (WELL), abril 2025.** Carta sobre el 10-K
FY2024, Item 1 Business:

> "Please tell us, and revise future filings, to discuss your data science
> platform and the status of any AI-integration efforts. In this regard, we
> note your disclosure on page 40 that you are integrating generative AI
> tools into your systems. We also note discussion of your data science
> platform during the conference call... including that it is
> 'industry-leading' and one of the sources of your competitive advantage;
> and in the March 7, 2025 press release..."

Este es el patrón exacto de la hipótesis: lenguaje promocional de IA en
canales no regulados (earnings call, press release: "industry-leading",
"competitive advantage") sin respaldo proporcional en el 10-K. Welltower
respondió con contenido nuevo y sustantivo (equipo de datos desde 2016,
Chief Data Officer desde 2023, GenAI para contratos/leases desde 2023,
chatbots internos) y se comprometió a "revise future filings to include
additional disclosures about these matters."

**Interpretación.** No es un hallazgo nulo: el mecanismo hipotetizado
existe y es identificable (1 caso limpio, encontrado tanto por lectura
manual como por el patrón léxico ampliado). Es, sí, un evento de baja
frecuencia en este canal — 1 de 517 empresas, 1 de 5,057 cartas — lo cual
es consistente con cómo funciona el proceso de revisión de la SEC: solo
una fracción de los 10-K se revisa cada ciclo, y un comentario explícito
solo se emite cuando el staff ya sospecha algo concreto (no es un barrido
sistemático de todas las menciones de IA). Que la búsqueda ampliada NO
haya producido docenas de casos adicionales es en sí una señal de
que el patrón (14/15 falsos positivos) no está siendo sobre-ajustado al
regex — un hallazgo de "encontramos exactamente 1 caso de manual" es más
creíble que uno de "encontramos 50".

**Implicación para la tesis:** las cartas de comentario no son un canal de
alto volumen para medir AI-washing sistemáticamente (demasiado raras), pero
sirven como *validación de caso* de que el mecanismo existe, y el patrón
Welltower (frase promocional en canal no regulado + ausencia/insuficiencia
en el 10-K) es un candidato razonable de regla heurística para escanear
transcripciones de earnings calls o press releases contra el `paragraphs`
de 10-K de la misma empresa/período — un análisis pendiente, no ejecutado
en este documento.
