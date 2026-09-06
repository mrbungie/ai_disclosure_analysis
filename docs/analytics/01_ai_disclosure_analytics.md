# Analytics preliminares de divulgación de IA (EE.UU.)

10 preguntas cortas, resueltas con SQL simple sobre `gold_ai_frames` /
`gold_ai_entity_mentions` (`duckdb/thesis.duckdb`) — ver
`docs/prefilter_evaluation.md` §8.8-§8.13 para cómo se construyeron esas
tablas. Todas las consultas están en la sección final para reproducirlas.

Alcance: **solo EE.UU.** (Chile todavía no tiene embeddings/scoring —
ver §8.9). Población: **17.266 textos únicos** marcados IA-relevantes con
al menos un frame, **28.643 frames** semánticos extraídos vía
`ai_classify.py` (qwen/qwen3.7-flash), sobre un corpus prefiltrado de
4.316.284 textos únicos (19.698 candidatos, 22.481 instancias).

> **Población actualizada 2026-09-06 (cifras de abajo pendientes de re-correr).**
> El prefiltro se reajustó con un solo juez y `gold_ai_frames` dejó de acumular
> la unión histórica de despliegues (`prefilter_evaluation.md` §8.15): la
> población pasó a **19.717 textos marcados / 24.141 instancias de frame**
> (antes 19.698 / 24.328). Las tablas de este documento son de la corrida
> anterior; el cambio es de −0,8% en frames, así que sirven como aproximación
> pero no como cifra final.

**Actualizado 2026-09-05 con DEF 14A y 8-K.** La versión anterior de este
documento cubría 10-K y 10-Q solamente (13.442 textos, 22.622 frames). El
modelo del prefiltro NO se reentrenó: se aplicaron los coeficientes,
intercepto y threshold (0,75) del despliegue `20260904T160927Z` tal cual,
vía `ai_prefilter_classify.py --apply-only`, para que la regla de decisión
sea idéntica entre formularios y las cifras viejas de 10-K/10-Q sigan
siendo las mismas (verificado: `max |Δ predicted_proba| = 0` sobre las
1.650.145 filas del corpus anterior).

Composición de la población por formulario:

| Formulario | Textos únicos con frame | Frames |
|---|---:|---:|
| 10-K | 10.378 | 17.533 |
| DEF 14A | 4.621 | 6.533 |
| 10-Q | 2.359 | 4.315 |
| 8-K | 194 | 262 |

> **Bug de reproducibilidad corregido en el camino (2026-09-05).** Al
> intentar re-correr estos análisis aparecieron números distintos en cada
> consulta sobre las MISMAS filas: el conteo de frames promocionales
> oscilaba entre 2.903 y 2.911. La causa estaba en la vista
> `gold_ai_frames` de `build_duckdb.py`, que deduplicaba con
> `QUALIFY row_number() OVER (PARTITION BY text_hash, frame_index ORDER BY
> session_id DESC)`. Un mismo texto se clasifica varias veces dentro de una
> sesión —instancias de párrafo distintas comparten `text_hash`— así que
> había 1.225 grupos empatados, y en 976 de ellos las filas empatadas
> traían valores DISTINTOS: `row_number()` elegía una al azar en cada
> consulta. Además, particionar por `frame_index` podía quedarse con el
> frame 0 de una llamada al juez y el frame 1 de otra, mezclando dos
> lecturas del mismo párrafo.
>
> Corregido deduplicando por LLAMADA (`PARTITION BY text_hash ORDER BY
> session_id DESC, classified_at DESC`, trayendo todos los frames de la
> llamada ganadora). Verificado: tres consultas seguidas dan cifras
> idénticas, y re-correr los scripts de clustering produce parquets
> byte-idénticos. **Todas las cifras de este documento son posteriores al
> arreglo.** Las de la versión anterior no eran reproducibles, así que
> cualquier comparación con ellas tiene un margen de ±1% por esta causa
> además de los cambios reales de población.

## Advertencia de comparabilidad entre formularios

> **Medido 2026-09-06 (`docs/prefilter_evaluation.md` §8.15).** El prefiltro se
> ajustó y evaluó SÓLO con 10-K y 10-Q, y se aplicó sin revalidar a DEF 14A y
> 8-K. Sobre una muestra de 1.500 párrafos de esos dos formularios etiquetada
> con el mismo juez: F1 ponderado **0,755 en DEF 14A** y **0,647 en 8-K**,
> contra 0,925 out-of-fold en 10-K/10-Q. La precisión cae de 0,88 a 0,68 y
> 0,54. Es decir: **el instrumento no mide igual en los cuatro formularios**, y
> toda comparación entre ellos —incluida la #8, la más citada de este
> documento— mezcla diferencia de discurso con diferencia de error de medición.


Casi cualquier conteo agregado sobre estos cuatro formularios mezcla dos
cosas distintas: **intensidad de divulgación** y **volumen documental**.
Una empresa presenta un 10-K al año y una DEF 14A al año, pero muchos 8-K.
Por eso las preguntas sensibles a eso (#1, #8, #9) se reportan
desglosadas por `form`, no pooled — el agregado existe, pero no se lee
como "las empresas hablan más de IA".

## Resultados

| # | Pregunta | Hallazgo |
|---|---|---|
| 1 | ¿Cómo evoluciona la cantidad de párrafos con mención de IA por año de filing? | Agregado (10-K + DEF 14A + 8-K): 690 (2021) → 824 (2022) → 1.149 (2023) → 2.988 (2024) → 4.613 (2025) → 5.993 (2026, parcial). El salto 2023→2024 (+160%) se mantiene con los formularios nuevos y sigue coincidiendo con la ola de IA generativa post-ChatGPT. Desglosado, la trayectoria es la misma en 10-K (511→4.000) y DEF 14A (175→1.933); el 8-K aporta poco y tarde (4 en 2021, 70 en 2026). |
| 2 | ¿Qué sectores (SIC 2 dígitos) tienen mayor prevalencia de empresas con divulgación de IA? | SIC 73 (software/servicios de cómputo): 66/66 empresas (100%). SIC 63 (seguros): 23/23 (100%). SIC 49 (utilities): 33/34 (97%). SIC 38 (instrumentos): 40/42 (95%, era 90%). SIC 28 (químicos/farma): 37/40 (93%, era 85%). Sumar DEF 14A subió la prevalencia en casi todos los sectores: la divulgación de IA fuera del 10-K alcanza empresas que el 10-K no capturaba. |
| 3 | ¿Qué proporción de las afirmaciones sobre IA son sobre algo YA ocurrido vs. planeado/esperado/hipotético? | 71,3% `realized` (20.433), 12,7% `planned` (3.626), 8,5% `expected` (2.421), 7,6% `hypothetical` (2.163). La fracción "ya ocurrido" SUBE respecto de la versión 10-K/10-Q (67%): el contenido de proxy y 8-K es más factual y retrospectivo que el de los factores de riesgo del 10-K. |
| 4 | ¿De quién se habla cuando se habla de IA — la propia empresa, sus clientes, o la competencia? | 85,3% `firm` (24.434), 9,2% `competitors_or_industry` (2.649), 4,8% `customers` (1.389), 0,6% `suppliers_or_partners` (171). La autodescripción se acentúa (era 82%): la DEF 14A habla de lo que hace la propia empresa, casi nunca del sector. |
| 5 | ¿Qué empresas tienen más frames de IA extraídos (más "vocales" sobre IA en sus filings)? | MSFT (595), GOOGL (561), NVDA (549), INTC (490), ADBE (486), AMZN (436), CRM (348), SNOW (316), META (308), HPE (294), WDAY (291), CTSH (275), AMD (272), PANW (263), NOW (250). El orden cambia respecto de la versión 10-K/10-Q (NVDA lideraba): MSFT y GOOGL suben al incorporar sus proxies. |
| 6 | ¿Cuáles son los riesgos de IA más mencionados? | Ciberseguridad (3.019), regulatorio/legal (2.765), competitivo/disrupción (1.713), confiabilidad/precisión (1.590), dependencia operacional (1.198), privacidad (1.154), propiedad intelectual (925), sesgo/equidad (664), fuerza laboral (409). El orden no cambia respecto de la versión anterior — los formularios nuevos agregan volumen, no un perfil de riesgo distinto. |
| 7 | ¿La IA se usa más internamente o de cara al cliente? | 45,9% interno (13.161), 38,1% customer-facing (10.899), 16,0% sin especificar (4.583). La brecha se abre respecto del casi-empate anterior (44%/43%): DEF 14A y 8-K hablan más de uso interno (gobernanza, operaciones, compensación ligada a IA) que de producto. |
| 8 | ¿Qué fracción de las afirmaciones sobre IA usa lenguaje promocional/superlativo? | 9,8% agregado (2.808 de 28.643) — pero el agregado esconde el hallazgo real: **DEF 14A 16,3%** (1.064/6.533) vs. **10-K 7,1%** (1.246/17.533), con 10-Q en 11,1% y 8-K en 8,0%. El proxy statement, que se dirige a accionistas y no al regulador, es más del doble de promocional sobre IA que el 10-K. Es la señal más directa de AI-washing que produjo esta actualización. |
| 9 | ¿Cómo evoluciona generativa vs. predictiva/ML clásica en el tiempo? | 2021: 160 gen / 172 predictiva (predictiva domina). 2023: 437 / 223 (cruce). 2026: 2.726 gen / 210 predictiva. La generativa pasa de minoría a ~93% de las menciones con tipo especificado; el cruce sigue cayendo entre 2022 y 2023, igual que en la versión anterior. |
| 10 | ¿Qué entidades de IA específicas se nombran más, y de dónde son? | Con frame de IA confirmado: OpenAI (105), Copilot (100), Gemini (56), Anthropic (35), ChatGPT (30), watsonx (28), Vertex AI (16). No estadounidenses: Cohere (5), DeepSeek (4), Stable Diffusion (3), Mistral AI (1), Stability AI (1), Qwen (1). El "AI-washing" nombrado sigue siendo casi exclusivamente sobre proveedores estadounidenses. **Ver la advertencia de falsos positivos abajo — esta pregunta cambió de consulta.** |

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
  rescates de esta corrida son de DEF 14A.
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
  otras seis. Esto ya era así en la versión anterior de este documento,
  donde la fila "evolución temporal" decía cubrir todos los frames pero
  de hecho era 10-K puro (512/578/868/2.184/3.316/4.018 — exactamente la
  columna 10-K de la consulta 1b de hoy).
- Ninguna de estas cifras está ponderada por `inclusion_weight` — son
  conteos directos sobre la población ya filtrada por el prefiltro, no
  estimaciones de prevalencia del corpus completo (para eso ver
  docs/prefilter_evaluation.md §8.9's estimador de `stage3_random`).

---

> **Qué se recalculó y qué no.** Los arquetipos de voz, la evolución por
> composición, los comportamientos por arquetipo y el panel empresa-año
> SÍ se recalcularon con DEF 14A y 8-K, vía el script
> `scripts/analytics/build_firm_clusters.py` reconstruido para esto.
>
> Las dos últimas secciones (cuasi-experimento SEC 2024 / DeepSeek, y
> cartas de comentario) **no se recalcularon, y sus cifras quedan
> desactualizadas por una razón distinta a las demás.** Su diseño es
> deliberadamente mono-formulario —forma grupos con 10-K pre-2024 y mide
> el shock sobre la serie trimestral de 10-Q— así que sumarle DEF 14A y
> 8-K no lo mejoraría, lo rompería: un 8-K no tiene periodicidad
> trimestral y una DEF 14A sigue el calendario de la junta.
>
> Pero **sí las afecta el arreglo del bug de deduplicación** descrito
> arriba: los frames de 10-K pasaron de 17.969 a 17.533 y los de 10-Q de
> 4.653 a 4.315 al dejar de mezclar llamadas distintas del juez. Es un
> −2,4% y −7,3% respectivamente, así que las cifras de esas dos secciones
> hay que tratarlas como aproximadas hasta que se re-corran. Recorrerlas
> es trabajo pendiente: su código tampoco está versionado.


# Arquetipos de voz (EE.UU.)

Primera exploración de la pregunta central de la tesis
(`docs/thesis_proposal.md`): ¿en qué arquetipos distintos se agrupan las
empresas según CÓMO divulgan IA, no solo cuánto? Se agregó cada empresa
(ticker) a partir de sus frames de `gold_ai_frames` en 9 métricas, y se
agruparon con K-means.

**Recalculado 2026-09-05 con DEF 14A y 8-K** vía
`scripts/analytics/build_firm_clusters.py`. Ese script no existía: los
parquets de `data/processed/clusters/` se habían producido ad hoc y sólo
sobrevivieron los outputs, así que se reconstruyó desde la prosa de este
documento y de `06_voice_vs_behavior_clustering.md`. **No reproduce las
cifras anteriores exactamente**, y no debería: K-means re-ajusta sus
centroides sobre una población distinta. Los parquets viejos están en
`data/archive/processed/clusters/` con su `POINTER.json`.

Se llamaban "arquetipos de comportamiento". Es un nombre equivocado que
`06_...md` ya había señalado: las 9 métricas son todas retórica. Acá
quedan renombrados a **arquetipos de voz**, que es lo que miden.

## Método

- **Población**: 446 empresas de EE.UU. con ≥5 frames (filtro de volumen
  mínimo para que el perfil agregado no sea ruido de 1-2 menciones),
  sobre 24.328 frames — 17.533 de 10-K, 6.533 de DEF 14A, 262 de 8-K.
- **El 10-Q no entra.** El JOIN va contra `filing_manifest`, que por
  decisión de alcance del proyecto no lo incluye (instrumento separado,
  con su propio `filing_manifest_10q`). Esto ya era así antes de esta
  actualización: la versión anterior decía cubrir 13.442 textos pero
  clusterizaba sólo los frames de 10-K.
- **Métricas por empresa** (promedio sobre sus frames):
  - `specificity_index`: promedio de las 5 banderas de especificidad
    (proceso de negocio, producto/sistema, proveedor, métrica
    cuantificada, fecha/cronograma) — "qué tan concreto" es el discurso.
  - `quantified_rate`: fracción de frames con una métrica numérica
    explícita (subconjunto de lo anterior, aislado porque es el
    indicador más directo de "sustancia" vs. "promesa vacía").
  - `promotional_rate` / `strategic_rate`: señales retóricas.
  - `realized_share` / `hypothetical_share`: fracción de frames en cada
    extremo de `temporal`.
  - `risk_share` / `gov_share`: fracción de frames con al menos un
    concepto `risk_*` / `gov_*`.
  - `firm_subject_share`: fracción de frames con `subject='firm'`.
- **Clustering**: K-means (k=4, `random_state=42`, `n_init=10`) sobre las
  9 métricas estandarizadas (media 0, varianza 1).
- **Las letras se asignan por perfil, no por el id de sklearn.** K-means
  renumera sus clusters en cada re-ajuste, así que el script reclama D
  por `promotional_rate`, después C por `quantified_rate`, después A por
  `risk_share`, y B es el residuo. Sin esto, un re-ajuste renombraría en
  silencio a todas las empresas del panel y de todos los docs que
  dependen de estas etiquetas.
- **Reproducible, y no lo era.** `load_frames()` lleva un `ORDER BY`
  explícito: sin él DuckDB devuelve las filas en el orden que produzca su
  escaneo paralelo, los promedios por empresa suman los mismos floats en
  distinto orden, y eso alcanzaba para mover los centroides y reetiquetar
  empresas entre dos corridas del mismo script con la misma semilla.
  Verificado: dos corridas seguidas producen parquets byte-idénticos.

## Los 4 arquetipos

| Arquetipo | n empresas | Especificidad | % cuantificado | % promocional | % ya realizado | % riesgo | Frames/empresa | Empresas típicas |
|---|---|---|---|---|---|---|---|---|
| **A. Listadores de riesgo cautelosos** | 111 | 0,064 (mín.) | 0,6% (mín.) | 1,6% (mín.) | 48% (mín.) | 69% (máx.) | 22 | MTCH, GS, CDW, BAX, TRV, IT, PRU, IFF |
| **B. Adoptantes genéricos** | 181 | 0,107 | 1,6% | 1,9% | 68% | 46% | 37 | META, MSCI, ADSK, NWS, AXP, V, OKTA, UNH |
| **C. Cuantificadores concretos** | 35 | **0,222** (máx.) | **16,9%** (máx.) | 7,6% | 68% | 25% (mín.) | 80 | NVDA, AMD, EFX, IQV, NDAQ, TSLA, AVGO, IDXX |
| **D. Líderes vocales de IA** | 119 | 0,152 | 3,0% | **12,3%** (máx.) | **75%** (máx.) | 26% | 103 | MSFT, GOOGL, INTC, ADBE, AMZN, CRM, SNOW, HPE |

**El cluster de cuantificadores concretos sobrevive, y es el hallazgo
estructural más interesante de la actualización.** En la versión anterior
C eran 7 empresas de energía/industrial con 31% de frames cuantificados.
Hoy son 35 empresas con 16,9% — sigue siendo ~5x el resto (D, el segundo,
está en 3,0%) y sigue siendo el grupo de menor foco en riesgo. Pero su
composición cambió por completo: ya no es energía/industrial sino
semiconductores y datos (NVDA, AMD, AVGO, TSLA, NDAQ, IQV). Ampliar el
corpus no disolvió la categoría "empresas que ponen números a sus
afirmaciones sobre IA"; la llenó de otras empresas.

**A — Listadores de riesgo cautelosos** (111 empresas). Mencionan IA casi
exclusivamente como riesgo genérico futuro: 69% de sus frames son de
riesgo y 28% hipotético, con especificidad y promoción casi en cero. Es
el patrón clásico de "boilerplate de risk factors" que enumera IA junto a
otras amenazas tecnológicas sin describir uso propio. Menor riesgo de
AI-washing simplemente porque casi no hace afirmaciones positivas
verificables.

**B — Adoptantes genéricos** (181 empresas, el grupo más grande). Hablan
de IA como algo ya en curso (68% realizado) pero de forma llana: baja
especificidad, casi nada cuantificado ni promocional. El punto medio del
espectro.

**C — Cuantificadores concretos** (35 empresas). Especificidad 0,222 y
16,9% de frames con una métrica numérica explícita, ambos máximos por
lejos. Son las empresas que, cuando hablan de IA, ponen cifras. Notar que
NVDA migró de D (donde estaba en la versión anterior) a C: con más datos,
su discurso resulta más cuantificado que promocional.

**D — Líderes vocales de IA** (119 empresas, 103 frames promedio). La
retórica promocional más alta (12,3%, 1,6x la de C) y la mayor proporción
de afirmaciones sobre hechos consumados (75%), pero con un
`quantified_rate` de 3,0% — **cinco veces menor que el de C**. Ese
contraste es el resultado más útil de la sección: hablar mucho y hablar
con números son ejes distintos, y el grupo que más habla no es el que más
cuantifica. La pregunta "¿es AI-washing?" se vuelve concreta acá — D
afirma mucho y verifica poco, C verifica.

## Notas metodológicas (arquetipos)

- Exploratorio, no definitivo: k=4 es una elección razonable, no la
  única.
- **Las etiquetas son sensibles a la población.** Entre la versión 10-K y
  esta, NVDA pasó de D a C, GOOGL de C a D, y el tamaño de cada grupo se
  movió sustancialmente. Los clusters describen una partición de esta
  muestra, no categorías del dominio.
- Ninguna métrica está ponderada por tamaño de filing ni por
  `duplicate_count`. Corregirlo sigue pendiente antes de cualquier
  resultado publicable.
- No incluye Chile (todavía sin embeddings/scoring, §8.9) ni pondera por
  `inclusion_weight`.
- **Sumar DEF 14A cambia qué mide el perfil pooled de una empresa.** El
  proxy tiene 16,3% de frames promocionales contra 7,1% del 10-K, así que
  una empresa cuyo proxy es extenso se corre hacia el perfil "vocal" por
  composición documental, no porque haya cambiado su discurso.

## Evolución temporal — y por qué el agregado engaña

### La tendencia agregada (todos los frames, por año de filing)

| año | frames | % promocional | % cuantificado | especificidad | % hipotético |
|---|---|---|---|---|---|
| 2021 | 982 | 14,0% | 7,2% | 0,209 | 3,5% |
| 2022 | 1.218 | 14,9% | 7,0% | 0,205 | 3,2% |
| 2023 | 1.710 | 12,4% | 5,7% | 0,174 | 4,7% |
| 2024 | 4.464 | 10,3% | 4,1% | 0,143 | 9,0% |
| 2025 | 6.981 | 8,8% | 4,1% | 0,143 | 8,8% |
| 2026 | 8.973 | 8,1% | 4,1% | 0,135 | 9,8% |

El patrón sobrevive intacto: la retórica promocional cae casi a la mitad
(14,0%→8,1%), la especificidad también (0,209→0,135), el hipotético casi
se triplica, y el volumen se multiplica por 9.

Sería tentador leerlo como "las empresas se volvieron más cautelosas tras
el escrutinio de la SEC en 2024" — pero la caída ya viene desde 2021 sin
quiebre visible en 2024, así que esa lectura causal no se sostiene con
este solo dato.

### La explicación real: es composición, no comportamiento individual

El panel empresa-año muestra que la caída agregada es composicional: cada
año entran al corpus muchas más empresas nuevas, y las que entran tarde
entran por el lado cauteloso del espectro. D pasa de 41 de 85
empresas-año en 2021 (48%) a 93 de 402 en 2026 (23%), no porque las
empresas D se hayan callado, sino porque el denominador se llenó de A y B:

| año | A | B | C | D | total |
|---|---|---|---|---|---|
| 2021 | 4 | 17 | 23 | 41 | 85 |
| 2022 | 1 | 29 | 26 | 51 | 107 |
| 2023 | 16 | 39 | 25 | 51 | 131 |
| 2024 | 85 | 83 | 23 | 77 | 268 |
| 2025 | 107 | 151 | 26 | 86 | 370 |
| 2026 | 102 | 174 | 33 | 93 | 402 |

C es notablemente estable en términos absolutos (23-33 empresas-año en
los seis años) mientras A crece de 4 a 102. La cantidad de empresas que
cuantifican sus afirmaciones sobre IA no creció con la ola; lo que creció
es la cantidad que la menciona sin cuantificar nada.

## Comportamientos: general y por arquetipo

"Comportamiento" aquí = los `concepts` de `gold_ai_frames` que describen
etapa de uso o resultado (no los de riesgo/gobernanza, cubiertos en la
pregunta #6). Un frame puede tener 0, 1 o varios a la vez.

### Comportamiento general (24.328 frames, EE.UU.)

| Concept | n | % de frames |
|---|---|---|
| `deployed` | 8.094 | 33,3% |
| `productivity_outcome` | 2.585 | 10,6% |
| `expansion_or_scaling` | 2.236 | 9,2% |
| `revenue_outcome` | 1.178 | 4,8% |
| `third_party_ai` | 789 | 3,2% |
| `ai_investment` | 777 | 3,2% |
| `cost_outcome` | 704 | 2,9% |
| `ai_infrastructure` | 691 | 2,8% |
| `ai_talent` | 650 | 2,7% |
| `use_stage_unspecified` | 543 | 2,2% |
| `proprietary_ai` | 518 | 2,1% |
| `customer_outcome` | 510 | 2,1% |
| `exploring` | 268 | 1,1% |
| `pilot_or_testing` | 221 | 0,9% |

Un tercio de los frames (33,3%) describe IA ya desplegada, casi idéntico
al 32,7% de la versión 10-K. Piloto/exploración combinados siguen siendo
apenas 2,0%: casi nadie describe la fase temprana de adopción. Que esto
no se mueva al sumar proxies y 8-K refuerza la lectura de selección — las
empresas no reportan lo que todavía no funciona, en ningún formulario.

### Comportamiento por arquetipo (% de frames del arquetipo con el concept)

| Concept | A (cauteloso) | B (genérico) | C (cuantificador) | D (líder vocal) |
|---|---|---|---|---|
| `deployed` | 14,2% | 24,2% | **42,4%** | 40,1% |
| `productivity_outcome` | 3,0% | 7,2% | 13,0% | **13,6%** |
| `expansion_or_scaling` | 5,5% | 7,9% | **11,6%** | 10,1% |
| `revenue_outcome` | 1,3% | 3,6% | **9,5%** | 5,2% |
| `ai_investment` | 2,5% | 2,6% | 3,2% | **3,6%** |
| `ai_infrastructure` | 0,8% | 2,1% | **4,2%** | 3,3% |
| `third_party_ai` | 2,5% | 3,1% | **4,1%** | 3,3% |
| `ai_talent` | 1,4% | 2,2% | 2,9% | **3,1%** |
| `cost_outcome` | 2,2% | 2,8% | 2,9% | **3,1%** |
| `customer_outcome` | 0,6% | 1,5% | 1,7% | **2,8%** |
| `proprietary_ai` | 0,8% | 1,3% | **4,3%** | 2,4% |
| `pilot_or_testing` | 0,6% | **1,1%** | 1,0% | 0,9% |
| `exploring` | **2,3%** | 1,4% | 0,5% | 0,8% |

Frames por arquetipo: A 2.394, B 6.787, C 2.785, D 12.235.

**C supera a D en 6 de 13 conceptos**, incluidos los más sustantivos:
`deployed` (42,4% vs. 40,1%), `revenue_outcome` (9,5% vs. 5,2%),
`proprietary_ai` (4,3% vs. 2,4%) e `ai_infrastructure` (4,2% vs. 3,3%).
D gana en los conceptos más blandos (talento, costos, cliente). Es la
misma historia que el `quantified_rate`: **el grupo que más habla no es
el que más comportamiento concreto describe.**

Las dos excepciones al orden creciente siguen siendo informativas:
`exploring` va al revés (A 2,3% → D 0,8%) y `pilot_or_testing` es plano.
La voz alta no viene con más lenguaje de fase temprana, sino con menos.

### Dominio por arquetipo (%)

| Arquetipo | customer-facing | interno | sin especificar |
|---|---|---|---|
| A | 15,1% | 60,4% | 24,5% |
| B | 24,2% | 56,9% | 18,9% |
| C | **46,1%** | 39,6% | 14,3% |
| D | 44,9% | 41,7% | 13,4% |

C y D son los únicos arquetipos donde la IA de cara al cliente se acerca
al uso interno, con C ligeramente arriba. A y B hablan mayoritariamente
de IA puertas adentro.

### Menciones de entidades por arquetipo

Sobre el panel empresa-año, contando sólo menciones en textos con frame
de IA confirmado por el juez (ver la sección de falsos positivos arriba):

| Arquetipo | empresas-año | % con ≥1 mención | menciones promedio |
|---|---|---|---|
| A | 243 | 0,8% | 0,01 |
| B | 523 | 5,4% | 0,09 |
| C | 120 | 8,3% | 0,29 |
| D | 466 | **12,5%** | **0,56** |

Nombrar un proveedor concreto es raro en todo el corpus, pero 15x más
frecuente en D que en A. Acá D sí lidera sobre C — nombrar es una forma
de concreción distinta de cuantificar, y es la que los vocales prefieren.

## Panel empresa-año: transiciones de arquetipo y estrategias

Todo lo anterior es transversal (una empresa = un punto). Esta sección
baja el análisis a **empresa × año**: ¿el arquetipo es un rasgo fijo o
cambia?, y dentro de "cómo divulgan IA", ¿qué **hacen**?

### Construcción del panel

- Se recalculan las 9 métricas por `(ticker, año de filing)` en vez de
  pooladas por empresa.
- Los años-empresa se **proyectan** contra los mismos 4 centroides ya
  entrenados sobre el pool completo (no se re-clusteriza por año).
- Umbral de volumen: **≥3 frames en el año** (más bajo que el ≥5 pooled).
- `data/processed/clusters/firm_year_archetype_behaviors.parquet` —
  **1.363 filas** (454 empresas × hasta 6 años, 2021-2026), 34 columnas.
- Es un archivo **derivado**, no una extracción LLM: se reconstruye con
  `scripts/analytics/build_firm_clusters.py`, sin costo de API.

### Persistencia y transiciones de arquetipo

De 454 empresas, **366 (81%)** aparecen en ≥2 años del panel — 909 pares
consecutivos. El **56,8%** de esos pares mantiene el arquetipo.

| desde \ hacia | A | B | C | D |
|---|---|---|---|---|
| **A** | **50,7%** | 38,3% | 4,5% | 6,5% |
| **B** | 14,0% | **61,1%** | 5,3% | 19,6% |
| **C** | 8,4% | 12,6% | **43,7%** | 35,3% |
| **D** | 5,2% | 24,0% | 9,0% | **61,8%** |

El arquetipo es persistente pero no fijo. B es el más estable (61,1%) y
**C el menos (43,7%), con una fuga del 35,3% hacia D** — el flujo más
grande fuera de la diagonal en toda la matriz. Es un movimiento con
lectura clara: una empresa que un año cuantificó sus afirmaciones sobre
IA, al siguiente las hace promocionales sin números. El camino inverso
(D→C) es 9,0%, menos de un tercio. **El corpus se mueve de cuantificar a
promocionar, no al revés.**

El movimiento D→A sigue siendo el más raro de la matriz (5,2%).

### Estrategias: qué hacen las empresas, no solo cómo lo cuentan

Usando `domain_share_*` y `behavior_share_*` del último año observado por
empresa (n=443 "instantáneas" de estrategia actual):

**1. Producto vs. uso interno.** Sólo 68 de 443 empresas (15%) son
"product-focused" (más frames `customer_facing` que `internal`):

| Arquetipo | % product-focused |
|---|---|
| A cauteloso | 1% |
| B genérico | 8% |
| C cuantificador | **38%** |
| D vocal | 33% |

**2. Build vs. buy — nadie construye más de lo que integra.**
`proprietary_ai` vs. `third_party_ai`, share medio por arquetipo:

| Arquetipo | IA propia | IA de terceros |
|---|---|---|
| A | 0,7% | 2,6% |
| B | 0,7% | 2,8% |
| C | **3,1%** | 3,1% |
| D | 2,5% | 3,5% |

C es el único que empata (3,1% vs. 3,1%); el resto integra más de lo que
construye, D incluido. Constructores puros 2024-26 (`build_minus_buy` más
alto): BNY, JCI, ARE, AAPL, VTR, HPE, GWW, GD. Integradores puros: IPG,
WAB, DGX, APTV, MPC, PVH, RCL, STE.

**3. Qué resultado enfatiza cada arquetipo** (`dominant_outcome` = el
`behavior_share_*_outcome` más alto; `none` cuando no narra ninguno):

| Arquetipo | ninguno | productividad | costo | ingresos | cliente |
|---|---|---|---|---|---|
| A | **56%** | 25% | 12% | 5% | 2% |
| B | 28% | 48% | 9% | 13% | 2% |
| C | 12% | 56% | 3% | **26%** | 3% |
| D | 11% | **75%** | 4% | 9% | 1% |

La productividad domina en todos los que narran algo. La diferencia
interesante está en ingresos: **C enfatiza `revenue_outcome` en 26% de
los casos contra 9% de D**, casi 3x. D concentra su narrativa en
productividad (75%), que es el outcome más difícil de auditar
externamente; C reparte hacia ingresos, que sí aparece en los estados
financieros.

### Notas metodológicas (panel empresa-año)

- `archetype_dist` no se usa para filtrar en ninguna tabla de arriba —
  vale la pena repetir las transiciones excluyendo asignaciones de baja
  confianza antes de tratar la matriz como definitiva.
- "Constructor puro" / "integrador puro" son EXTREMOS del ranking
  `build_minus_buy`, no una clasificación binaria de la población.
- El `dominant_outcome` de una empresa sin frames de outcome se marca
  `'none'` explícitamente, no se excluye.
- **Las transiciones mezclan formularios.** Una empresa puede cambiar de
  arquetipo entre 2024 y 2025 porque cambió su discurso o porque ese año
  su DEF 14A aportó proporcionalmente más frames que su 10-K. Con 16,3%
  vs. 7,1% de retórica promocional entre formularios, el segundo efecto
  es material y el panel no lo separa. La fuga C→D de 35,3% podría ser en
  parte eso, y verificarlo es el próximo paso obvio.
- Sólo EE.UU., sin ponderar por `inclusion_weight`.

## SEC 2024 y DeepSeek: ¿cambió la TENDENCIA, no el nivel?

> **RETIRADO (2026-09-06).** Rehecho como event study empresa-trimestre con
> efectos fijos de empresa y de trimestre y errores estándar clusterizados por
> empresa (`scripts/analytics/sec_event_study.py`), **todos los outcomes fallan
> el test de tendencias paralelas**: la diferencia entre el grupo "vago" y el
> "específico" ya está presente y es significativa SEIS TRIMESTRES ANTES del
> corte, y no cambia después. Ejemplo con `specificity_index` (grupo binario):
> t−4 = −0,145\*, t−3 = −0,159\*, t−2 = −0,108\*, t+1 = −0,090\*, t+4 = −0,138\*.
>
> Eso es una diferencia permanente entre grupos —que es lo esperable, porque
> los grupos se definieron justamente por esas métricas— y no un efecto del
> evento. La lectura de abajo ("el grupo específico reacciona, el vago no") es
> un artefacto del diseño anterior: una regresión segmentada sobre medias
> trimestrales por grupo, sin efectos fijos, con la composición del grupo
> cambiando cada trimestre y con reversión a la media incorporada por
> construcción.
>
> Además el panel correcto es chico: 257 observaciones y 54 empresas, porque el
> 10-Q aporta pocos frames por trimestre. **No hay evidencia de que el
> escrutinio de la SEC de 2024 cambiara la divulgación de IA en este corpus**, y
> el diseño anterior no podía haberla detectado aunque existiera.
>
> Lo que sigue se conserva como registro de lo que se intentó.


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
