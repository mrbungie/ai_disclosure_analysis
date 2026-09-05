# Voz vs. comportamiento: dos clusterings separados, cruzados

Corrige un problema conceptual de `01_ai_disclosure_analytics.md`: los
4 "arquetipos de comportamiento" (A/B/C/D) se construyeron sobre 9
métricas que son **todas retórica/voz** — `specificity_index`,
`quantified_rate`, `promotional_rate`, `strategic_rate`,
`realized_share`, `hypothetical_share`, `risk_share`, `gov_share`,
`firm_subject_share` — ninguna es un `behavior_share_*` real
(`deployed`, `ai_investment`, `ai_infrastructure`, etc.). Esos
conceptos de comportamiento genuino se usaron después, como variable
DESCRIPTIVA de cada arquetipo ya formado — nunca como insumo del
clustering. El nombre "líder vocal" para D ya lo delataba: es vocal
(habla mucho, con cierto tono), no necesariamente "actúa mucho".

Esta sección corre un **segundo clustering, independiente, sobre los
`behavior_share_*` reales**, y lo cruza contra el arquetipo de voz —
la comparación correcta para identificar candidatos a AI-washing (voz
alta + comportamiento bajo) vs. sustancia callada (voz baja +
comportamiento alto), en vez de asumir que ambos ejes son lo mismo.

## Construcción

Mismas 421 empresas (≥5 frames, agregado por ticker, todos los años
pooled — igual que la población de `01_...md`), pero 15 features de
comportamiento en vez de las 9 de voz:

```python
BEHAVIOR_CONCEPTS = ['deployed','pilot_or_testing','exploring','ai_investment',
                     'ai_infrastructure','ai_talent','proprietary_ai','third_party_ai',
                     'expansion_or_scaling','productivity_outcome','revenue_outcome',
                     'cost_outcome','customer_outcome']  # sin use_stage_unspecified (residual)
FEAT_COLS = BEHAVIOR_CONCEPTS + ['domain_customer_facing', 'domain_internal']
X = StandardScaler().fit_transform(pop[FEAT_COLS].values)
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
```

k=4 elegido por comparabilidad directa con el arquetipo de voz (A/B/C/D),
no porque el codo/silhouette lo pidan con fuerza — silhouette entre
0,182 (k=2) y 0,190 (k=6), sin quiebre claro, **más bajo que lo
esperable** — el comportamiento real está MENOS diferenciado entre
empresas que la retórica. Es en sí mismo un hallazgo: las empresas se
diferencian más en CÓMO hablan que en QUÉ hacen.

## Los 4 clusters de comportamiento (sin relación, por construcción, con A/B/C/D de voz)

| Cluster | n | `deployed` | `revenue_outcome` | `ai_investment` | `ai_infrastructure` | Customer-facing | Perfil |
|---|---|---|---|---|---|---|---|
| **0 — Narradores de revenue** | 26 | 15,4% | **26,9%** | 2,2% | 1,5% | 26,7% | Hablan de IA casi exclusivamente como motor de ingresos, poco despliegue propio |
| **1 — Comportamiento mínimo** | 233 | 16,0% | 1,3% | 1,1% | 0,5% | 13,6% | El grupo más grande — casi nada de comportamiento específico en ningún eje |
| **2 — Desplegadores de producto** | 114 | **42,4%** | 4,4% | 2,7% | 2,0% | **48,6%** | Alto despliegue real Y orientación a producto/cliente |
| **3 — Inversores en infraestructura** | 48 | 22,7% | 3,4% | **10,5%** | **8,1%** | 22,4% | Foco en `ai_investment`/`ai_infrastructure`, mayormente interno |

## Cruce voz × comportamiento (417 empresas con ambas etiquetas)

| Voz \ Comportamiento | 0 Narradores | 1 Mínimo | 2 Desplegadores | 3 Infraestructura |
|---|---|---|---|---|
| **A cauteloso** | 4,7% | **80,0%** | 6,7% | 8,7% |
| **B genérico** | 6,3% | 60,0% | 22,9% | 10,9% |
| **C cuantificador** | 28,6% | **0,0%** | 42,9% | 28,6% |
| **D vocal** | 4,7% | 7,1% | **71,8%** | 16,5% |

## Lectura: dónde voz y comportamiento coinciden, y dónde no

**A y D son mayormente consistentes con su propia narrativa** — A
(80% cae en "comportamiento mínimo") y D (72% cae en "desplegadores de
producto") tienen el comportamiento que su voz sugiere. **C nunca cae
en "comportamiento mínimo" (0%)** — por construcción, ya que su voz
se define por tener métricas cuantificadas, así que es casi imposible
que caiga en el cluster de comportamiento vacío; siempre tiene ALGO de
sustancia, aunque distribuida entre narrar revenue e invertir en
infraestructura.

**Pero hay dos grupos donde voz y comportamiento DIVERGEN — los casos
interesantes:**

### Candidatos a AI-washing: voz de "líder vocal" (D) + comportamiento MÍNIMO

6 empresas (7,1% de D) hablan de IA como los líderes vocales —
específico y promocional — pero su comportamiento real cae en el
cluster de "casi nada de sustancia":

**CCL, FE, GPC, HII, IQV, NEM**

Son los casos que un análisis de AI-washing debería mirar primero,
frame por frame — su voz no está respaldada por el mismo patrón de
comportamiento que el resto de D.

> **Corrección — la etiqueta NO persiste año a año para la mayoría de
> estos 6.** Esta clasificación se construyó sobre el pool de TODOS
> los años combinados; mirando el arquetipo de voz año a año
> (`firm_year_archetype_behaviors.parquet`) la imagen es mucho menos
> limpia:
>
> | Ticker | Años en panel | Arquetipo por año | n_frames por año |
> |---|---|---|---|
> | CCL | 1 (solo 2026) | D | 4 |
> | FE | 1 (solo 2026) | D | 3 |
> | GPC | 3 (2024-2026) | D, D, D | 8, 5, 6 |
> | HII | 4 (2023-2026) | D, D, **B**, **A** | 5, 5, 16, 14 |
> | IQV | 2 (2025-2026) | D, D | 29, 32 |
> | NEM | 2 (2025-2026) | D, D | 3, 3 |
>
> CCL, FE y NEM solo tienen 1-2 años de datos y con muy pocos frames
> (3-4/año) — su clasificación "D" pooled puede estar dominada por un
> solo año de bajo volumen, sin nada parecido a la persistencia de
> NVDA/MSFT (6/6 años en D, docenas-cientos de frames cada uno). **HII
> directamente migra fuera de D** (D→D→B→A) — no es un caso estable de
> "voz de líder con comportamiento mínimo", es una empresa que tuvo un
> par de años de voz más vistosa y volvió a un registro discreto. Solo
> GPC muestra algo parecido a persistencia real (3/3 años en D),
> aunque con specificity cayendo (0,15→0,04→0,03) y n_frames bajo.
> **Tratar estos 6 como "candidatos a washing" con la misma confianza
> que el resto-de-D estable (79 empresas, en su mayoría con presencia
> en D todos los años del panel) es un error de lectura — la etiqueta
> pooled no captura que la mayoría de estos casos es transitoria o de
> muestra chica, no una identidad de disclosure sostenida.**

### Sustancia callada: voz "genérica" o "cautelosa" + comportamiento de despliegue ALTO

40 empresas con voz B (genérica) pero comportamiento de despliegue de
producto (cluster 2) — hacen más de lo que su tono sugiere:

**ADSK, AKAM, ALGN, BKNG, CDAY, CDW, CFG, CNC, CPRT, DAL, DE, EFX, EXPE,
FAST, FMC, HPQ, IRM, JKHY, LH, MKTX, MTD, NET, NWS, OKTA, OMC, OTIS,
PLTR, RCL, RHI, RVTY, SEDG, SOLV, SQ, TEAM, TEL, TSCO, V, WDAY, WMT, ZTS**

10 empresas con voz A (cauteloso/riesgo) pero el MISMO comportamiento
de despliegue alto — el caso más extremo de "hacen mucho, cuentan
poco":

**CERN, DXCM, EMR, FISV, GM, HAL, LUV, MCO, ROP, ULTA**

Nota curiosa: PLTR (Palantir) aparece en la lista de "voz genérica" —
contraintuitivo dado su perfil público de marketing de IA muy agresivo.
Vale la pena revisar manualmente ese caso — puede ser un límite del
umbral de volumen/ventana temporal del panel, no necesariamente un
error, pero merece un chequeo antes de citarlo.

## Implicancia para la tesis

Esto reemplaza la lectura de `01_...md` de "D es sustancia Y
promoción a la vez, así que la variación interesante está dentro de
D" por algo más preciso: **la variación interesante no es dentro de un
arquetipo, es entre los dos ejes (voz, comportamiento) tratados como
independientes.** El framework de benchmarking que promete
`docs/thesis_proposal.md` debería reportarse en esta matriz 2D (voz ×
comportamiento), no en un solo arquetipo 1D — es la diferencia entre
"esta empresa habla mucho de IA" y "esta empresa habla mucho Y hace
poco", que es la definición operativa real de AI-washing.

## Limitaciones

- Silhouette bajo (0,18-0,19) en todos los k probados — los clusters de
  comportamiento son menos nítidos que los de voz; el k=4 es por
  comparabilidad, no porque los datos lo pidan con fuerza. Repetir con
  k=2 o k=3 antes de tratar 4 categorías de comportamiento como
  definitivas.
- Mismo umbral de volumen (≥5 frames pooled) y mismas limitaciones de
  no ponderar por `duplicate_count`/`inclusion_weight` que el resto del
  proyecto.
- Las listas de "washing"/"sustancia callada" son de la corrida k=4
  con seed=42 — no se verificó estabilidad frente a otras semillas
  (mismo caveat que el arquetipo de voz en `01_...md`).
- No se revisó ningún caso manualmente (frame por frame) — las listas
  son candidatos para revisión cualitativa, no conclusiones.
