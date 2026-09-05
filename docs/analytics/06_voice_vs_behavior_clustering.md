# Voz vs. comportamiento: dos clusterings separados, cruzados

> **Recalculado 2026-09-05 con DEF 14A y 8-K** (población: 24.328 frames,
> 446 empresas con ≥5 frames), vía `scripts/analytics/build_firm_clusters.py`.
> La versión anterior corría sobre 10-K solamente (421 empresas). K-means
> re-ajusta centroides sobre la población nueva, así que las membresías
> cambian; los parquets viejos quedaron en
> `data/archive/processed/clusters/`. En el camino se corrigió un bug de
> reproducibilidad en la vista `gold_ai_frames` que hacía que cada
> consulta devolviera cifras distintas — ver `01_...md`.

Corrige un problema conceptual de `01_ai_disclosure_analytics.md`: los 4
arquetipos (A/B/C/D) se construyen sobre 9 métricas que son **todas
retórica/voz** — `specificity_index`, `quantified_rate`,
`promotional_rate`, `strategic_rate`, `realized_share`,
`hypothetical_share`, `risk_share`, `gov_share`, `firm_subject_share` —
ninguna es un `behavior_share_*` real. Esos conceptos de comportamiento
genuino se usaban después, como variable DESCRIPTIVA de cada arquetipo ya
formado, nunca como insumo del clustering. El nombre "líder vocal" para D
ya lo delataba: es vocal, no necesariamente "actúa mucho". (`01_...md`
ahora los llama "arquetipos de voz" por esto.)

Esta sección corre un **segundo clustering, independiente, sobre los
`behavior_share_*` reales**, y lo cruza contra el arquetipo de voz — la
comparación correcta para identificar candidatos a AI-washing (voz alta +
comportamiento bajo) vs. sustancia callada (voz baja + comportamiento
alto), en vez de asumir que ambos ejes son lo mismo.

## Construcción

Mismas 446 empresas (≥5 frames, agregado por ticker, todos los años
pooled), pero 15 features de comportamiento en vez de las 9 de voz:

```python
BEHAVIOR_CONCEPTS = ['deployed','pilot_or_testing','exploring','ai_investment',
                     'ai_infrastructure','ai_talent','proprietary_ai','third_party_ai',
                     'expansion_or_scaling','productivity_outcome','revenue_outcome',
                     'cost_outcome','customer_outcome']  # sin use_stage_unspecified (residual)
FEAT_COLS = BEHAVIOR_CONCEPTS + ['domain_customer_facing', 'domain_internal']
X = StandardScaler().fit_transform(pop[FEAT_COLS].values)
km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
```

k=4 elegido por comparabilidad directa con el arquetipo de voz, no porque
el codo/silhouette lo pidan: silhouette entre 0,150 (k=5) y 0,162 (k=2 y
k=3), sin quiebre claro, **más bajo que lo esperable** y algo más bajo que
en la versión 10-K (0,182-0,190). El comportamiento real está MENOS
diferenciado entre empresas que la retórica, y sumar proxies y 8-K lo
diferencia todavía menos. Es en sí mismo un hallazgo, ahora sobre más
datos: **las empresas se diferencian más en CÓMO hablan que en QUÉ hacen.**

Los números de cluster se asignan por perfil, no por el id que devuelve
sklearn (`_label_behavior_clusters` en el script): 3 reclama
`ai_infrastructure`, 2 `domain_customer_facing`, 0 `deployed`, y 1 es el
residuo. El orden va del marcador más separado al menos separado — una
versión anterior reclamaba por `revenue_outcome` y era inestable, porque
dos clusters quedaban a 0,4 p.p. en esa columna y un cambio de último
dígito daba vuelta 225 de 446 etiquetas entre corridas.

## Los 4 clusters de comportamiento (sin relación, por construcción, con A/B/C/D de voz)

| Cluster | n | `deployed` | `revenue_outcome` | `ai_investment` | `ai_infrastructure` | Customer-facing | Perfil |
|---|---|---|---|---|---|---|---|
| **0 — Intermedio interno** | 25 | 17,9% | 2,8% | 1,5% | 0,7% | 18,6% | Poco de todo, algo más de inversión que el mínimo |
| **1 — Comportamiento mínimo** | 214 | 17,1% | 2,1% | 0,8% | 0,6% | 13,1% | El grupo más grande — casi nada de comportamiento específico en ningún eje |
| **2 — Desplegadores de producto** | 132 | **41,7%** | **7,1%** | 2,4% | 2,1% | **44,2%** | Despliegue real Y orientación a producto/cliente |
| **3 — Inversores en infraestructura** | 75 | 23,7% | 3,5% | **7,2%** | **5,6%** | 18,5% | Foco en `ai_investment`/`ai_infrastructure`, mayormente interno |

Sólo dos de los cuatro clusters tienen identidad propia clara: el 2
(despliegue de producto) y el 3 (infraestructura). El 0 y el 1 son
variaciones del mismo perfil apagado, separadas por márgenes de menos de
1 p.p. en casi todas las columnas — **tratarlos como dos categorías
distintas no está justificado por los datos**, y es parte de por qué el
silhouette es tan bajo.

**El cluster "narradores de revenue" desapareció.** En la versión 10-K
eran 26 empresas con 26,9% de `revenue_outcome`, las que hablaban de IA
como motor de ingresos de terceros. Con la población ampliada ningún
cluster supera 7,1% en esa columna. Igual que pasó con otros grupos
chicos y extremos de este proyecto, no sobrevivió al cambio de población.

## Cruce voz × comportamiento (446 empresas con ambas etiquetas)

| Voz \ Comportamiento | 0 Intermedio | 1 Mínimo | 2 Despl. producto | 3 Infraestructura |
|---|---|---|---|---|
| **A cauteloso** | 8,1% | **81,1%** | 0,9% | 9,9% |
| **B genérico** | 7,7% | 53,6% | 21,5% | 17,1% |
| **C cuantificador** | 0,0% | 20,0% | **57,1%** | 22,9% |
| **D vocal** | 1,7% | 16,8% | **60,5%** | 21,0% |

## Lectura: dónde voz y comportamiento coinciden, y dónde no

**Los dos ejes están fuertemente alineados en los extremos.** A cae 81,1%
en comportamiento mínimo; C y D caen 57,1% y 60,5% en despliegue de
producto. Ninguna empresa C tiene el perfil intermedio-interno (0,0%).

**C y D son casi indistinguibles en el eje de comportamiento** (57,1% vs.
60,5% en el cluster 2; 22,9% vs. 21,0% en el 3). Eso es informativo dado
que en el eje de VOZ son muy distintos: C cuantifica 5,6x más que D y D
promociona 1,6x más que C (`01_...md`). **Hacen lo mismo y lo cuentan
distinto** — que es exactamente la separación que este documento fue
escrito para hacer visible.

### Candidatos a AI-washing: voz de "líder vocal" (D) + comportamiento MÍNIMO

> **SUPERADO por `09_washing_score.md` (2026-09-05).** Esta definición
> —el cruce de dos etiquetas de cluster— no es medible: los arquetipos de
> voz se construyen sobre tasas con denominadores de 5 a 500 frames, y
> cuatro de las empresas que caen acá tienen CERO frames promocionales.
> El reemplazo modela conteos con un test binomial exacto y corrección
> por comparaciones múltiples. Lo que sigue se conserva como registro de
> lo que se intentó, no como resultado.


**20 empresas** (16,8% de D):

**AAPL, ADI, ALL, BMY, CHD, CL, CMS, ECL, ELV, FE, GILD, GPC, HUM, JPM,
LDOS, RL, SO, SOLS, TFC, UDR**

El grupo creció respecto de la versión anterior (6 empresas, 7,1% de D) y
ahora tiene un tamaño con el que se puede trabajar. Pero la advertencia de
persistencia se mantiene y hay que leerla antes que la lista:

> **Sólo 2 de las 20 sostienen la etiqueta D en todos sus años con al
> menos 3 años de panel: AAPL y CL.** El resto o tiene 1-2 años de datos
> (CHD, CMS, FE, SOLS...) o migra dentro y fuera de D (ADI:
> C→D→B→D→D; ALL: B→D→D→D; BMY: D→B; ELV: D→D→B). Compárese con la
> persistencia de los D grandes: GOOGL y ADBE en D los 6 años, INTC 5 de
> 6, MSFT 5 de 6.
>
> La etiqueta pooled promedia todos los años, así que una empresa con dos
> años vocales y tres callados puede entrar igual. **Antes de tratar a
> estas 20 como candidatos a washing hay que exigirles persistencia**, y
> con ese filtro quedan 2.

Sectorialmente el grupo es coherente con la hipótesis: farma (BMY, GILD),
seguros de salud (ELV, HUM), utilities (CMS, FE, SO), consumo (CHD, CL,
RL), banca (JPM, TFC). Son sectores donde la IA es plausible como
herramienta interna pero difícilmente como producto — y el frame vocal sin
comportamiento observable es justo lo que uno esperaría de un discurso
adoptado por presión de mercado más que por despliegue real. AAPL en esa
lista es el caso que más merece revisión manual.

### Sustancia callada: voz "genérica" o "cautelosa" + comportamiento de despliegue ALTO

**39 empresas** con voz B (genérica) pero comportamiento de despliegue de
producto (cluster 2) — hacen más de lo que su tono sugiere:

**ADSK, AON, AVB, CCI, CDAY, CERN, CPRT, DAL, EMR, EQIX, EXPE, FAST, FLT,
ICE, IRM, J, JKHY, KHC, KLAC, LH, LUV, MSCI, NET, NWS, OKTA, OTIS, QRVO,
RHI, RMD, ROP, SEDG, STX, T, TEL, TER, ULTA, V, WAB, ZTS**

**1 empresa** con voz A (cautelosa) y el mismo comportamiento de
despliegue alto: **CDW**. Con n=1 no es un grupo, es una anécdota.

El cuadrante de sustancia callada (40 empresas) y el de washing (20) son
hoy del mismo orden de magnitud, a diferencia de la versión anterior donde
la sustancia callada era 8x más grande. Aplicando el filtro de
persistencia a ambos, ninguno de los dos queda con tamaño suficiente para
sostener una afirmación.

### El cuadrante inverso: C con comportamiento mínimo

7 empresas cuantifican sus afirmaciones sobre IA (voz C) pero caen en el
cluster de comportamiento mínimo: **CMG, CSX, CTLT, DUK, JBHT, NRG, PSA**.

Son mayormente transporte, utilities y logística — empresas que citan
cifras sobre el efecto de la IA en su **demanda** (centros de datos,
consumo eléctrico, volumen de carga) sin desplegar IA propia. Es el mismo
patrón que la versión original identificaba en su cluster C de 7 empresas
de energía/industrial, y que ya entonces se señalaba como
metodológicamente incómodo: es sustancia real, pero sobre el efecto de la
IA en el negocio de un tercero, no sobre capacidad propia. **No encaja en
el eje washing-vs-creíble tal como está planteado**, y conviene tratarlo
como una tercera categoría en vez de forzarlo.

## Implicancia para la tesis

Esto reemplaza la lectura de `01_...md` de "D es sustancia Y promoción a
la vez" por algo más preciso: **la variación interesante no es dentro de
un arquetipo, es entre los dos ejes tratados como independientes.** El
framework de benchmarking que promete `docs/thesis_proposal.md` debería
reportarse en esta matriz 2D, no en un solo arquetipo 1D.

Lo que agrega la actualización, y no es cómodo: **el eje de comportamiento
no separa a C de D**, que son los dos grupos que más se diferencian en
voz. Si el objetivo es detectar AI-washing como "dice mucho, hace poco",
la matriz 2D lo detecta en pocos casos y ninguno persistente. La
distinción que sí aparece con fuerza es otra: **entre cuantificar y
promocionar**, ambos con comportamiento equivalente detrás. Esa —y no la
matriz voz×comportamiento— parece ser la separación con contenido en este
corpus.

## Limitaciones

- Silhouette bajo (0,150-0,162 en todos los k probados, más bajo que en la
  versión 10-K) — k=4 es por comparabilidad, no porque los datos lo pidan.
  Los clusters 0 y 1 no están realmente separados; repetir con k=2 o k=3
  antes de tratar 4 categorías como definitivas.
- **Los grupos chicos y extremos no sobreviven cambios de población.** Los
  "narradores de revenue" desaparecieron entre una versión y otra, igual
  que otros grupos de este proyecto. Cualquier lectura sustantiva debería
  apoyarse en los cuadrantes grandes (A×1, C/D×2), no en las esquinas.
- **La etiqueta pooled no implica persistencia.** Es el problema central
  de las listas de washing y sustancia callada: sólo 2 de 20 candidatos
  sostienen su etiqueta año a año. Cualquier versión futura debería
  construir los cuadrantes sobre el panel empresa-año con un requisito de
  persistencia, no sobre el pool.
- Mismo umbral de volumen (≥5 frames pooled) y mismas limitaciones de no
  ponderar por `duplicate_count`/`inclusion_weight` que el resto.
- Listas de la corrida k=4 con seed=42; no se verificó estabilidad frente
  a otras semillas.
- **La composición por formulario contamina el eje de voz.** La DEF 14A
  tiene 16,3% de frames promocionales contra 7,1% del 10-K, así que una
  empresa con proxy extenso se corre hacia "vocal" por mezcla documental.
