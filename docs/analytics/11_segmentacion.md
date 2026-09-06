# Segmentación de empresas por cómo divulgan IA

Responde la pregunta central de `docs/thesis_proposal.md`:

> *How do public firms differ in their AI disclosure behaviors regarding AI
> **adoption**, **capabilities**, **risks** and **governance**, and what
> distinct disclosure archetypes emerge?*

Producido por `scripts/analytics/build_segments.py`. Reemplaza los arquetipos
A/B/C/D, que no eran reproducibles (Jaccard bootstrap 0,53 —
`cluster_diagnostics.py`).

## Cómo se construyó, y por qué así

**Las features son las dimensiones de la pregunta**, en la unidad que se lee sin
traducir: *porcentaje de las afirmaciones de IA de esa empresa*.

| dimensión de la pregunta | features |
|---|---|
| Adopción | % despliegue, % escalamiento, % etapa temprana (piloto/exploración), % resultados |
| Capacidades | % capacidad propia (inversión, infraestructura, talento, IA propia), % IA de terceros |
| Riesgos | % riesgo, % hipotético |
| Gobernanza | % gobernanza |
| Registro | % promocional, % cuantificado, % de cara al producto |

Tres decisiones que hacen la diferencia entre esto y la versión anterior:

1. **Los conceptos raros se agrupan en su dimensión.** `ai_investment`,
   `ai_infrastructure`, `ai_talent` y `proprietary_ai` aparecen cada uno en 2-3%
   de los frames y por separado una tasa así es casi toda ruido (confiabilidad
   0,35-0,52). Juntos son "capacidades", llegan a ~10% y se miden. El
   agrupamiento es sustantivo: la pregunta es por capacidades, no por
   `ai_talent`.
2. **Encogimiento empírico-Bayes**: la tasa de una empresa con 8 frames se corre
   hacia el promedio del corpus, porque de ella no se sabe nada. Sin esto,
   k-means agrupa denominadores.
3. **k se elige por ESTABILIDAD, no por silhouette.** Se remuestrean los frames
   de cada empresa y se vuelve a segmentar: si le hubieran tocado otros de sus
   propios frames, ¿queda en el mismo grupo?

| k | Jaccard por segmento | ¿usable? |
|---|---|---|
| 2 | 0,80 / 0,87 | sí |
| **3** | **0,63 / 0,72 / 0,70** | **sí — elegido** |
| 4 | 0,56 / 0,62 / 0,72 / 0,71 | no |
| 5 | 0,42 / 0,67 / 0,23 / 0,58 / 0,68 | no |

Los cuatro arquetipos anteriores caían justo acá: **k=4 nunca fue reproducible**.

## Los tres segmentos

419 empresas con ≥8 frames (re-corrida 2026-09-06 sobre el prefiltro v2; antes
420). Todos los números son % de las afirmaciones de IA de la empresa.

| | Desplegadores de producto | Adoptantes con gobernanza | Listadores de riesgo |
|---|---:|---:|---:|
| **empresas** | **101** | **189** | **129** |
| frames (mediana) | 91 | 32 | 23 |
| estabilidad (Jaccard) | 0,70 | 0,63 | 0,72 |
| % despliegue | **43,5** | 24,7 | 13,8 |
| % resultados | **26,9** | 13,3 | 6,1 |
| % de cara al producto | **46,5** | 20,4 | 13,2 |
| % capacidad propia | **11,5** | 8,5 | 4,1 |
| % gobernanza | 8,5 | **23,4** | 13,2 |
| % riesgo | 22,6 | 36,4 | **66,3** |
| % hipotético | 5,6 | 7,0 | **23,5** |
| % promocional | **11,9** | 3,8 | 1,5 |
| % cuantificado | **6,7** | 2,6 | 0,8 |

Empresas típicas (las más cercanas al centroide):

- **Desplegadores de producto**: HPE, TEAM, NOW, ACN, NTAP, ANET, KEYS, TRMB.
- **Adoptantes con gobernanza**: STT, AIZ, RMD, BLK, EIX, CINF, RF, UDR.
- **Listadores de riesgo**: DHI, HWM, NKE, EG, ITW, MKC, FITB, PPG.

**Los nombres salen del perfil, no de un orden fijo**: cada segmento se nombra
por la dimensión donde más se despega del promedio. (Una versión anterior
llamaba "constructores de capacidad" a un segmento cuya capacidad —8,5%— era
MENOR que la de los desplegadores —11,5%—: el nombre salía de cuál etiqueta
quedaba libre. Corregido con z-scores.)

### Qué dice cada uno

- **Desplegadores de producto (101)** — describen IA funcionando y de cara al
  cliente: 43,5% de sus afirmaciones son despliegue, 26,9% resultados, casi la
  mitad orientadas a producto. Son también los más promocionales (11,9%) *y* los
  que más cuantifican (6,7%): hablan fuerte y con números. Software y hardware
  (SIC 73, 35, 36).
- **Adoptantes con gobernanza (189)** — el segmento más grande. Adoptan (24,7%
  despliegue) pero su marca distintiva es la **gobernanza**: 23,4%, casi tres
  veces la de los desplegadores. Bancos, aseguradoras y utilities (SIC 73, 38,
  28, 49, 63): sectores regulados que cuando hablan de IA hablan de supervisión.
- **Listadores de riesgo (129)** — dos tercios de sus afirmaciones sobre IA son
  riesgo, un cuarto son hipotéticas, y prácticamente no hay lenguaje promocional
  (1,5%) ni cuantificación (0,8%). Es el boilerplate de factores de riesgo.

## Que la segmentación sirva río abajo

Persistencia año a año: **71,3%** sobre los pares empresa-año consecutivos
(`firm_year_segments.parquet` asigna cada empresa-año proyectando sobre los
centroides ya entrenados, sin re-segmentar por año).

Los segmentos separan cosas que NO entraron a construirlos:

| | Desplegadores | Adoptantes c/gob. | Listadores de riesgo |
|---|---:|---:|---:|
| I+D / ingresos | **13,0%** | 6,5% | 2,8% |
| Margen bruto | **62,0%** | 46,3% | 40,5% |
| Beta | **1,09** | 0,79 | 0,75 |
| P/E | **30,2** | 24,1 | 21,6 |
| Crecimiento de ingresos t+1 | **7,6%** | 6,0% | 4,4% |
| ROIC − WACC | **+5,1%** | +4,3% | +3,6% |
| **Empresas marcadas por el score de washing** | **8 de 101** | 0 de 189 | 0 de 129 |

Es el chequeo que importa: la segmentación se construyó sólo con texto y ordena
monótonamente I+D, beta, valuación y crecimiento — y **las 8 empresas con exceso
promocional estadísticamente significativo caen todas en un solo segmento**.

## Limitaciones

- El segmento de desplegadores tiene 91 frames medianos contra 23 de los
  listadores de riesgo: parte de lo que separa a los segmentos es **cuánto habla
  cada empresa**, no sólo cómo. El encogimiento controla el ruido de las tasas,
  no esa diferencia de volumen.
- 419 de 493 empresas entran (≥8 frames); las que quedan fuera son las que menos
  divulgan, así que el segmento "listadores de riesgo" está probablemente
  sub-representado.
- Todo descansa en etiquetas de un LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
