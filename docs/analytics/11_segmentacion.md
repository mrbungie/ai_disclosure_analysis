# Segmentación de empresas por cómo divulgan IA

Responde la pregunta central de `docs/thesis_proposal.md`:

> *How do public firms differ in their AI disclosure behaviors regarding AI
> **adoption**, **capabilities**, **risks** and **governance**, and what
> distinct disclosure archetypes emerge?*

Producido por `scripts/analytics/build_segments.py`. **Modo de análisis final:
todas las empresas con filings entran** —510, incluidas las 17 que no tienen
ningún frame de IA— y la intensidad de IA (frames por 1.000 párrafos de sus
filings, en log) es una dimensión más de la segmentación. Reemplaza los
arquetipos A/B/C/D, que no eran reproducibles (Jaccard bootstrap 0,53 —
`cluster_diagnostics.py`).

## Cómo se construyó, y por qué así

| dimensión de la pregunta | features |
|---|---|
| Adopción | % despliegue, % escalamiento, % etapa temprana (piloto/exploración), % resultados |
| Capacidades | % capacidad propia (inversión, infraestructura, talento, IA propia), % IA de terceros |
| Riesgos | % riesgo, % hipotético |
| Gobernanza | % gobernanza |
| Registro | % promocional, % cuantificado, % de cara al producto |
| Intensidad | log(1 + frames de IA por 1.000 párrafos) |

Cuatro decisiones de construcción:

1. **Los conceptos raros se agrupan en su dimensión.** `ai_investment`,
   `ai_infrastructure`, `ai_talent` y `proprietary_ai` aparecen cada uno en
   2-3% de los frames y por separado son ruido; juntos son "capacidades".
2. **Encogimiento empírico-Bayes de las tasas**: la tasa de una empresa con 8
   frames se corre hacia el promedio del corpus en proporción a lo poco que
   se sabe de ella. Con 0 frames queda exactamente en el prior: la empresa
   que no habla de IA entra con "cómo habla" indeterminado e intensidad cero.
3. **La intensidad es una dimensión**, no un filtro. Las tasas dicen cómo
   habla quien habla; la intensidad dice cuánto del filing se dedica a IA.
   Sin ella, la empresa con 3 frames y la de 300 con las mismas tasas serían
   la misma empresa.
4. **k se elige por ESTABILIDAD, no por silhouette.** Se remuestrean los
   frames de cada empresa y se vuelve a segmentar: si le hubieran tocado
   otros de sus propios frames, ¿queda en el mismo grupo?

| k | Jaccard por segmento | ¿usable? |
|---|---|---|
| 2 | 0,80 / 0,89 | sí |
| **3** | **0,71 / 0,75 / 0,73** | **sí — elegido** |
| 4 | 0,58 / 0,72 / 0,59 / 0,77 | no |
| 5 | 0,50 / 0,77 / 0,69 / 0,31 / 0,56 | no |

Los cuatro arquetipos A/B/C/D caían justo acá: **k=4 no es reproducible**.
Con las 510 empresas y la intensidad como dimensión, k=3 es más estable que
sobre las 419 con ≥8 frames (mínimo 0,71 contra 0,63).

## Los tres segmentos

510 empresas. Los porcentajes son sobre las afirmaciones de IA de la empresa;
la intensidad es la mediana de frames por 1.000 párrafos.

| | Desplegadores de producto | Adoptantes con gobernanza | Listadores de riesgo |
|---|---:|---:|---:|
| **empresas** | **115** | **239** | **156** |
| frames (mediana) | 97 | 21 | 18 |
| frames por 1.000 párrafos (mediana) | **8,4** | 1,4 | 1,2 |
| empresas sin ningún frame | 0 | 17 | 0 |
| estabilidad (Jaccard) | 0,73 | 0,71 | 0,75 |
| % despliegue | **42,2** | 22,8 | 12,0 |
| % resultados | **25,8** | 13,2 | 5,2 |
| % de cara al producto | **45,6** | 15,9 | 12,2 |
| % capacidad propia | **11,7** | 7,6 | 3,2 |
| % gobernanza | 9,3 | **22,7** | 11,5 |
| % riesgo | 23,9 | 33,4 | **70,6** |
| % hipotético | 6,1 | 5,7 | **25,7** |
| % promocional | **11,4** | 3,4 | 1,3 |
| % cuantificado | **6,4** | 2,6 | 0,6 |

Empresas típicas (las más cercanas al centroide):

- **Desplegadores de producto**: HPE, PAYX, ETSY, TRMB, NOW, NTAP, HPQ, KEYS.
- **Adoptantes con gobernanza**: ABMD, APA, ATO, BALL, CF, DRE, GRMN, HONA.
- **Listadores de riesgo**: TDG, CZR, LVS, DHI, NKE, HWM, TJX, CMA.

**Los nombres salen del perfil, no de un orden fijo**: cada segmento se nombra
por la dimensión donde más se despega del promedio, medido en z-scores. Las
17 empresas sin frames caen en adoptantes con gobernanza —el segmento cuyo
perfil está más cerca del prior— y no forman un segmento propio: con k=4 sí
lo formarían, pero k=4 no es estable.

### Qué dice cada uno

- **Desplegadores de producto (115)** — describen IA funcionando y de cara al
  cliente: 42% de sus afirmaciones son despliegue, 26% resultados, 46%
  orientadas a producto, y dedican **seis veces más de su filing a IA** que
  los otros dos (8,4 contra 1,2-1,4 frames por 1.000 párrafos). Son también
  los más promocionales (11,4%) *y* los que más cuantifican (6,4%): hablan
  fuerte y con números. Software y hardware.
- **Adoptantes con gobernanza (239)** — el segmento más grande. Adoptan (23%
  despliegue) pero su marca distintiva es la **gobernanza**: 22,7%, dos veces
  y media la de los desplegadores. Bancos, aseguradoras, utilities y las
  empresas que apenas hablan de IA.
- **Listadores de riesgo (156)** — el 71% de sus afirmaciones sobre IA son
  riesgo, un cuarto son hipotéticas, y prácticamente no hay lenguaje
  promocional (1,3%) ni cuantificación (0,6%). Es el boilerplate de factores
  de riesgo.

## Que la segmentación sirva río abajo

Persistencia año a año: **73,6%** sobre 2.454 pares empresa-año consecutivos
(`firm_year_segments.parquet` asigna cada empresa-año —todas las 2.964 con
filings— proyectando sobre los centroides ya entrenados).

Los segmentos separan cosas que NO entraron a construirlos (medianas por
empresa, `firm_year_master_v2`):

| | Desplegadores | Adoptantes c/gob. | Listadores de riesgo |
|---|---:|---:|---:|
| I+D / ingresos | **12,0%** | 4,8% | 2,3% |
| Margen bruto | **60,7%** | 41,8% | 38,7% |
| Beta | **1,08** | 0,82 | 0,81 |
| P/E | **29,7** | 22,9 | 22,4 |
| Crecimiento de ingresos t+1 | **7,6%** | 6,8% | 6,0% |
| ROIC − WACC | **+5,4%** | +4,9% | +4,7% |
| **Empresas marcadas por el score de washing (`09`)** | **8 de 115** | 0 de 239 | 0 de 156 |

Es el chequeo que importa: la segmentación se construyó sólo con texto y
ordena monótonamente I+D, margen, beta, valuación y crecimiento — y **las 8
empresas con exceso promocional estadísticamente significativo caen todas
en un solo segmento**. Los adoptantes y los listadores no se distinguen en
lo financiero: la distinción entre ellos es de discurso, no de tipo de
empresa.

## Limitaciones

- Con la intensidad como dimensión, el segmento de desplegadores es en parte
  "las que más hablan de IA"; eso es deliberado y está en la tabla (8,4
  contra 1,2-1,4 frames por 1.000 párrafos).
- Las 17 empresas sin frames tienen "cómo hablan" indeterminado; su
  asignación a adoptantes con gobernanza es la del prior, no un dato sobre
  ellas.
- Todo descansa en etiquetas de un LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
