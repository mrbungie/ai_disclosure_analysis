# Segmentación de empresas por cómo divulgan IA

Responde la pregunta central de `docs/thesis_proposal.md`:

> *How do public firms differ in their AI disclosure behaviors regarding AI
> **adoption**, **capabilities**, **risks** and **governance**, and what
> distinct disclosure archetypes emerge?*

Producido por `scripts/analytics/build_segments.py`. **Modo de análisis final:
todas las empresas con filings entran** —510— y la intensidad de IA (frames
por 1.000 párrafos de sus filings, en log) es una dimensión más de la
segmentación. La empresa que no dice nada de IA es un segmento propio, **sin
IA**, por regla: no hay "cómo habla" que segmentar. Son 17 empresas en el
pooled 2021-2026 y **1.088 de las 2.964 empresas-año** en el panel (66% de
2021, 8% de 2026). Reemplaza los arquetipos A/B/C/D, que no eran
reproducibles (Jaccard bootstrap 0,53 — `cluster_diagnostics.py`).

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
   se sabe de ella. Con 0 frames no hay tasa: la empresa va al segmento
   **sin IA** por regla, y el K-means se ajusta sobre las que sí hablan.
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
Con la intensidad como dimensión, k=3 es más estable que sobre las 419 con
≥8 frames (mínimo 0,71 contra 0,63).

## Los cuatro segmentos: tres por K-means, uno por regla

510 empresas. Los porcentajes son sobre las afirmaciones de IA de la empresa;
la intensidad es la mediana de frames por 1.000 párrafos.

| | Desplegadores de producto | Adoptantes con gobernanza | Listadores de riesgo | Sin IA |
|---|---:|---:|---:|---:|
| **empresas** | **113** | **222** | **158** | **17** |
| frames (mediana) | 97 | 24 | 18 | 0 |
| frames por 1.000 párrafos (mediana) | **8,4** | 1,6 | 1,2 | 0 |
| estabilidad (Jaccard) | 0,75 | 0,73 | 0,71 | 1 (por regla) |
| % despliegue | **42,2** | 24,8 | 12,0 | — |
| % resultados | **26,1** | 14,2 | 5,4 | — |
| % de cara al producto | **45,8** | 17,3 | 12,1 | — |
| % capacidad propia | **11,7** | 8,3 | 3,2 | — |
| % gobernanza | 9,1 | **24,4** | 11,7 | — |
| % riesgo | 23,8 | 35,8 | **70,4** | — |
| % hipotético | 6,2 | 6,1 | **25,6** | — |
| % promocional | **11,5** | 3,8 | 1,3 | — |
| % cuantificado | **6,4** | 2,9 | 0,6 | — |

Empresas típicas (las más cercanas al centroide):

- **Desplegadores de producto**: HPE, PAYX, ETSY, TRMB, NOW, NTAP, HPQ, KEYS.
- **Adoptantes con gobernanza**: STT, BRK.B, PWR, DHR, CINF, RF, AIZ, IEX.
- **Listadores de riesgo**: TDG, CZR, LVS, DHI, NKE, TJX, HWM, CMA.
- **Sin IA** (las 17): ABMD, APA, ATO, BALL, CF, DRE, GRMN, HONA, HSIC, IPGP,
  MOS, MRO, PBCT, PKG, PSKY, SIVB, TFX. Varias son empresas adquiridas o
  salidas del índice con pocos años de filings (ABMD, DRE, PBCT, SIVB, TFX);
  en el pooled casi nadie se queda sin IA los seis años. **Donde el segmento
  importa es en el panel**: 1.088 empresas-año sin IA, dos tercios de 2021 y
  menos de una de cada diez desde 2025.

**Los nombres salen del perfil, no de un orden fijo**: cada segmento se nombra
por la dimensión donde más se despega del promedio, medido en z-scores.

### Qué dice cada uno

- **Desplegadores de producto (113)** — describen IA funcionando y de cara al
  cliente: 42% de sus afirmaciones son despliegue, 26% resultados, 46%
  orientadas a producto, y dedican **cinco a siete veces más de su filing a
  IA** que los otros dos (8,4 contra 1,2-1,6 frames por 1.000 párrafos). Son
  también los más promocionales (11,5%) *y* los que más cuantifican (6,4%):
  hablan fuerte y con números. Software y hardware.
- **Adoptantes con gobernanza (222)** — el segmento más grande. Adoptan (25%
  despliegue) pero su marca distintiva es la **gobernanza**: 24,4%, dos veces
  y media la de los desplegadores. Bancos, aseguradoras, utilities.
- **Listadores de riesgo (158)** — el 70% de sus afirmaciones sobre IA son
  riesgo, un cuarto son hipotéticas, y prácticamente no hay lenguaje
  promocional (1,3%) ni cuantificación (0,6%). Es el boilerplate de factores
  de riesgo.
- **Sin IA (17 pooled; 1.088 empresas-año)** — ninguna afirmación de IA en
  los filings del período. En el panel es el segmento mayoritario hasta 2023
  y residual desde 2025: el boom se ve como el vaciado de este segmento.

## Qué hace cada segmento, en actividades

Las tasas dicen de qué habla cada segmento; las actividades divulgadas
(`09_actividades_ia.md`: la empresa hace ACCIÓN sobre OBJETO para FUNCIÓN,
con origen de la IA y entidades nombradas) dicen qué dice que hace. % de
empresas del segmento con al menos una actividad de cada tipo, salvo donde se
indica:

| | Desplegadores de producto | Adoptantes con gobernanza | Listadores de riesgo |
|---|---|---|---|
| actividades por empresa (mediana) | **112** | 18 | 5 |
| acciones principales | deploy, scale, invest_infrastructure, partner, hire_or_train | deploy, hire_or_train, invest_infrastructure | deploy, hire_or_train |
| objetos más frecuentes | plataforma de analítica (88%), producto con nombre propio (84%: AIOps, AIP, Mist AI), modelos predictivos (80%), data centers (62%), aceleradores y AI PC (56%) | modelos predictivos (46%), plataforma de analítica (42%), automatización (33%), producto con nombre (29%), visión y robótica industrial (16%), copilots (15%) | modelos predictivos (25%), plataforma (22%), automatización (16%), producto con nombre (8%) |
| funciones (% de las actividades) | operaciones 66 (deploy producto), producto 46, atención al cliente 40, IT 35 | automatización de operaciones 28, producto en operaciones 27, atención al cliente 15, marketing 10 | operaciones 11, automatización 10, modelos en operaciones 10, marketing 6 |
| para clientes / interno (% de las actividades, media) | 51 / 33 | 27 / 60 | 21 / 66 |
| origen de la IA: propia / de terceros / no dice (% de las actividades) | 80 / 6 / 10 | 55 / 12 / 30 | 46 / 17 / 34 |
| marca propia nombrada / proveedor o socio externo nombrado / cliente nombrado | **97 / 89 / 48** | 64 / 46 / 10 | 34 / 27 / 5 |
| inversión en infraestructura / co-desarrollo o adquisición | **90 / 66** | 55 / 19 | 25 / 7 |
| resultado cuantificado | **94** | 65 | 37 |
| etapa máxima: escalado / desplegado / piloto o exploración / ninguna | 93 / 7 / 0 / 0 | 64 / 27 / 3 / 4 | 34 / 44 / 7 / 11 |

Los desplegadores de producto venden IA propia con marca: casi todos nombran
un producto propio y un proveedor o socio externo, la mitad nombra clientes,
dos tercios co-desarrollan o adquieren, y 80% de sus actividades declaran la
IA como propia. Los adoptantes con gobernanza automatizan operaciones y
corren modelos predictivos hacia adentro, con IA propia sin marca y en un
tercio de los casos sin decir de dónde sale. Los listadores de riesgo
describen cinco actividades en la mediana, uno de cada diez ninguna, y lo
poco que describen es automatización o un modelo interno, con la mayor
proporción de IA de terceros (17%) y de origen no declarado (34%). Fichas
por empresa (Microsoft, ServiceNow, JPMorgan, Nike, Howmet…) en `09`.

## Que la segmentación sirva río abajo

Persistencia año a año: **65,0%** sobre 2.454 pares empresa-año consecutivos
(`firm_year_segments.parquet` asigna cada empresa-año —todas las 2.964 con
filings— proyectando sobre los centroides ya entrenados, y a "sin IA" por
regla). La mayor parte de los cambios son salidas de "sin IA": entre 2023 y
2024 el segmento pasa de 56% a 21% de las empresas-año.

Los segmentos separan cosas que NO entraron a construirlos (medianas por
empresa, `firm_year_master_v2`):

| | Desplegadores | Adoptantes c/gob. | Listadores de riesgo | Sin IA |
|---|---:|---:|---:|---:|
| I+D / ingresos | **12,1%** | 4,6% | 2,3% | 7,7% |
| Margen bruto | **61,0%** | 41,8% | 38,7% | 40,5% |
| Beta | **1,08** | 0,82 | 0,81 | 0,77 |
| P/E | **29,7** | 22,9 | 22,4 | 22,8 |
| Market cap (mediana) | $39B | $37B | $25B | **$13B** |
| Crecimiento de ingresos t+1 | 7,6% | 6,7% | 6,0% | **11,1%** |
| ROIC − WACC | +5,1% | +5,0% | +4,7% | +5,1% |
| **Empresas marcadas por el score de washing (`09`)** | **8 de 113** | 0 de 222 | 0 de 158 | 0 de 17 |

Es el chequeo que importa: la segmentación se construyó sólo con texto y
ordena monótonamente I+D, margen, beta y valuación entre los tres segmentos
que hablan — y **las 8 empresas con exceso promocional estadísticamente
significativo caen todas en un solo segmento**. Los adoptantes y los
listadores no se distinguen en lo financiero: la distinción entre ellos es
de discurso, no de tipo de empresa. Las sin IA son las más chicas (mediana
$13B) y las que más crecen: coincide con el margen extensivo de `05_senal_incremental.md`,
donde no hablar de IA en 2021-2022 identifica a la empresa chica en
expansión, no a la rezagada.

## Limitaciones

- Con la intensidad como dimensión, el segmento de desplegadores es en parte
  "las que más hablan de IA"; eso es deliberado y está en la tabla (8,4
  contra 1,2-1,6 frames por 1.000 párrafos).
- "Sin IA" es una regla (cero frames), no un cluster: su estabilidad de 1 no
  es evidencia de nada. En el pooled es casi vacío; su contenido está en el
  panel por año.
- Todo descansa en etiquetas de un LLM sin validación humana
  (`docs/problemas_academicos.md` #1).
