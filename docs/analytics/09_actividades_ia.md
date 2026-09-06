# Actividades de IA divulgadas: qué dicen las empresas que hacen con IA

Capa conductual concreta sobre los frames. La segmentación de `02` sabe que
los desplegadores tienen más despliegue y más resultados, pero no qué hacen.
Esta capa convierte cada frame conductual en "la empresa hace ACCIÓN sobre
OBJETO para FUNCIÓN" y con eso arma inventarios por empresa, top behaviours
del S&P 500, composición conductual de los segmentos y fichas de empresas.
**Todo es conducta divulgada**: lo que la empresa dice que hace, con la
evidencia (oraciones) que lo sostiene; nadie verificó el despliegue.

Producido por `scripts/common/ai_activities_from_frames.py` (segunda pasada de
LLM, `qwen3.7-flash`, prompt `v1`) y `scripts/analytics/activity_profiles.py`.
No hay re-clustering ni modelo econométrico.

## Qué se extrajo

Población: los 17.420 textos únicos de EE.UU. con al menos un frame
`subject=firm` que afirme un concepto conductual (despliegue, piloto,
exploración, escalamiento, capacidad propia, IA de terceros, infraestructura,
talento, inversión, resultado). Los frames de puro riesgo o gobernanza no
entran: no describen una actividad. El modelo recibe las oraciones numeradas
más los frames ya identificados y devuelve sólo actividades:

| campo | valores |
|---|---|
| `action` | deploy, develop, integrate, buy_or_license, partner, invest_infrastructure, hire_or_train, pilot_or_explore, scale, measure_outcome, govern_or_control, restrict |
| `object` | texto corto libre ("copilot", "fraud model", "data center") |
| `function` | snake_case libre; se agrupa después en 18 familias por palabras clave |
| `target` | employees, customers, developers, internal_process, partners_or_suppliers, unspecified |
| `stage` | exploring, piloting, deployed, scaled, unspecified |
| `provider_or_model` | texto corto; "proprietary"; "unspecified"; se agrupa en familias |
| `evidence_strength` | named_product_or_process, metric, vendor, generic |
| `sentence_ids` | evidencia |

Resultado: 17.420 textos procesados, 0 errores, **27.468 actividades únicas
por empresa** (una por texto único × índice, así el boilerplate repetido no
infla), 471 de 510 empresas con al menos una. 48% de las actividades vienen
sólo de earnings calls (misma advertencia de precisión que `06`: 0,60 en
calls contra 0,98 en 10-K).

## Cómo se reparten las actividades

| dimensión | distribución (% de las 27.468 actividades) |
|---|---|
| acción | deploy 60 · invest_infrastructure 8 · develop 8 · integrate 7 · scale 4 · partner 3 · pilot_or_explore 3 · hire_or_train 2 · measure_outcome 2 · buy_or_license 1 · govern_or_control 1 · restrict 0 |
| etapa | deployed 62 · scaled 12 · piloting 10 · exploring 8 · unspecified 7 |
| destinatario | customers 43 · internal_process 26 · unspecified 17 · employees 11 · partners 3 · developers 1 |
| evidencia | generic 46 · named_product_or_process 40 · metric 9 · vendor 5 |
| proveedor | proprietary 48 · unspecified 40 · otro nombrado 6 · OpenAI/Microsoft 1,5 · NVIDIA 1,5 · terceros sin nombre 1 · Google 0,6 · AWS 0,3 · IBM 0,3 · Meta 0,2 · Anthropic 0,1 |
| función | unspecified 31 · operaciones y cadena de suministro 16 · producto 7 · IT y ciberseguridad 6 · atención al cliente 5 · desarrollo de software 5 · marketing y ventas 4 · contenido y medios 3 · datos y analítica 3 · fraude y riesgo 3 · I+D 3 · otras 9 |

Tres lecturas. **Seis de cada diez actividades son "deploy"** y dos tercios se
declaran desplegadas o escaladas: el discurso es de uso, no de exploración.
**Casi la mitad no dice para qué función** y otro 46% no trae ni producto,
ni métrica, ni proveedor: la actividad típica es "usamos IA en nuestras
operaciones". **Los proveedores casi no se nombran**: 88% es propio o sin
especificar; OpenAI/Microsoft, NVIDIA y Google suman 3,6%. Las empresas
cuentan qué hacen con IA mucho más que con qué la hacen.

## Top behaviours del S&P 500

% de las 510 empresas con filings que divulgan al menos una actividad de cada
tipo (pooled filings + calls; entre paréntesis, sólo filings):

| tipo de actividad | % empresas | sólo filings |
|---|---:|---:|
| despliegue interno (empleados o procesos) | 84,5 | 76,1 |
| IA propia (desarrolla o declara proprietary) | 78,2 | 66,1 |
| nombra un producto, sistema o proceso | 74,9 | 59,6 |
| despliegue de cara al cliente | 67,8 | 54,7 |
| inversión en infraestructura | 55,7 | 40,0 |
| resultado cuantificado | 55,7 | 34,9 |
| proveedor de terceros nombrado | 53,3 | 38,8 |
| piloto o exploración | 52,4 | 30,8 |
| talento o capacitación | 35,3 | 30,2 |
| alianza con un proveedor de IA | 31,4 | 18,4 |
| gobernanza o restricción de uso | 28,2 | 25,3 |
| adquisición o licencia | 16,5 | 12,7 |
| herramientas para desarrolladores | 8,0 | 5,3 |

Etapa máxima alcanzada por empresa: escalado 57,5%, desplegado 30,8%,
ninguna actividad 7,6%, sólo explorando o pilotando 3,2%.

La brecha entre pooled y sólo filings es la de `06` vista desde las
actividades: el resultado cuantificado y el piloto se cuentan en la call
(55,7 → 34,9 y 52,4 → 30,8); el despliegue interno y la gobernanza casi no
cambian de canal.

## Composición conductual de los segmentos

% de empresas del segmento (`02_segmentacion.md`) con al menos una actividad
de cada tipo:

| | Desplegadores de producto | Adoptantes con gobernanza | Listadores de riesgo | Sin IA |
|---|---:|---:|---:|---:|
| empresas | 113 | 222 | 158 | 17 |
| actividades (mediana) | **108** | 19,5 | 6 | 0 |
| despliegue de cara al cliente | **100** | 71 | 45 | 24 |
| despliegue interno | 97 | 88 | 78 | 24 |
| IA propia | 99 | 86 | 59 | 24 |
| proveedor de terceros nombrado | **90** | 53 | 32 | 6 |
| inversión en infraestructura | **92** | 57 | 33 | 6 |
| resultado cuantificado | **90** | 62 | 27 | 6 |
| piloto o exploración | 78 | 55 | 35 | 6 |
| alianza | 68 | 28 | 13 | 6 |
| talento o capacitación | 66 | 33 | 20 | 0 |
| adquisición o licencia | 50 | 8 | 6 | 0 |
| gobernanza o restricción | 49 | 29 | 15 | 0 |
| herramientas para desarrolladores | 29 | 2 | 2 | 0 |
| % de actividades para clientes (media) | 45 | 25 | 18 | — |
| % de actividades internas (media) | 35 | 58 | 62 | — |
| etapa máxima = escalado | 93 | 61 | 32 | 12 |

(Las 17 "sin IA" con alguna actividad la tienen en calls; el segmento se
define sobre filings.)

Funciones por segmento (% de las actividades del segmento): operaciones 13
/ 22 / 19; IT y ciberseguridad 7,5 / 3 / 2; desarrollo de software 5 / 3 /
3; fraude y riesgo 2 / 5 / 4; atención al cliente 4 / 6 / 5; marketing y
ventas 4 / 5 / 6; sin especificar 32 / 26 / 30.

**Los segmentos ahora tienen contenido tangible.** Los desplegadores de
producto hacen de todo, y lo hacen con nombre y número: 100% despliega de
cara al cliente, 92% invierte en infraestructura, 90% nombra un proveedor y
90% reporta un resultado con cifra; la mitad compra o licencia y dos tercios
tienen alianzas. Los adoptantes con gobernanza despliegan sobre todo hacia
adentro (58% de sus actividades son internas), en operaciones, fraude y
riesgo, y con IA propia sin proveedor nombrado; un tercio pilota o entrena.
Los listadores de riesgo tienen seis actividades en la mediana, tres de
cada cinco escalan nada, y lo poco que describen es despliegue interno en
operaciones sin producto, métrica ni proveedor.

## Fichas

| empresa | segmento | n | acciones principales | funciones | objetos típicos | etapa | clientes / interno | proveedores | con nombre / con cifra |
|---|---|---:|---|---|---|---|---|---|---|
| MSFT | Desplegadores | 950 | deploy, invest_infrastructure, integrate | operaciones, desarrollo de software | ai infrastructure, copilot, compute capacity, ai talent | escalado | 39% / 30% | proprietary, OpenAI, Azure | 35% / 21% |
| NOW | Desplegadores | 418 | deploy, integrate, partner | operaciones, atención al cliente | generative ai, ai platform, ai agents | escalado | 45% / 39% | proprietary, Now Assist, NVIDIA | 48% / 12% |
| HPE | Desplegadores | 385 | deploy, develop, invest_infrastructure | operaciones, IT | ai systems, ai servers, ai solutions | escalado | 48% / 23% | proprietary, NVIDIA, Determined AI | 43% / 12% |
| ETSY | Desplegadores | 172 | deploy, invest_infrastructure, pilot_or_explore | contenido, marketing | search engine, ml models, llms | escalado | 61% / 30% | proprietary, modelos fundacionales de terceros | 33% / 9% |
| PAYX | Desplegadores | 163 | deploy, invest_infrastructure, develop | RR.HH., atención al cliente | ai models, flex assistant | escalado | 49% / 43% | proprietary, Flex Assistant | 36% / 10% |
| JPM | Adoptantes c/gob. | 53 | deploy, invest_infrastructure, hire_or_train | operaciones, fraude y riesgo | ai/ml technologies, ai research and capabilities | escalado | 21% / 60% | proprietary | 4% / 6% |
| STT | Adoptantes c/gob. | 30 | deploy, integrate, develop | operaciones, fraude y riesgo | quantitative models, automation | escalado | 7% / 73% | proprietary | 13% / 13% |
| DHR | Adoptantes c/gob. | 25 | deploy, invest_infrastructure, partner | operaciones, producto | ai technologies, ai expert | escalado | 28% / 56% | proprietary, Microsoft Copilot | 40% / 0% |
| CINF | Adoptantes c/gob. | 25 | deploy, develop, hire_or_train | marketing, operaciones financieras, gobernanza | underwriting and pricing models, predictive models, director with ai skills | escalado | 0% / 88% | proprietary | 24% / 4% |
| LOW | Adoptantes c/gob. | 19 | deploy, develop, pilot_or_explore | atención al cliente, operaciones | virtual advisor, ai cybersecurity strategy | escalado | 47% / 37% | proprietary, NVIDIA, OpenAI, Palantir | 32% / 0% |
| BAC | Listadores de riesgo | 63 | deploy, invest_infrastructure, govern_or_control | atención al cliente, fraude y riesgo | erica, models, predictive language program | escalado | 27% / 59% | proprietary, Erica | 37% / 14% |
| NKE | Listadores de riesgo | 28 | deploy, scale, invest_infrastructure | operaciones, marketing, producto | machine learning applications, ai-enabled tools | escalado | 29% / 68% | proprietary, Datalogue | 39% / 7% |
| CMA | Listadores de riesgo | 3 | deploy, integrate | operaciones financieras | real-time payments system, mobile check scanning | desplegado | 67% / 33% | — | 100% / 0% |
| HWM | Listadores de riesgo | 3 | deploy | operaciones | ai, ai in tests | desplegado | 0% / 100% | — | 0% / 0% |
| TDG | Listadores de riesgo | 1 | invest_infrastructure | operaciones | process and test automation | explorando | 0% / 100% | — | 0% / 0% |

Ahora se puede decir, en vez de "MSFT = desplegador, 42% despliegue":
Microsoft integra IA generativa en productos de cara al cliente, despliega
copilots en flujos internos, invierte en infraestructura y capacidad de
cómputo, se apoya en OpenAI y Azure, y cuantifica resultados en una de cada
cinco actividades. JPMorgan despliega sobre todo hacia adentro (60%), en
operaciones y fraude, con IA propia, casi sin nombrar productos (4%) ni
cifras (6%). Nike describe aplicaciones de machine learning internas en
operaciones y marketing; Howmet y TransDigm, una o tres frases sobre
automatización de pruebas. BAC es el listador de riesgo atípico: 63
actividades, Erica, y gobernanza explícita, lo que muestra que el segmento
se define por la proporción de riesgo en el filing y no por la ausencia de
actividad.

## Limitaciones

- Conducta divulgada, no observada. La evidencia es oraciones del propio
  documento; `evidence_strength` dice cuán concreta es, no si es cierta.
- `object` y `function` son texto libre del LLM agrupado por palabras
  clave; 31% de las actividades no declara función y 9% cae en "otras".
- Los proveedores de IA (AMD, Adobe, NVIDIA) aparecen como "proveedor
  nombrado" cuando nombran sus propios productos; para ellos proveedor y
  producto coinciden.
- Segunda pasada de LLM sobre etiquetas de la primera: hereda el error de
  frames y agrega el suyo; sin validación humana
  (`docs/problemas_academicos.md` #1).
