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

## Qué hacen, concretamente: acción · objeto

El objeto de cada actividad se agrupa en familias (copilot o asistente,
agentes, chatbot, LLM o modelo fundacional, modelo de ML o predictivo,
cómputo e infraestructura, talento, automatización, búsqueda o
recomendación, plataforma de analítica o datos, herramienta de seguridad,
visión/robótica/autonomía, producto o servicio, producto con nombre propio,
proveedor o modelo externo, resultado financiero, marco de gobernanza,
empresa adquirida o licencia, caso de uso o proyecto). "IA, objeto sin
especificar" es la palabra IA con un sustantivo vacío ("ai capabilities",
"generative ai tools"): 87% de las empresas tienen al menos una así, y es la
familia más grande.

% de las 510 empresas con al menos una actividad de cada tipo, con objetos
literales de ejemplo:

| acción · objeto | % empresas | ejemplos literales |
|---|---:|---|
| deploy · producto o servicio | 54,3 | ai engine, firefly services, advertising tools |
| deploy · modelo de ML o predictivo | 45,7 | machine learning models, ai algorithms |
| deploy · plataforma de analítica o datos | 44,3 | ai platform, ai-powered analytics, data analytics |
| deploy · producto con nombre propio | 41,8 | AIP, AIOps, DSO.ai, Firefly |
| deploy · automatización | 38,0 | automation, intelligent automation |
| hire_or_train · talento | 28,6 | ai talent, workforce |
| develop · producto o servicio | 25,9 | ai-driven software, interconnect products |
| invest_infrastructure · cómputo e infraestructura | 22,5 | ai infrastructure, data centers, compute capacity |
| deploy · copilot o asistente | 20,4 | copilot, ai assistant, agentforce, ai overviews |
| deploy · cómputo e infraestructura | 20,2 | ai pc, ai accelerators, ai workloads |
| integrate · producto o servicio | 19,4 | ai into products, ai into offerings |
| develop · modelo de ML o predictivo | 19,0 | ai models, machine learning models |
| deploy · visión, robótica o autonomía | 16,7 | autonomous driving software, autonomous database |
| develop · producto con nombre propio | 16,7 | Omniverse, MI350 series, Blackwell |
| integrate · producto con nombre propio | 15,3 | Meta AI, Firefly, ADEM |
| deploy · agentes de IA | 14,5 | ai agents, agentic ai |
| partner · proveedor o modelo externo | ~14 | OpenAI, NVIDIA, cloud providers |
| deploy · LLM o modelo fundacional | 13,3 | large language models, llms |
| deploy · búsqueda o recomendación | 13,1 | recommendation engine, personalization engine, search engine |
| measure_outcome · resultado financiero | 12,5 | ai revenue, productivity savings, cost savings |
| deploy · chatbot | 12,7 (objeto) | chatbot, conversational ai |
| deploy · herramienta de seguridad | 12,5 (objeto) | ml-powered firewall, threat graph |

Con la función declarada (acción · objeto · función), lo más común en el
S&P 500 es **desplegar un producto o herramienta de IA en operaciones**
(28%), **automatizar operaciones** (27%), **desplegar un producto con nombre
propio en operaciones** (19%: AIOps, Cortex XSIAM, Mist AI, AIP), **modelos
predictivos en operaciones** (18%), **IA en el producto mismo** (15%: FSD,
Apple Intelligence, cockpits), **producto de IA para atención al cliente**
(12,5%: virtual try-on, guest journey), **copilot o asistente para atención
al cliente** (9,6%), **modelos predictivos para marketing** (8,4%) y **para
fraude y riesgo** (7,3%), **motor de recomendación para marketing** (7,1%),
**herramientas de código** (7,1%: text-to-code, coding tools).

## Con qué: proveedores nombrados

246 de 510 empresas nombran algún proveedor o modelo externo en alguna
actividad; 88% de las actividades no lo hacen.

| familia | % empresas | nombres literales más frecuentes |
|---|---:|---|
| proveedor nombrado fuera de las grandes (incluye productos propios de vendedores de IA) | 42,5 | Firefly, Adobe Sensei, Mist AI, Kensho, Palantir, AMD, Oracle, Nuance |
| OpenAI / Microsoft | 16,3 | OpenAI, Microsoft, GitHub Copilot, Azure OpenAI |
| NVIDIA | 9,4 | NVIDIA, NVIDIA AI Enterprise, Hopper |
| Google | 6,3 | Gemini, Google Cloud, Vertex AI |
| Amazon / AWS | 3,9 | AWS, Rekognition, Bedrock |
| Meta | 1,2 | Meta AI, Llama |
| Anthropic | 1,0 | Anthropic, Claude (junto a Cohere, Mistral, Stability, AI21 en listas de modelos) |
| IBM | 1,0 | watsonx, Watson |
| Salesforce | 1,0 | Einstein, Agentforce |

OpenAI/Microsoft es el proveedor externo dominante (uno de cada seis), NVIDIA
el segundo; Google, AWS, Meta y Anthropic son marginales en el discurso.

## Top behaviours del S&P 500

% de las 510 empresas con filings que divulgan al menos una actividad de cada
tipo (pooled filings + calls; sólo filings entre paréntesis):

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
ninguna actividad 7,6%, sólo explorando o pilotando 3,2%. La brecha entre
pooled y sólo filings es la de `06` vista desde las actividades: el resultado
cuantificado y el piloto se cuentan en la call; el despliegue interno y la
gobernanza casi no cambian de canal.

## Qué hace cada segmento

Top acción · objeto por segmento (`02_segmentacion.md`), % de empresas del
segmento:

| Desplegadores de producto (113) | % | Adoptantes con gobernanza (222) | % | Listadores de riesgo (158) | % |
|---|---:|---|---:|---|---:|
| deploy · producto o servicio | 96 | deploy · producto o servicio | 52 | deploy · producto o servicio | 32 |
| deploy · producto con nombre propio (AIP, AIOps, DSO.ai, Firefly) | 90 | deploy · modelo predictivo | 46 | deploy · modelo predictivo | 27 |
| deploy · plataforma de analítica | 82 | deploy · plataforma de analítica | 45 | deploy · automatización | 21 |
| deploy · modelo predictivo | 78 | deploy · automatización | 39 | deploy · plataforma de analítica | 20 |
| deploy · automatización | 65 | deploy · producto con nombre propio (Apple Intelligence, Compliance Coach) | 38 | deploy · producto con nombre propio (Erica, Photo Selector) | 17 |
| develop · producto o servicio | 64 | hire_or_train · talento | 26 | hire_or_train · talento | 16 |
| invest_infrastructure · cómputo (data centers, compute capacity) | 61 | develop · producto o servicio | 21 | integrate · producto o servicio | 9 |
| deploy · cómputo (AI PC, aceleradores) | 60 | deploy · visión/robótica/autonomía (cámaras, robots móviles, vehículos) | 19 | pilot_or_explore | 8 |
| hire_or_train · talento | 56 | deploy · copilot o asistente | 18 | develop · producto o servicio | 8 |
| integrate · producto o servicio | 51 | invest_infrastructure · cómputo | 16 | invest_infrastructure · cómputo | 7 |

Con función: los desplegadores despliegan producto de IA en operaciones
(64%), producto con nombre en operaciones (49%), automatizan (43%), ponen IA
en el producto (42%), en IT y ciberseguridad (32%) y en analítica (31%). Los
adoptantes con gobernanza automatizan operaciones (28%), despliegan producto
en operaciones (24%), modelos predictivos en operaciones (17%) y en fraude y
riesgo (9%), y atención al cliente (9,5%). Los listadores de riesgo:
automatización en operaciones (15%), producto en operaciones (11%), modelos
predictivos (9%), y nada más pasa de 7%.

Composición conductual, % de empresas del segmento con ≥1 actividad de cada
tipo:

| | Desplegadores | Adoptantes c/gob. | Listadores de riesgo | Sin IA |
|---|---:|---:|---:|---:|
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
| % de actividades para clientes / internas (media) | 45 / 35 | 25 / 58 | 18 / 62 | — |
| etapa máxima = escalado | 93 | 61 | 32 | 12 |

(Las "sin IA" con alguna actividad la tienen en calls; el segmento se define
sobre filings.)

**Los segmentos ahora tienen contenido tangible.** Los desplegadores de
producto venden IA: producto con nombre propio, plataforma, aceleradores,
copilots, y lo hacen con proveedor y con cifra. Los adoptantes con
gobernanza automatizan operaciones y despliegan modelos predictivos hacia
adentro (fraude, riesgo, forecasting), con IA propia y sin proveedor
nombrado, y son el segmento de la visión y la robótica industrial. Los
listadores de riesgo tienen seis actividades en la mediana: automatización y
algún modelo, casi siempre sin producto, métrica ni proveedor.

## Fichas: inventario de actividades por empresa

Cada línea es acción · familia de objeto (n): objetos literales | función |
destinatario | etapa | proveedor | % con producto, cifra o proveedor.

**Microsoft** (Desplegadores, 950 actividades)

- deploy · copilot (154): Copilot, GitHub Copilot, Microsoft 365 Copilot | operaciones, desarrollo de software | clientes y empleados | escalado | 96% concreto
- invest_infrastructure · cómputo (132): AI infrastructure, compute capacity | interno | escalado | OpenAI | 48%
- deploy · proveedor/modelo externo (40): Azure OpenAI Service, Azure AI | IT | clientes | escalado | OpenAI | 90%
- deploy · producto (38): Cognitive Services, AI-backed tools | operaciones | clientes y empleados | 68%
- hire_or_train · talento (33): AI talent, responsible AI team | empleados | 15%
- scale · cómputo (31): data center capacity, edge workloads | 35%
- deploy · producto con nombre (27): intelligent recaps, Azure SQL | clientes y desarrolladores | 100%

Microsoft integra IA generativa en productos de cara al cliente, despliega
copilots para empleados y desarrolladores, invierte y escala infraestructura
de cómputo, se apoya en OpenAI y Azure, y una de cada cinco actividades trae
una cifra.

**ServiceNow** (Desplegadores, 418): deploy · copilot (43: Now Assist) para
operaciones y atención al cliente, escalado, 98% concreto; deploy · agentes
(41: AI agents, agentic AI) para empleados y clientes; deploy · plataforma
(30: Now Platform, AI Platform); producto (23: text-to-code tool); integra IA
con Accenture, Deloitte, EY, KPMG; automatización de workflows (15).

**HPE** (Desplegadores, 385): deploy · cómputo (31: AI servers) a clientes,
84% concreto; producto (28: networking, integrated systems) en IT; producto
con nombre (21: Private Cloud AI, ML development environment) con NVIDIA;
plataforma (19: GreenLake edge-to-cloud); invest_infrastructure · HPC y
soluciones de IA con NVIDIA.

**Etsy** (Desplegadores, 172): deploy · modelos de ML (23: search algorithms,
ML models) para contenido y marketing, 83% concreto; deploy · búsqueda y
recomendación (21: search engine, recommendation engine, XWalk); machine
translation y shop manager chat; LLMs de terceros (10); contrata ML
engineers (6); desarrolla modelos neuronales (semantic bridge model).

**Paychex** (Desplegadores, 163): deploy · modelos predictivos (28:
AI-powered labor forecasting) para analítica, 61%; plataforma de analítica
(21: retention insights tool, people analytics) para RR.HH.; producto (12:
AI-driven HCM, WISE engine, proposal system); copilot (7: recruiting copilot,
voice assistant) para RR.HH. y finanzas, 100% concreto.

**JPMorgan** (Adoptantes con gobernanza, 53): deploy · IA sin objeto (20:
"AI/ML") para fraude y riesgo, interno, escalado, **0% concreto**;
invest_infrastructure · "AI/ML capabilities" (5); hire_or_train · talento (4);
workflow tools (3) para fraude y operaciones; develop · automated solutions
(2) para fraude. JPMorgan despliega hacia adentro, en fraude, riesgo y
operaciones, con IA propia, sin nombrar un solo producto ni proveedor.

**State Street** (Adoptantes, 30): deploy · modelos cuantitativos (9) para
operaciones y fraude, interno; automatización (3); NAV calculation y fund
accounting (2); HR inquiry system en piloto; mide productividad (2).

**Cincinnati Financial** (Adoptantes, 25): deploy · modelos predictivos (20:
underwriting models, pricing models) para marketing y operaciones de seguros,
interno, escalado; develop · underwriting and pricing models (2); un director
con habilidades de IA (gobernanza).

**Danaher** (Adoptantes, 25): copilot (2: Microsoft Copilot) para desarrollo
de software; immunoassay analyzer con IA; AI-powered digital pathology con
Indica Labs; predictive algorithms en producto; AI expert contratado para
gobernanza.

**Lowe's** (Adoptantes, 19): pilotea IA generativa en operaciones (3);
herramientas de atención al cliente (2); virtual advisor; forecasting and
planning tools; desarrolla con NVIDIA, OpenAI y Palantir (1, explorando).

**Bank of America** (Listadores de riesgo, 63): deploy · copilot (8: Erica,
virtual financial assistant, advisor insights tool) para clientes y
empleados, 100% concreto; chatbot (3: Erica); modelos (9: predictive language
program) para fraude y operaciones; govern_or_control · model risk (3);
patentes de IA (5). Es el listador de riesgo atípico: el segmento se define
por la proporción de riesgo en el filing, no por la ausencia de actividad.

**Nike** (Listadores, 28): scale · machine learning applications (4) en
operaciones y marketing, piloto; ML search model, insights model (3);
analítica y data science (3+3); compra la plataforma Datalogue (1); robotics
and automation (1).

**Comerica** (Listadores, 3): real-time payments system, mobile check
scanning (2) para clientes; intelligent automation (1).
**Howmet** (3): "AI" (2), "AI in tests" (1), 0% concreto.
**TransDigm** (1): process and test automation, explorando.

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
