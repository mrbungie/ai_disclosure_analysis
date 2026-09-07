# Actividades de IA divulgadas: qué dicen las empresas que hacen con IA

Capa conductual concreta sobre los frames. La segmentación de `02` sabe que
los desplegadores tienen más despliegue y más resultados, pero no qué hacen.
Esta capa convierte cada frame conductual en "la empresa hace ACCIÓN sobre
OBJETO para FUNCIÓN", con etapa, destinatario, origen de la IA y todas las
entidades nombradas con su rol, y con eso arma inventarios por empresa, top
behaviours del S&P 500, composición conductual de los segmentos y fichas.
**Todo es conducta divulgada**: lo que la empresa dice que hace, con las
oraciones que lo sostienen; nadie verificó el despliegue.

Producido por `scripts/common/ai_activities_from_frames.py` (segunda pasada de
LLM, `qwen3.7-flash`, prompt `v2`) y `scripts/analytics/activity_profiles.py`.
No hay re-clustering ni modelo econométrico. La capa entra en cuatro lugares:
la descripción de los segmentos (`02`), la concreción del eje de conducta
(`03`, `activity_grounding.py`), la descomposición de la señal incremental en
estilo y actividad (`05`, `incremental_signal.py`) y la brecha de actividades
entre canales (`06`, `activity_grounding.py`); en `08` separa promoción de
respaldo conductual.

## Qué se extrajo

Población: los 17.420 textos únicos de EE.UU. con al menos un frame
`subject=firm` que afirme un concepto conductual (despliegue, piloto,
exploración, escalamiento, capacidad propia, IA de terceros, infraestructura,
talento, inversión, resultado). Los frames de puro riesgo o gobernanza no
entran: no describen una actividad. El modelo recibe **qué empresa presenta el
documento** (ticker y nombre), las oraciones numeradas y los frames ya
identificados, y devuelve una actividad por cada acción·objeto·función
distinta:

| campo | valores |
|---|---|
| `action` | deploy, develop, integrate, buy_or_license, partner, invest_infrastructure, hire_or_train, pilot_or_explore, scale, measure_outcome, govern_or_control, restrict |
| `object` | texto corto libre ("copilot", "fraud model", "data center"); se agrupa después en familias |
| `function` | snake_case libre; se agrupa en 18 familias por palabras clave |
| `target` | employees, customers, developers, internal_process, partners_or_suppliers, unspecified |
| `stage` | exploring, piloting, deployed, scaled, unspecified |
| `ai_source` | de dónde sale la IA: own, third_party, co_developed, acquired, open_source, mixed, unspecified |
| `named_entities` | todas las empresas, productos o modelos nombrados, cada uno con rol: own_product_or_brand, external_provider, external_model, partner, acquired_company, customer, distribution_channel, competitor_or_reference |
| `evidence_strength` | named_product_or_process, metric, vendor, generic |
| `sentence_ids` | evidencia |

Saber quién presenta el documento es lo que permite distinguir "AMD despliega
aceleradores Instinct" (marca propia, `own`) de "una aseguradora usa NVIDIA e
Intel" (dos proveedores externos). Una actividad sin oración de evidencia no
se crea; un objeto que no es IA explícita tampoco.

Resultado: 17.420 textos procesados, 0 errores, **27.583 actividades únicas
por empresa** (una por texto único × índice, así el boilerplate repetido no
infla), 471 de 510 empresas con al menos una. 51% de las actividades vienen
sólo de earnings calls, donde el prefiltro tiene precisión 0,59 contra 0,98
en 10-K (`00`, tabla de conjuntos etiquetados). Todo lo que sigue muestra también el valor "sólo
filings" cuando la afirmación es sobre empresas.

## Cómo se reparten las actividades

| dimensión | distribución (% de las 27.583 actividades) |
|---|---|
| acción | deploy 63 · invest_infrastructure 8 · scale 7 · pilot_or_explore 4 · integrate 3,5 · develop 3,5 · measure_outcome 3 · partner 3 · hire_or_train 2 · buy_or_license 1 · govern_or_control 1 · restrict 0 |
| etapa | deployed 58 · scaled 14 · piloting 10 · exploring 9 · unspecified 8 |
| destinatario | customers 49 · internal_process 26 · unspecified 14 · employees 9 · partners 2,5 · developers 1 |
| origen de la IA | own 73 · unspecified 16 · third_party 8 · acquired 1,5 · mixed 1 · co_developed 1 · open_source 0,1 |
| evidencia | generic 53 · named_product_or_process 29 · metric 13 · vendor 4 |
| función | unspecified 32 · operaciones y cadena de suministro 16,5 · producto 8 · IT y ciberseguridad 7 · atención al cliente 5 · desarrollo de software 5 · marketing y ventas 4 · contenido y medios 3 · fraude y riesgo 3 · datos y analítica 2 · I+D 2 · otras 7 |

Entidades nombradas (17.193 en total): 13.333 productos o marcas propias,
1.185 proveedores externos, 780 socios, 728 clientes, 551 empresas
adquiridas, 474 modelos externos, 72 canales de distribución, 70
competidores. El 35% de las actividades nombra una marca propia; el 5%
nombra un proveedor o modelo externo; el 0,7% nombra dos o más externos.

Tres lecturas. **Seis de cada diez actividades son "deploy"** y siete de cada
diez se declaran desplegadas o escaladas: el discurso es de uso, no de
exploración. **Casi un tercio no dice para qué función** y más de la mitad no
trae producto, métrica ni proveedor: la actividad típica es "usamos IA en
nuestras operaciones". **La IA que se cuenta es propia**: 73% de las
actividades la declaran propia y sólo 8% de terceros; los grandes proveedores
casi no aparecen (OpenAI/Microsoft, NVIDIA y Google suman 2,4% de las
actividades). Las empresas cuentan qué hacen con IA mucho más que con qué la
hacen.

## Qué hacen, concretamente: acción · objeto

El objeto de cada actividad se agrupa en familias (copilot o asistente,
agentes, chatbot, LLM o modelo fundacional, modelo de ML o predictivo, cómputo
e infraestructura, talento, automatización, búsqueda o recomendación,
plataforma de analítica o datos, herramienta de seguridad, visión/robótica/
autonomía, producto o servicio, producto con nombre propio, proveedor o modelo
externo, resultado financiero, marco de gobernanza, empresa adquirida, caso de
uso o proyecto). "IA, objeto sin especificar" es la palabra IA con un
sustantivo vacío ("ai capabilities"): 86% de las empresas tienen al menos una
así, y es la familia más grande.

% de las 510 empresas con al menos una actividad de cada tipo, con objetos
literales de ejemplo:

| acción · objeto | % empresas | ejemplos literales |
|---|---:|---|
| deploy · producto o servicio con IA | 55,9 | ai engine, interconnect products, advertising tools |
| deploy · modelo de ML o predictivo | 45,7 | machine learning models, ai algorithms |
| deploy · plataforma de analítica o datos | 45,1 | ai platform, expert platform, ai-powered analytics |
| deploy · producto con nombre propio | 34,1 | AIOps, AIP, Mist AI, Cerebrus |
| deploy · automatización | 33,7 | automation, automation tools |
| hire_or_train · talento | 28,6 | ai talent, workforce |
| invest_infrastructure · cómputo e infraestructura | 25,9 | ai infrastructure, data centers, compute capacity |
| scale · producto o servicio | 23,3 | data ecosystem, EFX.AI capabilities, Firefly services |
| measure_outcome · resultado financiero | 19,6 | ai revenue, productivity gains |
| deploy · cómputo e infraestructura | 19,4 | ai pc, ai accelerators, gpu |
| deploy · copilot o asistente | 18,8 | copilot, ai assistant, GitHub Copilot |
| deploy · visión, robótica o autonomía | 17,3 | autonomous driving system, Optimus robot, autonomous database |
| deploy · LLM o modelo fundacional | 14,9 | large language models, generative ai models |
| develop · producto o servicio | 14,7 | ad products, developer tools, ai experiences |
| deploy · búsqueda o recomendación | 14,3 | recommendation engine, discovery engine, personalization engine |
| deploy · agentes de IA | 13,3 | ai agents, agentic ai |
| deploy · herramienta de seguridad | 12,9 | ML-powered next-generation firewall, ai-driven security operations |
| integrate · producto o servicio | 12,4 | ai into products and services |
| partner · (socio o ecosistema) | 11,8 | ai collaboration, cloud providers |
| develop · modelo de ML o predictivo | 11,2 | ai models, Gemini model |
| deploy · chatbot | 11,8 (objeto) | chatbot, conversational ai |

Con la función declarada (acción · objeto · función), lo más común en el
S&P 500 es **desplegar un producto o herramienta de IA en operaciones**
(30%), **automatizar operaciones** (27%), **modelos predictivos en
operaciones** (19%), **plataforma de analítica en operaciones** (18%),
**producto de IA para atención al cliente** (17,5%: virtual try-on,
AI-powered experiences), **IA en el producto mismo** (17%: FSD, Apple
Intelligence, cockpits), **producto con nombre propio en operaciones** (15%:
AIOps, Flyways AI, document intelligence), **inversión en cómputo para
operaciones** (12,5%), **medir productividad** (10%), **herramientas de
marketing** (10%), **herramientas de código** (9,6%), **copilot para atención
al cliente** (9,2%), **modelos para marketing** (9,2%) y **para fraude y
riesgo** (8,8%), **motor de recomendación para marketing** (7,6%).

## Con qué: origen y proveedores nombrados

246 de 510 empresas nombran algún proveedor, modelo o socio externo en alguna
actividad (rol `external_provider`, `external_model` o `partner`).

| familia | % empresas | nombres literales más frecuentes |
|---|---:|---|
| otro proveedor nombrado (incluye consultoras, nichos y vendedores de IA nombrados por sus clientes) | 41,8 | AMD, Palantir, ServiceNow, Hugging Face, Snowflake |
| OpenAI / Microsoft | 17,3 | OpenAI, Microsoft, ChatGPT, Azure, GitHub Copilot |
| Google | 9,6 | Google Cloud, Gemini, Vertex AI |
| NVIDIA | 9,0 | NVIDIA, H100 |
| Amazon / AWS | 6,9 | AWS, Bedrock |
| Meta | 3,5 | Llama 2, Llama 3 |
| Anthropic | 2,4 | Anthropic, Claude |
| IBM | 2,4 | watsonx, Watson |
| Salesforce | 1,4 | Einstein |

OpenAI/Microsoft es el proveedor externo dominante (una de cada seis
empresas), Google y NVIDIA le siguen; AWS, Meta y Anthropic son marginales
en el discurso. Como AMD, ServiceNow o Palantir aparecen aquí sólo cuando
otra empresa los nombra, la tabla mide **a quién nombran los demás**, no
cuánto habla cada vendedor de sí mismo: eso está en `own_product_or_brand`.

## Top behaviours del S&P 500

% de las 510 empresas con filings que divulgan al menos una actividad de cada
tipo (pooled filings + calls; sólo filings al lado):

| tipo de actividad | % empresas | sólo filings |
|---|---:|---:|
| con función de negocio declarada | 88,0 | 79,0 |
| desplegada o escalada | 86,7 | 77,5 |
| despliegue interno (empleados o procesos) | 86,5 | 76,1 |
| IA propia (`ai_source=own` o develop) | 81,0 | 69,6 |
| IA de terceros, mixta u open source | 69,0 | 51,2 |
| despliegue de cara al cliente | 68,8 | 54,7 |
| nombra un producto, sistema o proceso | 64,3 | 45,3 |
| resultado cuantificado | 61,4 | 36,5 |
| nombra una marca o producto propio | 60,2 | 43,1 |
| piloto o exploración | 55,9 | 33,7 |
| inversión en infraestructura | 51,6 | 36,9 |
| proveedor, modelo o socio externo nombrado | 48,2 | 26,5 |
| talento o capacitación | 33,9 | 29,2 |
| alianza | 32,2 | 15,5 |
| gobernanza o restricción de uso | 29,0 | 26,3 |
| co-desarrollo o adquisición | 25,1 | 14,7 |
| cliente nombrado | 16,7 | 4,1 |
| adquisición o licencia | 16,3 | 12,4 |
| herramientas para desarrolladores | 8,2 | 4,9 |

Etapa máxima alcanzada por empresa: escalado 59,6%, desplegado 27,1%,
ninguna actividad 7,6%, sólo explorando o pilotando 3,6%.

La brecha entre pooled y sólo filings es la de `06` vista desde las
actividades: el resultado cuantificado (61 → 37), el proveedor nombrado (48 →
27) y el cliente nombrado (17 → 4) se cuentan en la call; el despliegue
interno y la gobernanza casi no cambian de canal.

## Qué hace cada segmento

Top acción · objeto por segmento (`02_segmentacion.md`), % de empresas del
segmento:

| Desplegadores de producto (113) | % | Adoptantes con gobernanza (222) | % | Listadores de riesgo (158) | % |
|---|---:|---|---:|---|---:|
| deploy · producto o servicio | 95 | deploy · producto o servicio | 58 | deploy · producto o servicio | 29 |
| deploy · plataforma de analítica | 88 | deploy · modelo predictivo | 46 | deploy · modelo predictivo | 25 |
| deploy · producto con nombre propio (AIOps, AIP, Mist AI) | 84 | deploy · plataforma de analítica | 42 | deploy · plataforma de analítica | 22 |
| deploy · modelo predictivo | 80 | deploy · automatización | 33 | deploy · automatización | 16 |
| scale · (portafolio, integración) | 67 | deploy · producto con nombre propio (Flyways AI, Compliance Coach, Apple Intelligence) | 29 | hire_or_train · talento | 13 |
| deploy · automatización | 65 | hire_or_train · talento | 26 | deploy · producto con nombre propio | 8 |
| invest_infrastructure · cómputo (data centers) | 62 | invest_infrastructure · cómputo | 23 | measure_outcome · resultado financiero | 8 |
| hire_or_train · talento | 61 | deploy · visión/robótica (cámaras, robots) | 16 | deploy · copilot o asistente | 7 |
| deploy · cómputo (aceleradores, AI PC) | 56 | deploy · copilot o asistente | 15 | pilot_or_explore | 8 |

Composición conductual, % de empresas del segmento con ≥1 actividad de cada
tipo, y origen de la IA:

| | Desplegadores | Adoptantes c/gob. | Listadores de riesgo | Sin IA |
|---|---:|---:|---:|---:|
| actividades (mediana) | **112** | 18 | 5 | 0 |
| despliegue de cara al cliente | **99** | 73 | 46 | 24 |
| despliegue interno | 99 | 89 | 80 | 24 |
| IA propia | **100** | 90 | 61 | 24 |
| marca propia nombrada | **97** | 64 | 34 | 18 |
| proveedor, modelo o socio externo nombrado | **89** | 46 | 27 | 12 |
| co-desarrollo o adquisición | **66** | 19 | 7 | 0 |
| cliente nombrado | **48** | 10 | 5 | 0 |
| inversión en infraestructura | **90** | 55 | 25 | 0 |
| resultado cuantificado | **94** | 65 | 37 | 18 |
| piloto o exploración | 77 | 57 | 44 | 12 |
| alianza | 71 | 29 | 13 | 0 |
| talento o capacitación | 70 | 32 | 15 | 0 |
| adquisición o licencia | 46 | 11 | 4 | 0 |
| gobernanza o restricción | 55 | 27 | 17 | 6 |
| % de actividades para clientes / internas (media) | 51 / 33 | 27 / 60 | 21 / 66 | — |
| origen: propia / de terceros / no dice (% de actividades) | 80 / 6 / 10 | 55 / 12 / 30 | 46 / 17 / 34 | — |
| etapa máxima = escalado / desplegado / sólo piloto o exploración / ninguna | 93 / 7 / 0 / 0 | 64 / 27 / 3 / 4 | 34 / 44 / 7 / 11 | 24 / 0 / 0 / 76 |

(Las "sin IA" con alguna actividad la tienen en calls; el segmento se define
sobre filings.)

**Los segmentos ahora tienen contenido tangible.** Los desplegadores de
producto venden IA propia con marca: 97% nombra una marca propia, 84% un
producto con nombre, 89% nombra además un proveedor o socio externo, 62%
invierte en data centers, 48% nombra un cliente, dos tercios co-desarrollan o
adquieren; 80% de sus actividades declaran la IA como propia. Los adoptantes
con gobernanza automatizan operaciones y despliegan modelos predictivos hacia
adentro (60% de sus actividades internas), con IA propia sin marca (20% de
sus actividades la nombra) y en un tercio de las actividades sin decir de
dónde sale; son el segmento de la visión y la robótica industrial. Los
listadores de riesgo tienen cinco actividades en la mediana, una de cada
diez ninguna, el 17% de lo que describen es IA de terceros y el 34% no dice
de dónde sale.

## Fichas: inventario de actividades por empresa

Cada línea es acción · familia de objeto (n): objetos literales | función |
destinatario | etapa | origen | proveedores externos | marcas propias | % con
producto, cifra o proveedor.

**Microsoft** (Desplegadores, 966 actividades)

- deploy · copilot (168): Copilot, GitHub Copilot, Security Copilot | operaciones, desarrollo de software | clientes y empleados | escalado | propia | ext.: OpenAI, Azure OpenAI | propias: GitHub Copilot, Microsoft 365 Copilot | 86% concreto
- invest_infrastructure · cómputo (142): AI infrastructure, compute capacity, data centers | interno | escalado | propia | ext.: OpenAI, NVIDIA, AMD | propias: Azure | 47%
- deploy · proveedor/modelo externo (37): Azure OpenAI Service, Azure AI | desarrollo de software | clientes y desarrolladores | ext.: OpenAI, ChatGPT | propias: Azure AI | 86%
- deploy · producto (47): AI-backed tools, LinkedIn AI features, Dynamics 365 | 55%
- hire_or_train · talento (32): AI talent | 9%
- deploy · otros (24): Nuance DAX, intelligent recaps, Teams Premium | ext.: GPT-4, Epic | 54%

Microsoft despliega copilots para clientes y empleados, invierte y escala
cómputo con OpenAI, NVIDIA y AMD como proveedores, ofrece Azure OpenAI a
terceros, y la IA que describe es propia en casi todas sus actividades.

**ServiceNow** (Desplegadores, 455): agentes (47: AI agents, agentic AI) para
operaciones, clientes y empleados, con NVIDIA, Google y AWS como socios; Now
Assist (36) para atención al cliente, 86% concreto; Now Platform y RaptorDB;
Celonis como socio; todo `own`.

**HPE** (Desplegadores, 443): servidores de IA (60) a clientes con NVIDIA
como proveedor (H100, Blackwell, Grace Hopper), 67% concreto; GreenLake,
Private Cloud AI, Cray EX, Aruba Central, InfoSight como marcas propias; El
Capitan; mide ingresos de HPC & AI (16).

**Etsy** (Desplegadores, 175): algoritmos de búsqueda y modelos de ML (30)
para contenido y marketing; motor de recomendación (26: XWalk, Etsy Lens);
LLMs de terceros y OpenAI, Google Cloud como proveedores; machine translation;
contrata ML engineers.

**Paychex** (Desplegadores, 153): Flex Intelligence Engine, WISE; modelos
predictivos (20: labor forecasting) para marketing y operaciones; Retention
Insights; Recruiting Copilot y Voice Assist (con Google Assistant).

**JPMorgan** (Adoptantes con gobernanza, 31): "AI/ML" en fraude, riesgo y
marketing (8), interno, **0% concreto y origen no declarado**; modelos AI/ML
(3); talento (4); risk assessment system y lead optimization tools (2) en
AWM. JPMorgan describe IA interna en fraude y operaciones sin nombrar un
producto, un proveedor ni una cifra.

**State Street** (Adoptantes, 28): modelos cuantitativos (7) en fraude y
operaciones; State Street Alpha y State Street Digital como marcas; "IA de
terceros" sin nombre (1); NAV calculation.

**Lowe's** (Adoptantes, 19): herramientas de IA en operaciones con NVIDIA,
OpenAI y Palantir como proveedores (4, `third_party`); Mylow, asesor virtual
para clientes; piloto de IA generativa; freight flow optimization.

**Danaher** (Adoptantes, 23): DxI 9000, CytoFLEX, Mica (imaging con ML),
CellXpress.ai, IDBS Polar Insight: producto propio con nombre (100% concreto
en esas líneas); "IA" en healthcare delivery sin origen (4).

**Cincinnati Financial** (Adoptantes, 13): modelos de underwriting y pricing
(8), interno; analítica con BCG y Deloitte (piloto, `third_party`); un
director con experiencia en IA.

**Bank of America** (Listadores de riesgo, 59): Erica en asistente (7),
chatbot (4) y producto (4: CashPro), 75-100% concreto; predictive language
program para fraude (10); gobernanza de modelos (5); "terceros" sin nombre.
Es el listador de riesgo atípico: el segmento se define por la proporción
de riesgo en el filing, no por la ausencia de actividad.

**Nike** (Listadores, 30): modelos de ML (5: insights model) para operaciones
y analítica; SNKRS App, NIKE App; búsqueda y recomendación con IA (2, `mixed`);
robótica en cadena de suministro (2).

**Comerica** (2): intelligent automation, mobile check scanning, sin origen.
**Howmet** (3): "AI" en operaciones, 0% concreto. **TransDigm** (1):
automatización de procesos.

## Limitaciones

- Conducta divulgada, no observada. La evidencia es oraciones del propio
  documento; `evidence_strength` dice cuán concreta es, no si es cierta.
- `object` y `function` son texto libre agrupado por palabras clave; 32% de
  las actividades no declara función y 7% cae en "otras".
- `ai_source=unspecified` en 16% de las actividades: el texto no dice de
  dónde sale la IA, y el modelo tiene instrucción de no inferirlo.
- 51% de las actividades vienen sólo de calls, donde el prefiltro tiene
  precisión 0,59; los % de empresas se muestran también sólo con filings.
- Segunda pasada de LLM sobre etiquetas de la primera: hereda el error de
  frames y agrega el suyo. La validación humana (`ui-validator/`, pestaña
  Actividades) mide existencia, precisión por campo y recall por párrafo.
