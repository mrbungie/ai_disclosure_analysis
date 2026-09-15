# Panel de Patentes de Inteligencia Artificial (OECD 2025) para el S&P 500 Extendido

Este documento detalla la reconstrucción del universo de patentes de inteligencia artificial (IA) para las 547 firmas del universo extendido del S&P 500 (`configs/us/universe.csv`), utilizando los datos globales de **Google Patents** en Google BigQuery (`patents-public-data.patents.publications`) y la taxonomía estándar revisada de la **OECD (2025)**.

*(Actualizado 2026-09-11 tras una auditoría de los alias de patentes: el join original hacía match exacto entre `assignee_harmonized.name` y una lista de 7 variantes de sufijo legal por empresa — cualquier subsidiaria, entidad histórica o abreviación propia de Google Patents no listada literalmente quedaba fuera del conteo. TE Connectivity, por ejemplo, salía con ~0 patentes de IA porque casi todas sus patentes están registradas bajo "Tyco Electronics" (nombre anterior a 2011), no bajo "TE Connectivity". El join ahora usa `REGEXP_CONTAINS` sobre un regex por empresa, con los alias reales descubiertos y verificados contra BigQuery -- ver §1.1. También se corrigieron tres pares de tickers duplicados en `configs/us/universe.csv` que compartían CIK (UA/UAA, DISCA/DISCK, DISH/ECHO), que hacían contar la misma patente dos veces.)*

---

## 1. Universo y Resolución de Aliases de Patentes

A diferencia de los reportes financieros presentados ante la SEC (asociados rígidamente al CIK del emisor), las patentes suelen ser registradas bajo subsidiarias operativas, entidades tenedoras de propiedad intelectual (e.g., *Microsoft Technology Licensing LLC*, *Amazon Tech Inc.*, *Google LLC*, *GM Global Technology Operations LLC*) o denominaciones comerciales abreviadas por los examinadores de patentes.

Para garantizar cobertura completa sin perder patentes relevantes:
1. **Universo de origen**: Las 547 firmas de `configs/us/universe.csv` (tras remover 3 tickers duplicados que compartían CIK con otro ticker ya en el universo: UA/UAA, DISCA/DISCK, DISH/ECHO).
2. **Generación de Stems y Expansión de Sufijos Legales**: Normalización de nombres corporativos, eliminación de prefijos y sufijos de registro (`INC`, `CORP`, `CO`, `LLC`, `LTD`, `PLC`, `SA`, `AG`, `NV`, etc.) y estandarización a mayúsculas limpias.
3. **Mapeo Curado de Subsidiarias de Patentes**: Incorporación explícita de las principales entidades titulares de patentes para conglomerados tecnológicos, farmacéuticos e industriales (e.g., *Waymo*, *DeepMind*, *Nuance*, *LinkedIn*, *Zoox*, *Instagram*, *Oculus*, *Mellanox*, *Mobileye*, *VMware*, *Janssen*, *Ethicon*, *Celgene*), más una segunda ronda de nombres históricos y abreviaciones descubiertos y verificados directamente contra los assignees reales de Google Patents (e.g., *Tyco Electronics* para TE Connectivity, *ModernaTX* para Moderna, *Otis Elevator Co* para Otis, *Gen Electric* para GE, *Lam Res Corp* para Lam Research).
4. **Estandarización de Abreviaciones de Google Patents**: Reglas de armonización para abreviaturas habituales en `assignee_harmonized.name` (`TECHNOLOGIES` $\rightarrow$ `TECH`, `INTERNATIONAL` $\rightarrow$ `INT`, etc.).
5. **Resolución de colisiones cruzadas entre tickers**: dos empresas del universo pueden reclamar el mismo assignee tras un spin-off o M&A. Cerner Innovation Inc (adquirida por Oracle en 2022) queda solo en CERN, no en ORCL, siguiendo la convención de mantener firmas delisted con su propia ventana histórica. Symantec Corp (repartida entre Broadcom y NortonLifeLock/Gen Digital en el carve-up de 2019) queda solo en AVGO, por ser el segmento enterprise -- el más intensivo en patentes de los dos -- el que adquirió Broadcom.
6. **Matching por regex, no por igualdad exacta**: el join contra `patents-public-data.patents.publications` usa `REGEXP_CONTAINS(UPPER(assignee_harmonized.name), assignee_regex)`, no `=`. Una lista de alias exactos nunca cubre todas las variantes reales (subsidiarias regionales, GmbH, abreviaciones internas de Google Patents); el regex por empresa sí, con una excepción: el ticker SO (Southern Co) excluye su stem libre ("SOUTHERN") del regex porque es una palabra común que colisiona con universidades y entidades chinas no relacionadas -- se valida caso por caso antes de ampliar el regex a una palabra genérica.

### Artefactos Generados en `data/raw/`:
* `data/raw/reference/sp500_patent_aliases_seed.csv` y `.parquet`: Tabla ancha (547 filas) con CIK, ticker, nombre oficial, stem limpio, lista de aliases delimitada por `;` y expresión regular compilada.
* `data/raw/sp500_patent_aliases_seed.csv`: Copia directa en `data/raw/` como semilla raw.
* `data/raw/sp500_patent_assignees_long.parquet` y `.csv`: Tabla larga (4.304 filas) con el mapeo `(ticker, cik, company_name, assignee_alias, is_known_subsidiary, active_status)`.

---

## 2. Metodología de Clasificación de Patentes de IA (OECD 2025)

Siguiendo el reporte oficial de la OECD:
> *“Identifying emerging AI technologies using patent data: A semi-automated approach”*, OECD Science, Technology and Industry Working Papers (Septiembre 2025).

La identificación opera bajo una **regla de doble canal (dual-channel approach)**:

$$\text{AI Patent} = \text{CPC} \in \mathcal{C}_{\text{Core}} \;\lor\; \Big(\text{CPC} \in \mathcal{C}_{\text{Related}} \;\land\; \text{Texto} \text{ contiene } \ge 1 \text{ keyword OECD}\Big)$$

### A. 5 Grupos CPC "Core AI" (Incondicionales)
Cualquier patente clasificada en estos grupos cuenta directamente como IA, independientemente de su título o resumen:
1. `G06N3`: Biological models, neural networks.
2. `G06N5`: Knowledge-based models.
3. `G06N7`: Specific mathematical models.
4. `G06N20`: Machine learning.
5. `G06F18`: Pattern recognition.

### B. 95 Grupos CPC "AI-Related" (Condicionales a Keyword)
Grupos de aplicación que sólo califican si además contienen terminología explícita de IA:
* **Vehículos autónomos**: `B60W30`, `B60W40`, `B60W50`, `B60W60`, `B60W2040`, `B60W2050`, `B60W2400`, `B60W2420`, `B60W2422`, `B60W2520`, `B60W2540`, `B60W2552`, `B60W2554`, `B60W2555`, `B60W2556`, `B60W2720`, `B60W2754`, `B60W2756`
* **Vehículos no tripulados / Drones**: `B64U10`, `B64U20`, `B64U30`, `B64U40`, `B64U50`, `B64U60`, `B64U70`, `B64U80`, `B64U2101`, `B64U2201`
* **Procesamiento de datos / IR / NLP**: `G06F5`, `G06F7`, `G06F15`, `G06F16`, `G06F17`, `G06F30`, `G06F40`, `G06F2111`, `G06F2113`, `G06F2119`, `G06F2207`, `G06F2209`, `G06F2216`, `G06F2218`
* **Computación analógica / híbrida**: `G06J1`
* **Otros sistemas de IA**: `G06N99`
* **Métodos de negocio y finanzas**: `G06Q10`, `G06Q30`, `G06Q40`, `G06Q50`, `G06Q99`
* **Procesamiento y generación de imágenes**: `G06T1`, `G06T3`, `G06T5`, `G06T7`, `G06T9`, `G06T11`, `G06T13`, `G06T15`, `G06T17`, `G06T19`, `G06T2200`, `G06T2201`, `G06T2207`, `G06T2210`, `G06T2211`, `G06T2215`, `G06T2219`
* **Visión por computador y reconocimiento**: `G06V10`, `G06V20`, `G06V30`, `G06V40`, `G06V2201`
* **Bioinformática**: `G16B5`, `G16B10`, `G16B15`, `G16B20`, `G16B25`, `G16B30`, `G16B35`, `G16B40`, `G16B45`, `G16B50`
* **Quimioinformática**: `G16C10`, `G16C20`, `G16C60`
* **Informática médica y diagnóstico**: `G16H10`, `G16H15`, `G16H20`, `G16H30`, `G16H40`, `G16H50`, `G16H70`, `G16H80`
* **Internet de las Cosas (IoT)**: `G16Y20`, `G16Y40`
* **Otras aplicaciones TIC**: `G16Z99`

### C. 274 Keywords de IA (Tabla A B.1 OECD 2025)
Extraídas textualmente de la especificación técnica de la OECD, cubriendo desde modelos clásicos (*random forest*, *support vector machine*, *hidden markov*) hasta técnicas de vanguardia (*large language model*, *diffusion model*, *generative adversarial network*, *attention-based transformer*, *federated learning*, *neural radiance field*, *agentic AI*).

---

## 3. Infraestructura en BigQuery y Ejecución

* **Proyecto GCP**: `skillforge-dev-502010`
* **Dataset**: `thesis_patents`
* **Tablas Creadas**:
  1. `skillforge-dev-502010.thesis_patents.sp500_patent_aliases_seed` (547 filas)
  2. `skillforge-dev-502010.thesis_patents.sp500_patent_assignees_long` (4.304 filas)
  3. `skillforge-dev-502010.thesis_patents.sp500_firm_year_ai_patents_oecd2025` (3.886 filas año-firma; 424 de las 547 firmas tienen al menos una patente matcheada)
* **Scripts de Pipeline**:
  * `scripts/gold/external_patents/build_sp500_patent_aliases.py`: Extracción y expansión de nombres, más el diccionario curado de subsidiarias/nombres históricos.
  * `scripts/gold/external_patents/upload_aliases_to_bigquery.py`: Carga hacia BigQuery.
  * `scripts/gold/external_patents/build_oecd_patents_panel.py`: Generador de SQL con las 100 clases CPC y 274 keywords.
  * `scripts/gold/external_patents/run_oecd_ai_patents.py`: Ejecutor integral que procesa los ~170M de registros de patentes (filas patente × assignee) y persiste el panel procesado.

### Optimización: deduplicar assignees antes del regex

El join contra `patents-public-data.patents.publications` evalúa un regex de 547 empresas por fila -- caro si se corre sobre las ~170M filas patente×assignee del rango 2015-2026. Pero `assignee_harmonized.name` se repite miles de veces por empresa (una patente comparte el mismo nombre de assignee que todas las demás de esa firma), así que el universo real de nombres *distintos* en ese rango es de solo **5.187.016** -- una reducción de ~33x. La query ahora corre el regex una sola vez por nombre distinto (`DISTINCT assignee_harmonized.name`), y recién después hace un join barato por igualdad exacta (`hash join`) de vuelta contra la tabla completa para traer título/resumen/CPC de cada patente candidata. Resultado medido: **157 segundos** contra **1.438 segundos** (24 min) de la versión sin optimizar -- 9,1x más rápido, mismo costo en bytes escaneados (~244 GB, ≈ USD 1,50 a precio on-demand).

---

## 4. Estadísticas del Panel Resultante (2015–2026)

### Evolución Agregada por Año de Solicitud (Filing Year)

| Año | Patentes Totales | Patentes IA (OECD 2025) | Patentes Core AI | Patentes Related + KW | Intensidad IA (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **2015** | 302.289 | 7.148 | 5.033 | 3.286 | 2,36% |
| **2016** | 301.670 | 10.368 | 7.788 | 4.825 | 3,44% |
| **2017** | 302.357 | 14.553 | 11.371 | 7.139 | 4,81% |
| **2018** | 289.632 | 18.793 | 16.020 | 8.490 | 6,49% |
| **2019** | 293.321 | 23.458 | 20.237 | 10.323 | 8,00% |
| **2020** | 274.084 | 25.517 | 22.603 | 10.858 | 9,31% |
| **2021** | 265.983 | 25.597 | 22.546 | 10.499 | 9,62% |
| **2022** | 243.648 | 21.463 | 17.274 | 9.796 | 8,81% |
| **2023** | 205.333 | 18.054 | 13.337 | 9.078 | 8,79% |
| **2024** | 140.574 | 13.070 | 9.056 | 7.039 | 9,30% |
| **2025** | 57.039 | 5.093 | 3.487 | 2.942 | 8,93% |
| **2026** | 283 | 20 | 18 | 10 | 7,07% |

*(Los totales por año subieron respecto a la primera corrida -- p.ej. 2015 pasó de 209.614 a 302.289 patentes totales -- porque el match exacto original perdía subsidiarias y nombres históricos de muchas de las 547 firmas; el fix de alias captura ese volumen real, no infla intensidad de IA de forma artificial: la intensidad anual se mueve apenas unas décimas.)*

*(Nota: La reducción observada en 2024-2026 en el total de patentes refleja el rezago estándar de publicación de 18 meses de las oficinas de patentes internacionales).*

### Top 15 Firmas del S&P 500 en Patentes de IA (2020–2025)

| Ticker | Empresa | Patentes Totales | Patentes IA (OECD) | Core AI | Related + KW | Intensidad IA (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **GOOGL** | Alphabet Inc. | 40.036 | 12.132 | 9.934 | 5.513 | 30,30% |
| **IBM** | International Business Machines Corp. | 40.621 | 11.277 | 9.955 | 4.286 | 27,76% |
| **MSFT** | Microsoft Corp. | 33.359 | 8.794 | 7.278 | 4.224 | 26,36% |
| **QCOM** | Qualcomm Inc. | 101.165 | 5.758 | 4.814 | 1.911 | 5,69% |
| **INTC** | Intel Corp. | 37.759 | 4.769 | 4.269 | 1.707 | 12,63% |
| **NVDA** | NVIDIA Corp. | 11.827 | 4.367 | 3.735 | 1.987 | 36,92% |
| **COF** | Capital One Financial Corp. | 12.781 | 3.681 | 3.163 | 1.865 | 28,80% |
| **TM** | Toyota Motor Corp. | 78.528 | 3.314 | 2.033 | 1.754 | 4,22% |
| **ADBE** | Adobe Inc. | 4.999 | 3.094 | 2.239 | 2.142 | 61,89% |
| **AMZN** | Amazon.com Inc. | 10.950 | 2.568 | 1.741 | 1.481 | 23,45% |
| **ORCL** | Oracle Corp. | 8.663 | 2.167 | 1.768 | 1.168 | 25,01% |
| **BAC** | Bank of America Corp. | 6.663 | 1.958 | 1.576 | 884 | 29,39% |
| **AAPL** | Apple Inc. | 52.549 | 1.802 | 1.359 | 761 | 3,43% |
| **F** | Ford Motor Co. | 23.860 | 1.604 | 1.231 | 787 | 6,72% |
| **MU** | Micron Technology Inc. | 28.186 | 1.564 | 1.460 | 538 | 5,55% |

*(Toyota Motor entra al top 15 con el fix -- antes su alias no capturaba variantes como "Toyota Motor North America Inc" o "GAC Toyota Motor Co Ltd". Oracle bajó de 2.289 a 2.167 patentes de IA porque las patentes de Cerner Innovation Inc, adquirida en 2022, se le quitaron y quedaron solo en CERN. Intuit sale del top 15 por el mismo motivo que Toyota entra: el ranking se corrió, no porque Intuit haya perdido patentes.)*
