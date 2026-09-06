# Los builders del panel financiero, y qué números se movieron al recalcular

Hasta esta sesión, cinco parquets del lado contable/mercado existían como
**datos sin código que los generara**: el script original (`build_financials_v2.py`
y los pedazos que produjeron beta, CAR y ROIC/WACC) nunca se versionó y se
perdió, igual que había pasado antes con los clusters. Es decir: los
documentos `02`, `03`, `04`, `05`, `07` y `08` reportaban cifras que **nadie
podía volver a producir**.

Este documento registra los builders que ahora existen, cómo se verificó que
reproducen lo anterior, y qué cifras cambian.

## Los scripts

| Script | Produce | Lee |
|---|---|---|
| `scripts/analytics/build_firm_clusters.py` | arquetipos de voz, clusters de comportamiento, panel empresa-año | `gold_ai_frames` |
| **`scripts/analytics/build_firm_financials.py`** | `firm_year_financials`, `firm_year_financials_ratios` | `data/raw/xbrl_facts/us/`, `filing_manifest` |
| **`scripts/analytics/build_market_factors.py`** | `firm_year_filing_returns`, `firm_year_market_factors` | precios, factores FF3, ratios |
| **`scripts/analytics/build_roic_wacc.py`** | `firm_year_roic_wacc` | ratios, factores, mercado |
| `scripts/analytics/build_firm_panels.py` | los cuatro parquets con etiquetas que leen 02-08 | todo lo anterior |
| **`scripts/analytics/report_crosscheck_stats.py`** | las tablas numéricas de 02/04/05/08 | los parquets |
| `scripts/analytics/washing_score.py` | `firm_washing_score` | `gold_ai_frames` |

En negrita los cuatro nuevos. Todo corre con **`make analytics`**, en el orden
de dependencia correcto. Ninguno llama a un LLM ni cuesta un peso de API:
leen `gold_ai_frames` (que sí costó llamadas) sin tocarlo.

El insumo `data/raw/xbrl_facts/us/` (515 tickers, 113 MB) no estaba en el
disco local: se baja con
`scripts/common/sync_data_b2.sh pull --path raw/xbrl_facts`.

## Verificación: ¿el código nuevo reproduce lo que había?

Contra las copias archivadas en `data/archive/processed/clusters/`
(recuperadas del bucket B2 y guardadas con su `POINTER.json`):

| Tabla | Resultado |
|---|---|
| `firm_year_filing_returns` | **idéntico**: 2.746 filas, correlación 1,000, diferencia < 1e-6 en el 100% |
| `firm_year_financials` | 88-100% de las celdas idénticas donde ambas versiones tienen dato |
| `firm_year_financials_ratios` | correlaciones 0,98-1,00 (`operating_margin` 0,999, `shares_out` 1,000) |
| `firm_year_market_factors` | beta 0,993, market cap 1,000, P/B 1,000, CAR 0,892 |
| `firm_year_roic_wacc` | corriendo con `--erp 0.082` (el valor viejo): ROIC 0,992, spread 0,993; medianas ROIC 11,4%→11,7%, spread 2,80%→2,93% |

O sea: donde la metodología es la misma, el código nuevo da lo mismo. Lo que
cambia son las tres correcciones deliberadas de abajo.

## Qué se corrigió (rige lo nuevo)

1. **Cadenas de fallback de conceptos XBRL.** Un filer que reporta
   `RevenueFromContractWithCustomerExcludingAssessedTax` en vez de `Revenues`
   ya no queda sin revenue. Cobertura: capex **69% → 87%**, SG&A **56% → 80%**,
   `net_margin` **87% → 98%**, EV/EBITDA **48% → 68%**, revenue 96% → 98%.
2. **`next_*_yoy` se anula cuando el gap fiscal sale de [340, 380] días.** Son
   huecos de XBRL que ponen el "FY siguiente" a 2-4 años (11 filas acá), lo
   que `leakage-checking.md` pedía filtrar y nadie filtraba.
3. **Denominadores <= 0 producen NULL, no un ratio absurdo.** Equity negativo
   ya no genera ROE positivo espurio; EPS negativo ya no genera P/E negativo
   (por eso la cobertura de P/E baja de 98% a 89%: esas filas nunca debieron
   contar).
4. **ERP geométrico (6,48%) en vez de aritmético (8,20%)** en el WACC. El 8,2%
   es la media aritmética del `mktrf` diario 2000-2026 anualizada ×252; la
   geométrica del mismo período es 6,48% y es la que corresponde para
   descontar. **No es neutral entre segmentos**: como `coe = rf + β·ERP`, un
   ERP inflado infla las diferencias de WACC en proporción a las diferencias
   de beta, y los segmentos de `08_...md` difieren justamente en beta (0,46
   vs 1,00). `--erp 0.082` reproduce la versión vieja.

## Números que se movieron

Todo lo de abajo sale de `report_crosscheck_stats.py` (salida completa en
`data/processed/clusters/crosscheck_stats.json`).

### Talk vs. walk (`02_...md` §1, `04_...md`, `05_...md` §1-§2)

| especificación | antes | ahora | n |
|---|---|---|---|
| cruda | 0,091 / 0,071 | **0,085** | 889 |
| dentro de sector-año | 0,033 | **0,052** | 889 |
| dentro de empresa (efectos fijos) | 0,064 | **0,081** | 889 |
| primeras diferencias | 0,078 | **0,077** | 521 |
| permutación, cruda | p=0,035 | **p=0,014** | |
| permutación, dentro de sector-año | p=0,321 | **p=0,129** | |

El resultado central de `05_...md` **se refuerza**: la señal within-firm
(0,081) sigue siendo mayor que la within-sector (0,052), que es lo contrario
de lo que predeciría "todo es composición sectorial". Sigue siendo una
correlación débil.

### FDR sobre las 11 correlaciones (`05_...md` §3)

**Ninguna de las 11 sobrevive ahora al FDR de 5%** (antes: entre 1 y 2, ambas
al borde). El cambio decisivo es `ai_infrastructure ~ next_capex_yoy`, que
`05_...md` llamaba "la señal más robusta del cruce contable": pasa de
**r=0,114 / p=0,0045** a **r=0,080 / p=0,023**, con el umbral BH en 0,0091.
Se debilitó justamente al mejorar la cobertura de capex de 69% a 87% — la
correlación anterior se apoyaba en el subconjunto de filers que reportan el
tag más común.

`revenue_outcome ~ next_revenue_yoy` queda primera en el ranking (p=0,011)
pero tampoco pasa (umbral 0,0045).

**Lectura para la tesis: de once cruces contables/mercado, ninguno se
distingue del azar una vez corregido por comparaciones múltiples.** Es un
resultado más limpio de defender que el anterior, no peor: el instrumento de
texto no predice resultados financieros, y ahora eso está medido con
cobertura decente en vez de con la mitad de la muestra.

### ROIC − WACC (`08_...md`)

Medianas del panel: WACC **9,1% → 8,2%**, spread **+2,8% → +4,0%** (efecto ERP).

| arquetipo de voz | ROIC | WACC | spread | antes (spread) |
|---|---|---|---|---|
| A cauteloso | 12,3% | 8,0% | +3,66% | +1,69% |
| B genérico | 13,5% | 8,0% | +4,51% | +1,36% |
| C cuantificador | 13,3% | 10,0% | +4,32% | **−0,28%** |
| D vocal | 13,6% | 8,8% | **+5,67%** | +4,51% |

**El hallazgo de `08_...md` de que "C es el único con spread negativo" era un
artefacto del ERP inflado.** C tiene el beta más alto de los cuatro, así que
era el más castigado por sobreestimar la prima de riesgo. Con el ERP correcto
los cuatro arquetipos crean valor y el orden D > B > C > A es mucho más plano
que la historia anterior. D sigue arriba.

Por cluster de comportamiento el orden se mantiene (el de despliegue de
producto, 2, tiene el ROIC más alto: 14,2%), pero el "mínimo" (1) ya no queda
último — lo hace el 0, con n=55.

## Re-corrida 2026-09-06 (tarde): prefiltro v2 (umbral 0,17) y earnings calls completas

Después de lo anterior se desplegó el prefiltro v2 (árboles —
`prefilter_evaluation.md` §8.16) y se terminó de clasificar el corpus de
earnings calls. El despliegue de las 13:48 UTC usó umbral 0,30 ("validación
multicanal") y dejó el panel en 24.744 frames y 1.396 filas — menos que la
mañana, porque `gold_ai_frames` toma la población del ÚLTIMO despliegue y el
0,30 recorta DEF 14A (recall 0,93 contra 0,96 en el holdout de proxy/8-K, y
0,84 contra 0,97 en 8-K). Se re-desplegó con **0,17**, el umbral de recall
alto de la mañana (`run=20260906T160624Z`, 30.280 textos marcados), se
clasificaron los 750 textos que ese umbral agregaba, y todo lo de abajo es
sobre esa población. `make analytics` se volvió a correr entero, más los scripts
que el Makefile no encadena (`report_crosscheck_stats.py --json`,
`validate_washing_score.py`, `voice_behavior_factors.py`,
`behavior_block_eval.py`, `cluster_diagnostics.py`,
`washing_hierarchical.py`), y **las tablas de `02` a `09`, `11`, `12` y `13`
se reescribieron con estas cifras.** `01_...md` conserva las de la mañana
(consultas ad hoc sin script) con una nota al inicio.

Dos cambios técnicos hicieron falta para que corriera en el laptop: la vista
`filing_manifest` castea ahora `filing_date` a DATE (el manifiesto de calls lo
guardaba como texto y `build_firm_clusters.py` fallaba en
`extract(year from ...)`), y `statsmodels` + `matplotlib` —que `shock_*.py`
importan y `pyproject.toml` no declara— se instalaron a mano en el venv.

Qué se movió respecto de la corrida de la mañana (ver el detalle en cada doc):

| cifra | mañana (v1, umbral 0,18) | tarde (v2, umbral 0,17) |
|---|---:|---:|
| frames en el panel (10-K / DEF 14A / 8-K) | 25.443 (17.731 / 7.400 / 312) | **25.355** (17.758 / 7.284 / 313) |
| panel empresa-año / empresas | 1.423 / 457 | **1.426 / 460** |
| talk vs. walk, r cruda (n) | 0,085 (889) | **0,049 (939)** |
| ídem, dentro de sector-año | 0,052 | **0,019** |
| ídem, efectos fijos de empresa | 0,081 | **0,084** |
| permutación, cruda | p=0,014 | **p=0,12** |
| correlaciones que pasan FDR 5% (de 11) | 0 | **0** |
| `ai_infrastructure` ~ `capex` | r=0,080, p=0,023 | **r=0,086, p=0,013** |
| ROIC − WACC mediano del panel | +4,0% | **+4,9%** |
| spread por arquetipo A / B / C / D | +3,7 / +4,5 / +4,3 / +5,7 | **+3,8 / +5,0 / +4,9 / +5,6** |
| cluster de comportamiento mínimo, spread | último | **tercero de cuatro (+3,71%)** |
| grupo washing (D + mínimo) | 20 empresas | **12 empresas** |
| callados B | 39 empresas | **40 empresas** |
| score de washing: cola / callada | 7 / 2 (439 empresas) | **8 / 3 (449 empresas)**, entra CRM |
| segmentos (k=3) | 101 / 195 / 124 | **101 / 189 / 129** |
| grilla, esquinas W / SC / VS / S | 27 / 29 / 72 / 66 | **26 / 30 / 74 / 66** |
| cohorte 2021 | 85 empresas, 465 filas | **99 empresas, 537 filas** |
| DiD simple SEC, `promotional_rate` | −9,46 p.p. (pretrend p=0,100) | **−9,67 p.p. (pretrend p=0,120)** |
| earnings calls clasificadas | 31 empresas, 1.928 frames | **403 empresas, 16.267 frames** |

Lectura: con el mismo umbral que la mañana, el panel es prácticamente el
mismo (−0,3% de frames) y el lado de texto casi no se mueve (segmentos,
grilla y cola de washing idénticos salvo CRM). Lo que sí se mueve es el
cruce contable, y no por la población sino por la mejor cobertura XBRL de
los builders: la correlación cruda de revenue dejó de distinguirse de ruido
(0,085 → 0,049) y ya no hay ningún par del cruce que reportar como señal.
El grupo "washing = D × mínimo" cambió de 20 a 12 empresas sin que cambiara
casi nada más, lo que confirma que esa definición no es estable; la del
score (`09_...md`) sí lo es.

## Qué queda pendiente

- `07_...md` documenta `groupby('ticker').median()` pero `build_firm_panels.py`
  agrega con `.mean()`. Hay que decidir cuál es la correcta y dejar una sola.
- El panel empresa-año sigue condicionado a ≥3 frames en el año, así que
  entrar y salir del panel es endógeno al propio fenómeno que se mide.
- `01_...md` conserva sus tablas SQL de la corrida de la mañana; regenerarlas
  requiere convertir esas consultas en un script, como se hizo con
  `report_crosscheck_stats.py` para `02`/`04`/`05`/`08`.
- Declarar `statsmodels` y `matplotlib` en `pyproject.toml` y destrabar el
  conflicto `sentence-transformers` / extra `pdf-vlm-mineru` que hoy impide
  `uv sync` (se corre con `uv run --frozen --no-sync`).
