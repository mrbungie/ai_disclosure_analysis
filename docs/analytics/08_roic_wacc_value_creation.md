# ROIC − WACC: ¿los segmentos de "sustancia" realmente crean valor económico?

> **Recalculado con builders versionados (ver `10_builders_y_recalculo.md`).**
> El lado contable/mercado ya no viene de un script perdido: lo producen
> `build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
> (`make analytics`). Con mejor cobertura XBRL y el ERP corregido, varias
> cifras de este documento se movieron — las tablas de abajo son las de la
> corrida anterior; los deltas están listados en `10_...md`.

> **Recalculado 2026-09-05 con DEF 14A y 8-K.** Los insumos financieros
> (ROIC, WACC) no cambiaron; las etiquetas de segmento sí. En el camino se
> corrigió un bug de reproducibilidad en `gold_ai_frames` (ver `01_...md`).
> **El resultado sobre washing se dio vuelta por completo** — leer esa
> sección.


Extiende `07_segment_financial_profiles.md` de ratios contables y
múltiplos de mercado a una medida de **creación de valor económico**:
ROIC (retorno sobre capital invertido) vs. WACC (costo de capital) —
si ROIC > WACC, la empresa genera retornos por sobre lo que le cuesta
financiarse; si ROIC < WACC, destruye valor aunque sea rentable en
términos contables simples.

## Construcción (todas las fórmulas son aproximaciones — ver limitaciones)

```python
NOPAT = operating_income * (1 - effective_tax_rate)
effective_tax_rate = tax_expense / pretax_income  # por firm-year, XBRL; fallback 21% (tasa estatutaria US)
invested_capital = long_term_debt + equity          # capital empleado, valor libro
ROIC = NOPAT / invested_capital

cost_of_equity = rf_anualizado_al_filing + beta * ERP   # ERP = 8,2% (media histórica anualizada mktrf, Fama-French 2000-2026)
cost_of_debt = interest_expense / long_term_debt         # fallback: rf + 2% de spread si no hay dato
WACC = (E/(E+D))*cost_of_equity + (D/(E+D))*cost_of_debt*(1-effective_tax_rate)
# E = market_cap (de firm_year_market_factors.parquet), D = long_term_debt
```

Cobertura sobre 2.921 firma-años: ROIC 2.157 (74%), WACC 2.488 (85%),
`roic_minus_wacc` (ambos no-nulos) 1.814 (62%). Medianas del panel
completo: ROIC 11,4%, WACC 9,1%, spread +2,8% — valores creíbles para
un panel de empresas grandes/medianas ya rentables.

Guardado en `data/processed/clusters/firm_year_roic_wacc.parquet`.

## Resultados: por cluster de comportamiento

Mediana sobre empresas-año con ROIC y WACC disponibles (1.589
observaciones, 281 empresas):

| Cluster de comportamiento | ROIC | WACC | **ROIC − WACC** | % obs con spread > 0 | n obs |
|---|---|---|---|---|---|
| 0 — Intermedio interno | 13,4% | **7,7%** | **+4,70%** | 68,6% | 86 |
| 1 — Comportamiento mínimo | 10,7% | 8,7% | +1,32% | 58,0% | 697 |
| 2 — Desplegadores de producto | **13,5%** | **10,7%** | +3,36% | 59,5% | 538 |
| 3 — Inversores en infraestructura | 11,7% | 9,0% | +2,65% | 63,1% | 268 |

**El cluster de despliegue real (2) tiene el ROIC más alto (13,5%) pero no
el mayor spread**, porque también tiene el WACC más alto (10,7%). El
cluster 0 lo supera (+4,70%) por la vía opuesta: ROIC similar con el WACC
más bajo del conjunto. Con n=25 empresas, ese primer puesto no es
confiable; lo que sí es sólido es que **el cluster mínimo queda último
(+1,32%) con clara diferencia** — no describir comportamiento de IA va de
la mano de crear menos valor económico.

La brecha se achicó respecto de la versión anterior (era +3,6% contra
+0,5/0,6%, o sea 6-7x; ahora 3,6x entre el 2 y el 1).

## Resultados: por arquetipo de VOZ

| Arquetipo de voz | ROIC | WACC | **ROIC − WACC** | % obs con spread > 0 | n obs |
|---|---|---|---|---|---|
| A cauteloso | 11,7% | 9,1% | +1,69% | 57,6% | 373 |
| B genérico | 11,0% | 9,1% | +1,36% | 57,4% | 619 |
| C cuantificador | 10,4% | **11,8%** | **−0,28%** | 48,4% | 124 |
| D vocal | **13,4%** | 9,4% | **+4,51%** | **68,1%** | 473 |

**Esto invierte el resultado de la versión anterior, y es el cambio más
grande del documento.** Antes D quedaba por debajo de B y C porque su
WACC alto se comía su rentabilidad. Hoy D tiene el ROIC más alto (13,4%),
un WACC en el promedio (9,4%) y el mayor spread por lejos (+4,51%), con
el 68,1% de sus observaciones creando valor.

Y C, el arquetipo que cuantifica, es **el único con spread negativo**
(−0,28%): ROIC más bajo del conjunto (10,4%) y WACC más alto (11,8%). Su
composición nueva lo explica —semiconductores y datos, capital-intensivos
y de beta alto (`04_...md`)— pero el resultado es incómodo para la
lectura simple de que cuantificar señala sustancia: **el grupo que pone
números crea menos valor económico que el que promociona.**

Con n=124 observaciones y 22 empresas, C es el grupo más chico de los
cuatro y su cifra es la menos estable. Aun así, el contraste D vs. C está
medido sobre 597 observaciones combinadas y no es un artefacto de muestra
mínima.

## Resultados: washing vs. resto de D — el resultado se dio vuelta

> **La definición del grupo está superada** por `09_washing_score.md` (versión
> validada: 8 empresas, no 20). Las comparaciones ROIC-WACC de esta sección
> describen un grupo que ya no existe.


| | Washing (D + comportamiento mínimo) | Resto de D |
|---|---|---|
| ROIC | 13,4% | 13,5% |
| WACC | **6,6%** | 10,2% |
| **ROIC − WACC** | **+7,31%** | +4,17% |
| % obs con spread > 0 | **83,3%** | 65,6% |
| n obs (empresas) | 66 (11) | 407 (72) |

**El grupo de "candidatos a washing" crea MÁS valor económico que el resto
de D**, con el 83% de sus observaciones en spread positivo. La versión
anterior concluía exactamente lo contrario ("al menos la mitad está
destruyendo valor económico activamente"), sobre 4 empresas con dato.

**Esto NO significa que hablar vago genere valor.** El ROIC —lo que la
empresa produce con su capital— es idéntico entre los dos grupos: 13,4%
vs. 13,5%, una diferencia de 0,1 p.p. En desempeño operativo no hay
ninguna diferencia. Todo el spread extra viene del otro término: el WACC,
6,6% contra 10,2%.

Y ese WACC bajo es consecuencia del **beta: 0,46 contra 1,00**
(`07_...md`). Son utilities, consumo básico, farma, seguros y bancos —
negocios estables que se financian barato. `ROIC − WACC` premia por
construcción al negocio de bajo riesgo: una utility regulada que gana 13%
sobre capital que le cuesta 6% tiene un spread excelente que no tiene
nada que ver con IA.

La causalidad no es "es vago → crea valor". Hay una **causa común**: ser
una empresa estable, no-tech y de bajo I+D produce simultáneamente (a) un
costo de capital bajo, que infla el spread, y (b) un discurso sobre IA sin
comportamiento observable detrás, porque efectivamente no hay despliegue
que describir. El estilo de divulgación y el spread son dos consecuencias
del mismo hecho, no causa y efecto entre sí.

**Un control sectorial no lo elimina, lo que refuerza esa lectura.**
Comparando cada empresa-año contra la mediana de su propio SIC-2, el
grupo de washing queda +2,96 p.p. y el resto de D +0,09 p.p. — o sea que
no es sólo "están en otros sectores": es que **dentro de cualquier
sector, la empresa de beta bajo tiene el spread alto**. El mecanismo es
el riesgo, no la industria, y por eso el control sectorial no lo toca.

Conclusión: la comparación de `ROIC − WACC` entre estos dos grupos **no
es informativa sobre AI-washing en ninguna dirección.** La versión
anterior de este documento concluía lo contrario ("al menos la mitad está
destruyendo valor económico") sobre 4 empresas con dato; esta versión
mediría riesgo si se leyera literalmente. Para preguntar si el washing
tiene consecuencias económicas hay que usar una métrica que no esté
mecánicamente ligada al beta — ROIC solo, o crecimiento de ingresos, o
retorno ajustado por riesgo como el CAR de `04_...md`.

## Resultados: sustancia callada — se sostiene, debilitada

| | Callados B | Resto de B |
|---|---|---|
| ROIC | **13,5%** | 10,6% |
| WACC | 10,2% | 8,9% |
| **ROIC − WACC** | **+2,36%** | +1,21% |
| % obs con spread > 0 | 56,0% | 57,7% |
| n obs (empresas) | 141 (25) | 478 (86) |

Los "callados B" crean casi el doble de spread que el resto de su propio
arquetipo de voz (+2,36% vs. +1,21%), con un ROIC 2,9 p.p. mayor. La
versión anterior reportaba 3x (+6,5% vs. +2,1%); hoy es 1,9x. **La
dirección y el orden de magnitud sobreviven, la magnitud se reduce.**

A diferencia del grupo de washing, acá la ventaja viene del numerador: el
ROIC es genuinamente más alto, no el WACC más bajo. Es el hallazgo mejor
sostenido de este documento — 25 empresas con datos, contraste contra un
control de 86 empresas del mismo arquetipo de voz, y mecanismo coherente
con el perfil de `07_...md` (más I+D, mejor margen bruto, prima de
mercado).

El `% con spread > 0` casi no distingue a los dos grupos (56,0% vs.
57,7%), así que la diferencia está en la magnitud del spread de los que
crean valor, no en cuántos lo crean.

## Lectura conjunta

Qué queda en pie después de recalcular sobre una población distinta:

- **Se sostiene**: los "callados B" —empresas que despliegan IA y lo
  cuentan en registro llano— crean casi el doble de valor económico que
  sus pares de voz equivalente (+2,36% vs. +1,21%), y por la vía correcta
  (ROIC más alto, no WACC más bajo). Es el único resultado de este
  documento con muestra decente y mecanismo coherente entre versiones.
- **Se sostiene**: el cluster de comportamiento mínimo queda último en
  creación de valor (+1,32%). No describir comportamiento de IA
  correlaciona con crear menos valor.
- **Se dio vuelta, pero la comparación no servía en ninguna dirección**:
  "el washing destruye valor económico". Hoy ese grupo muestra más spread,
  con ROIC idéntico (13,4% vs. 13,5%) y WACC 3,6 p.p. menor por su beta de
  0,46. `ROIC − WACC` premia mecánicamente al negocio de bajo riesgo, así
  que entre grupos con betas tan distintos mide riesgo, no conducta de
  disclosure — y el control sectorial no lo arregla, porque el mecanismo
  es el beta, no la industria.
- **Se dio vuelta**: "D no es el que más valor crea". Hoy D tiene el mayor
  spread de los cuatro arquetipos (+4,51%) y C el único negativo
  (−0,28%).

La lección metodológica es más valiosa que cualquiera de los hallazgos, y
vale para todo el proyecto: **estos segmentos son particiones de K-means
sobre una muestra, no categorías del dominio.** Cuando la muestra cambió
—sin que cambiara un solo dato financiero— dos titulares se dieron vuelta
y uno se debilitó a la mitad. Y ninguna comparación financiera entre
segmentos tiene control sectorial, que es lo que explica el caso más
llamativo. Cualquier resultado que dependa de un grupo de menos de ~30
empresas, o que compare segmentos con composición sectorial distinta,
debería tratarse como generador de hipótesis, nunca como evidencia.

## Limitaciones (más pronunciadas que en docs anteriores — leer antes de citar)

- **ROIC y WACC aquí son aproximaciones de valor LIBRO/estimaciones de
  una sola fuente de datos**, no el ROIC/WACC "real" que calcularía un
  analista con acceso a todos los ajustes estándar (leases operativos,
  goodwill, ítems no recurrentes, etc.) — usar como señal direccional,
  no como cifra de valuación.
- **ERP fijo en 8,2% para todas las empresas y años** — no varía por
  año ni por condición de mercado; un ERP dinámico (o distinto por
  década) cambiaría el nivel de WACC para todas las empresas por igual,
  pero no debería cambiar las comparaciones RELATIVAS entre segmentos
  hechas aquí.
- **Costo de deuda con fallback grueso** (rf+2%) cuando no hay
  `InterestExpense` reportado — afecta especialmente a empresas con
  poca deuda donde el dato es más ruidoso.
- **Ninguna comparación entre segmentos tiene control sectorial**, y esa
  es la limitación decisiva de este documento, no el tamaño muestral. El
  grupo de washing tiene beta 0,46 y el resto de D 1,00: cualquier
  diferencia de WACC entre ellos mide sector, no conducta de disclosure.
  Repetir todas estas comparaciones dentro de SIC-2 es el próximo paso
  obligatorio antes de citar cualquier cifra como evidencia.
- **Los grupos de segmento son inestables entre poblaciones.** El de
  washing pasó de 6 a 20 empresas con composición casi totalmente
  distinta entre una versión y otra de este análisis; el de sustancia
  callada A quedó en n=1.
- El ERP fijo no afecta comparaciones relativas, pero el beta sí, y el
  beta es justamente donde los segmentos difieren más.
- No se validó el ROIC/WACC contra una fuente externa (Bloomberg,
  CapitalIQ, etc.) — es una construcción propia desde XBRL crudo, sin
  benchmark de sanity-check más allá de que las medianas del panel
  completo son razonables.
- Mismo alineamiento temporal por fecha real que `02_...md`/`04_...md`
  (`leakage-checking.md`) — hereda esas garantías, no las repite aquí.
