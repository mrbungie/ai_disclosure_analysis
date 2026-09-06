# ROIC − WACC: ¿los segmentos de "sustancia" realmente crean valor económico?

> **Recalculado 2026-09-06 sobre el prefiltro v2 y con builders versionados.**
> Las tablas de este documento salen de la corrida actual de `make analytics`
> (`report_crosscheck_stats.py` reproduce las numéricas): panel de 1.426
> empresas-año y 460 empresas, frames de 10-K, DEF 14A y 8-K, población
> marcada por el prefiltro v2 (árboles, umbral 0,17 — `prefilter_evaluation.md`
> §8.16). Lado contable/mercado producido por `build_firm_financials.py`,
> `build_market_factors.py` y `build_roic_wacc.py` (ERP geométrico 6,48%).
> Las corridas anteriores y sus deltas están en `10_builders_y_recalculo.md`.

> Con el ERP geométrico (6,48% en vez de 8,20%) el WACC mediano del panel
> baja de 9,1% a 7,1% y el spread mediano sube de +2,8% a +4,9%; **los
> cuatro arquetipos y los cuatro clusters crean valor**, y el "C con spread
> negativo" de la corrida anterior era un artefacto del ERP inflado
> (`10_...md`). Las secciones de washing y sustancia callada usan los
> segmentos re-ajustados de `07_...md`.


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

Mediana sobre empresas-año con ROIC y WACC disponibles (1.049
observaciones, 331 empresas):

| Cluster de comportamiento | ROIC | WACC | **ROIC − WACC** | % obs con spread > 0 | n obs |
|---|---|---|---|---|---|
| 0 — Intermedio interno | 11,0% | **6,0%** | +3,22% | **75,6%** | 41 |
| 1 — Comportamiento mínimo | 10,1% | 6,1% | +3,71% | 73,9% | 356 |
| 2 — Desplegadores de producto | **13,5%** | **8,0%** | +5,30% | 68,5% | 476 |
| 3 — Inversores en infraestructura | 13,0% | 6,8% | **+5,62%** | 75,4% | 171 |

**El cluster de despliegue real (2) tiene el ROIC más alto (13,5%) pero
no el mayor spread**, porque también tiene el WACC más alto (8,0%): el
cluster de infraestructura (3) lo supera por 0,3 p.p. (+5,62%) con ROIC
parecido y WACC menor. **El cluster mínimo ya no queda último**: +3,71%,
contra +3,22% del cluster 0 (14 empresas). La corrida anterior lo tenía
en +1,32% y a 3,6x del cluster 2; hoy la brecha es de 1,4x. La lectura
"no describir comportamiento de IA va de la mano de crear menos valor"
se debilitó a "crea menos valor que los clusters de despliegue e
infraestructura, pero no menos que el resto".

## Resultados: por arquetipo de VOZ

| Arquetipo de voz | ROIC | WACC | **ROIC − WACC** | % obs con spread > 0 | n obs |
|---|---|---|---|---|---|
| A cauteloso | 10,2% | 6,6% | +3,81% | 70,5% | 251 |
| B genérico | 12,3% | 6,4% | +5,02% | **75,5%** | 371 |
| C cuantificador | **13,2%** | 7,7% | +4,94% | 69,6% | 112 |
| D vocal | 13,0% | **7,9%** | **+5,62%** | 68,9% | 315 |


**Los cuatro arquetipos crean valor y el orden es plano: D > B > C > A,
con 1,8 p.p. entre el primero y el último.** La corrida anterior tenía a
D "por lejos" arriba (+4,51%) y a C como único negativo (−0,28%); `10_...md`
muestra que ese contraste era en buena parte el ERP aritmético inflando
el WACC del arquetipo de mayor beta. Con el ERP geométrico, C tiene el
ROIC más alto de los cuatro (13,2%) y WACC 7,7%, spread +4,94%; D sigue
arriba pero B está a 0,6 p.p. (con la mayor proporción de observaciones
en positivo, 75,5%).

La lectura incómoda de la corrida anterior —"el grupo que pone números
crea menos valor económico que el que promociona"— ya no tiene respaldo:
C crea menos que D y B, y más que A, por márgenes que con n=112 y 65
empresas no distinguen nada.

## Resultados: washing vs. resto de D — el resultado se dio vuelta

> **La definición del grupo está superada** por `09_washing_score.md` (versión
> validada: 7 empresas, ninguna de las cuales está en este grupo). Las
> comparaciones ROIC-WACC de esta sección describen un grupo de 23 empresas
> (D en el año × cluster mínimo) que se reconstruye distinto en cada
> re-ajuste.


| | Washing (D + comportamiento mínimo) | Resto de D |
|---|---|---|
| ROIC | 13,3% | 13,0% |
| WACC | **5,9%** | 8,0% |
| **ROIC − WACC** | +5,49% | +5,62% |
| % obs con spread > 0 | **76,7%** | 68,1% |
| n obs (empresas) | 30 (23) | 285 (120) |


**El grupo de "candidatos a washing" crea el mismo valor económico que el
resto de D** (+5,49% contra +5,62% de spread mediano, con más
observaciones en positivo: 77% contra 68%). La versión original concluía
que destruía valor ("al menos la mitad está destruyendo valor económico
activamente"), sobre 4 empresas con dato; la corrida anterior lo dio
vuelta (+7,31% vs. +4,17%); ésta lo deja empatado, con 30 observaciones
de 23 empresas.

**Esto NO significa nada sobre hablar vago.** El ROIC —lo que la empresa
produce con su capital— es el mismo en los dos grupos: 13,3% vs. 13,0%.
Lo que cambia es el otro término, el WACC, 5,9% contra 8,0% — y aun así
la mediana del spread no se separa, porque las medianas no se restan.

Y ese WACC bajo es consecuencia del **beta: 0,44 contra 1,02**
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
grupo de washing queda +1,49 p.p. y el resto de D −0,31 p.p. — o sea que
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
| ROIC | **14,0%** | 11,3% |
| WACC | 7,9% | 6,0% |
| **ROIC − WACC** | **+5,58%** | +3,84% |
| % obs con spread > 0 | 72,9% | 77,1% |
| n obs (empresas) | 140 (73) | 231 (132) |


Los "callados B" crean 1,5x el spread del resto de su propio arquetipo
de voz (+5,58% vs. +3,84%), con un ROIC 2,7 p.p. mayor. La versión
original reportaba 3x (+6,5% vs. +2,1%), la anterior 1,9x (+2,36% vs.
+1,21%); hoy es 1,5x. **La dirección sobrevive en las tres corridas; la
magnitud relativa se va reduciendo.**

A diferencia del grupo de washing, acá la ventaja viene del numerador: el
ROIC es genuinamente más alto, y el WACC es incluso 1,9 p.p. MAYOR (beta
0,97 vs. 0,76 en `07_...md`), así que el spread los subestima si acaso.
Es el hallazgo mejor sostenido de este documento — 73 empresas con datos,
contraste contra un control de 132 empresas del mismo arquetipo de voz, y
mecanismo coherente con el perfil de `07_...md` (más I+D, mejor margen
bruto, prima de mercado).

El `% con spread > 0` no favorece a los callados (72,9% vs. 77,1%), así
que la diferencia está en la magnitud del spread de los que crean valor,
no en cuántos lo crean.

## Lectura conjunta

Qué queda en pie después de recalcular sobre una población distinta:

- **Se sostiene**: los "callados B" —empresas que despliegan IA y lo
  cuentan en registro llano— crean 1,6x el valor económico de sus pares
  de voz equivalente (+5,58% vs. +3,84%), y por la vía correcta (ROIC más
  alto, con WACC incluso mayor). Es el único resultado de este documento
  con muestra decente y mecanismo coherente en las tres corridas.
- **Se debilitó**: el cluster de comportamiento mínimo ya no queda último
  en creación de valor (+3,71%, tercero de cuatro); sigue por debajo del
  cluster de despliegue (+5,30%), pero a 1,4x y no a 3,6x.
- **Se dio vuelta y volvió al medio, y la comparación no servía en ninguna
  dirección**: "el washing destruye valor económico". Hoy ese grupo
  muestra el mismo spread que el resto de D (+5,49% vs. +5,62%), con ROIC
  igual y WACC 2,1 p.p. menor por su beta de 0,44. `ROIC − WACC` premia
  mecánicamente al negocio de bajo riesgo, así que entre grupos con betas
  tan distintos mide riesgo, no conducta de disclosure — y el control
  sectorial no lo arregla, porque el mecanismo es el beta, no la
  industria.
- **Se aplanó**: "D es el que más valor crea" sigue siendo cierto por 0,6
  p.p. sobre B, y "C es el único negativo" desapareció con el ERP
  corregido: los cuatro arquetipos están entre +3,8% y +5,6%.

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
