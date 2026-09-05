# ROIC − WACC: ¿los segmentos de "sustancia" realmente crean valor económico?

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

| Cluster de comportamiento | ROIC | WACC | **ROIC − WACC** |
|---|---|---|---|
| 0 — Narradores de revenue | 11,9% | 12,6% | +0,5% |
| 1 — Comportamiento mínimo | 9,6% | 8,6% | +0,6% |
| **2 — Desplegadores de producto** | **12,5%** | 11,0% | **+3,6%** |
| 3 — Inversores en infraestructura | 11,4% | 10,2% | +0,6% |

**El cluster de despliegue real (2) es, con diferencia, el único que
crea valor económico de forma sistemática** (+3,6 p.p., 6-7 veces más
que los otros tres). Los "narradores de revenue" (cluster 0) tienen
ROIC decente (11,9%) pero su WACC es tan alto (12,6%, el más alto de
los 4 — coherente con su beta también más alto en `07_...md`) que casi
no queda spread — son rentables pero apenas cubren su costo de
capital, más riesgosos de lo que su rentabilidad contable sugiere.

## Resultados: por arquetipo de VOZ — la sorpresa

| Arquetipo de voz | ROIC | WACC | **ROIC − WACC** |
|---|---|---|---|
| A cauteloso | 10,6% | 9,3% | +0,5% |
| B genérico | 11,9% | 8,8% | +2,8% |
| **C cuantificador** | 11,8% | 11,4% | **+3,6%** |
| D vocal | 11,7% | **11,3%** | +1,6% |

**D, el arquetipo "líder", NO es el que más valor crea — queda por
debajo de B y C.** Su ROIC es similar al de los demás (11,7%, ni el más
alto), pero su WACC es el segundo más alto (11,3%, casi igual a C) por
su beta elevado (`07_...md`) — el costo de financiarse de una empresa D
es tan alto como su retorno, y el spread neto queda por debajo de la
mitad del de B. Es la misma historia que ya apareció en el CAR ajustado por
mercado de `04_...md` (A superaba a D en retorno ajustado por riesgo):
**ser "vocal" viene con un costo de capital más alto que compensa buena
parte de la rentabilidad aparente.**

## Resultados: washing vs. resto de D — deja de ser sutil

| Ticker | Años en panel (§`06_...md`) | ROIC | WACC | ROIC − WACC |
|---|---|---|---|---|
| CCL | 1 | −1,7% | 12,8% | **−12,7%** |
| FE | 1 | 15,1% | 6,6% | +10,2% |
| GPC | 3 | — (sin dato) | 8,3% | — |
| HII | 4 (migra D→D→B→A) | 7,5% | 7,2% | +0,5% |
| IQV | 2 | 8,0% | 10,2% | **−2,9%** |
| NEM | 2 | — (sin dato) | 8,0% | — |
| **Resto de D (mediana, n=79)** | mayoría 6/6 | 12,0% | 11,4% | **+2,2%** |

> **Nota de persistencia**: como se detalla en `06_...md`, la mayoría
> de estos 6 tiene 1-2 años de datos (frente a 6/6 años típico del
> resto de D) y HII migra fuera de D en los últimos dos años del panel
> — el ROIC−WACC de cada empresa es un dato financiero real e
> independiente del texto, pero conectarlo a "la voz de líder vocal
> predijo esto" es más débil de lo que parece cuando la etiqueta de voz
> detrás descansa en tan pocos años/frames.

De los 4 candidatos a washing con dato disponible, **2 destruyen valor
económico activamente** (CCL: −12,7 p.p., el peor del panel completo;
IQV: −2,9 p.p.) mientras el resto de D crea +2,2 p.p. en mediana. No es
un patrón perfectamente uniforme (FE de hecho crea mucho valor, +10,2
p.p., como utility estable de bajo riesgo) — pero el promedio del grupo
washing está claramente peor que el resto de D, y el peor caso del
panel (CCL) está en este grupo. Esto va más allá de "no son tech" (§
`07_...md`) — sugiere que al menos parte del grupo de washing usa el
lenguaje de IA mientras atraviesa problemas reales de rentabilidad
económica, no solo una diferencia de sector.

## Resultados: sustancia callada — se confirma, y con más fuerza en B que en A

| Segmento | ROIC | WACC | ROIC − WACC |
|---|---|---|---|
| Sustancia callada A | 12,3% | 10,5% | **+1,9%** |
| Resto de A | 10,2% | 9,2% | +0,3% |
| **Sustancia callada B** | 11,7% | 10,3% | **+6,5%** |
| Resto de B | 12,1% | 8,7% | +2,1% |

Ambos grupos de "sustancia callada" crean más valor económico que el
resto de su propio arquetipo de voz — 6x más en el caso de A, 3x más en
el caso de B. **El grupo B de sustancia callada (ADSK, WDAY, TEAM,
OKTA, PLTR, V, WMT, entre otras 33 más) tiene el spread ROIC-WACC más
alto de TODO el análisis (+6,5 p.p.)** — mayor incluso que el cluster
de comportamiento 2 completo (+3,6 p.p.) o que C (+3,6 p.p.). Son las
empresas que, en términos de creación de valor real, más se parecen a
"ganadoras genuinas de IA" — y son precisamente las que menos lo
cuentan con lenguaje promocional.

## Lectura conjunta

ROIC−WACC agrega una tercera confirmación independiente (después de
comportamiento textual en `06_...md` y ratios/multiplos en `07_...md`)
de que **voz y sustancia divergen, y la divergencia importa
económicamente, no solo retóricamente**:

- El washing no es solo "hablar de tech sin ser tech" — al menos la
  mitad de la muestra chica de candidatos está destruyendo valor
  económico activamente.
- La sustancia callada no es solo "más R&D/margen" — se traduce en
  spreads de creación de valor 3-6 veces más altos que sus pares de
  voz similar, con el grupo B como el caso más extremo de todo el
  proyecto.
- El arquetipo "líder vocal" (D) pierde su lugar de primero en creación
  de valor una vez que se descuenta correctamente su propio costo de
  capital (más alto por su beta) — el mismo patrón que ya había
  aparecido con el retorno ajustado por mercado.

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
- **n=4-6 en los grupos de washing** — cualquier lectura es
  descriptiva. El caso CCL (−12,7 p.p.) por sí solo domina la media del
  grupo; sensible a un solo caso extremo.
- No se validó el ROIC/WACC contra una fuente externa (Bloomberg,
  CapitalIQ, etc.) — es una construcción propia desde XBRL crudo, sin
  benchmark de sanity-check más allá de que las medianas del panel
  completo son razonables.
- Mismo alineamiento temporal por fecha real que `02_...md`/`04_...md`
  (`leakage-checking.md`) — hereda esas garantías, no las repite aquí.
