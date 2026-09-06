# Ratios contables, factores de mercado y estudio de volatilidad/beta (EE.UU.)

Todas las cifras salen de la corrida vigente de `make analytics`
(`report_crosscheck_stats.py` reproduce las tablas numéricas) sobre el panel
de 1.426 empresas-año y 460 empresas: frames de 10-K, DEF 14A y 8-K,
población marcada por el prefiltro v2 (árboles, umbral 0,17 —
`prefilter_evaluation.md` §8.16), lado contable/mercado de
`build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
(ERP geométrico 6,48%, `10_builders_y_recalculo.md`).

Extiende `02_market_accounting_crosscheck.md` de datos contables básicos
(revenue, R&D, capex, SG&A) a **ratios finales** (rentabilidad,
eficiencia, apalancamiento, valuación) y de retorno crudo a **factores
de mercado** (beta, volatilidad realizada, momentum, retorno ajustado
por mercado) usando los factores Fama-French ya descargados
(`data/raw/market/factors/ff3_daily.parquet`) y control por sector
(SIC-2, `firm_universe`). Responde directamente al hallazgo de
composición sectorial que `02_...md` dejó abierto ("no se puede separar
D-divulga-distinto de D-es-del-sector-que-crece-más").

**Estado: exploratorio.** Un resultado de esta sección **corrige** una
conclusión de `02_...md` (ver "Lectura conjunta" abajo): el CAR ajustado
por mercado invierte el signo del retorno crudo.

## Datos y construcción

Dos tablas nuevas en `data/processed/clusters/`:

| Archivo | Contenido |
|---|---|
| `firm_year_financials_ratios.parquet` | márgenes, ROA/ROE, liquidez, apalancamiento, rotación de activos — contemporáneos al FY disclosed en cada 10-K, + `sic2` |
| `firm_year_market_factors.parquet` | market cap, P/E, P/S, P/B, EV/Revenue, EV/EBITDA, beta, volatilidad realizada pre/post filing, momentum 12-1 meses, retorno ajustado por mercado (CAR) |

**Ratios contables**: mismo alineamiento por fecha real que
`02_...md`/`leakage-checking.md` (hechos XBRL de duración anual
340-380 días = income statement; hechos "instant" = balance sheet).
Encontré y corregí un segundo bug de alineamiento en el camino: los
hechos "instant" de XBRL (`Assets`, `StockholdersEquity`, y sobre todo
`dei:EntityCommonStockSharesOutstanding`) tienen su propia fecha
"as of", que para el balance general coincide con el cierre fiscal pero
para `shares_out` (cifra de portada del 10-K) cae cerca de la fecha de
FILING, no del cierre fiscal — semanas después. Mergear todo por
`period_end` exacto hacía que la búsqueda "fecha más reciente antes del
filing" agarrara la fila de portada (shares outstanding) en vez de la
fila del cierre fiscal real, perdiendo revenue/income de esa fila.
Corregido con `merge_asof` en dos pasos: los ratios de balance/income
se anclan al `period_end` de duración anual (ventana de 20 días);
`shares_out` se busca por separado, anclado al `filing_date` (ventana
de 75 días, cubre el rezago típico de reporte).

**Valuación**: `market_cap = precio de cierre el día hábil antes del
filing × shares_out`. `P/E = precio / EPS diluido`. `EV = market_cap +
long_term_debt − cash`.

**Beta, volatilidad, momentum**: para cada `(ticker, filing_date)`,
regresión OLS de `retorno_exceso_diario = α + β·mktrf` sobre los 252
días hábiles ANTERIORES al filing (mínimo 120 observaciones) —
`mktrf`/`rf` de Fama-French. Volatilidad realizada = desv. est. de
retornos diarios × √252, en ventanas de 60 días antes y después del
filing. Momentum 12-1 = retorno acumulado de t-252 a t-21 (salta el
último mes, convención estándar). **CAR** (retorno ajustado por
mercado) en la ventana `[-1,+5]` = suma de `(retorno_exceso −
β·mktrf)` día a día, usando el β estimado en la ventana pre-filing —
a diferencia del retorno crudo de `02_...md`, esto sí controla por
movimiento general del mercado.

```python
# beta + CAR, ver script completo en la sesión
pre = px.iloc[idx-253:idx-1]  # 252 días ANTES del filing
X = np.column_stack([np.ones(len(pre)), pre['mktrf']])
beta, _ = np.linalg.lstsq(X, pre['excess_ret'], rcond=None)[0]

window = px.iloc[idx-1:idx+6]  # [-1, +5] días alrededor del filing
car = (window['excess_ret'] - beta * window['mktrf']).sum()

# ratios contables: dentro de sector (SIC-2), para separar "divulga distinto"
# de "está en un sector que estructuralmente es distinto"
s['x_demeaned'] = s['x'] - s.groupby(['sic2','year'])['x'].transform('mean')
```

## Resultados: rentabilidad, eficiencia y apalancamiento por arquetipo

Medianas sobre empresas-año (n=1.426):

| Arquetipo | Margen bruto | Margen oper. | Margen neto | ROA | ROE | Rotación activos | Deuda/Equity | Current ratio | R&D/rev | Capex/rev |
|---|---|---|---|---|---|---|---|---|---|---|
| A cauteloso | 43,6% | 16,3% | 11,4% | 4,7% | 12,5% | 0,52 | 0,66 | 1,20 | 5,3% | 3,6% |
| B genérico | 46,7% | 15,5% | 11,5% | 4,7% | 14,4% | 0,53 | 0,73 | 1,25 | 6,9% | 3,4% |
| C cuantificador | 58,0% | **17,9%** | **14,5%** | **6,6%** | **19,3%** | **0,54** | 0,59 | **1,45** | 8,6% | 3,4% |
| D vocal | **59,9%** | 16,2% | 13,0% | **6,6%** | 17,5% | 0,51 | **0,57** | 1,34 | **13,2%** | 2,8% |

C y D dominan, y se reparten los primeros puestos: D tiene el margen
bruto más alto, empata con C en ROA y multiplica por 2,5 el R&D de A; C tiene el mejor margen
operativo, neto, ROE, rotación de activos y liquidez. **C convierte mejor
lo que gana en resultado final; D gasta más en I+D.** Es coherente con
sus perfiles de discurso: el que cuantifica muestra eficiencia, el que
promociona muestra inversión.

## Resultados: valuación por arquetipo

| Arquetipo | P/E | P/S | P/B | EV/Revenue | EV/EBITDA | Market cap |
|---|---|---|---|---|---|---|
| A cauteloso | 22,9 | 2,92 | 2,93 | 3,00 | 16,0 | $31,1B |
| B genérico | 24,2 | 2,89 | 3,29 | 3,08 | 16,0 | $40,4B |
| C cuantificador | **33,7** | **4,59** | **5,99** | **5,34** | **20,2** | $40,8B |
| D vocal | 26,6 | 3,54 | 4,42 | 3,82 | 16,7 | **$51,6B** |

**C cotiza más caro que D en todos los múltiplos.** El mercado paga
la prima por el grupo que cuantifica, no por el que promociona — un dato
directamente relevante para la pregunta de AI-washing, aunque no permite
separar "paga por la sustancia del disclosure" de "paga por el sector".

## Resultados: beta y volatilidad — el hallazgo más limpio de esta sección

| Arquetipo | Beta | Vol. pre-filing (60d) | Vol. post-filing (60d) | Momentum 12-1 |
|---|---|---|---|---|
| A cauteloso | 0,81 | 25,2% | 31,4% | 8,2% |
| B genérico | 0,81 | 26,4% | 31,8% | 9,2% |
| C cuantificador | 1,04 | **30,5%** | 32,3% | **15,1%** |
| D vocal | **1,05** | 29,1% | **33,0%** | 11,8% |

C y D empatan arriba (beta 1,04 y 1,05) contra A y B en 0,81.
**Hablar mucho de IA —cuantificando o promocionando— va de la mano de más
riesgo sistemático**, con la causalidad sin identificar. La brecha de
~0,25 de beta entre los arquetipos vocales y los otros dos es el
resultado más robusto de los tres documentos financieros.

## Resultados: retorno ajustado por mercado (CAR) — invierte el signo de `02_...md`

CAR [-1, +5] ajustado por mercado:

| Arquetipo | Media | Mediana | n |
|---|---|---|---|
| A cauteloso | +0,39% | +0,30% | 354 |
| B genérico | +0,08% | +0,20% | 488 |
| C cuantificador | **+0,67%** | **+0,87%** | 129 |
| D vocal | **−0,18%** | **−0,17%** | 356 |

Una vez descontado el movimiento del mercado, **D es el único arquetipo
con retorno anormal negativo**, y queda último tanto en media como en
mediana. C lidera, con la salvedad de que con n=129 su ventaja se mueve
con la población: bajo un umbral de prefiltro más estricto cae a la par
de A.

Las magnitudes son chicas y la dispersión grande, así que es direccional,
no concluyente. Pero apunta consistentemente en la misma dirección que
`08_...md`: la voz promocional viene con más riesgo y sin retorno extra.

## Resultados: talk-vs-walk CONTROLANDO por sector

Correlación `behavior_share_revenue_outcome` (t) vs. `next_revenue_yoy`:

| Especificación | r | n |
|---|---|---|
| Cruda | 0,049 | 939 |
| Dentro de sector-año (SIC-2 × año, ambas variables demeaned) | **0,019** | 939 |

**Más de la mitad de la correlación cruda es composición sectorial.** Lo
que queda dentro de sector-año es 0,019,
indistinguible de ruido según el test de permutación de `05_...md`
(p=0,53).

## Resultados: intensidad de R&D dentro de sector

| Arquetipo | R&D/revenue (mediana) | Desvío vs. mediana de su sector |
|---|---|---|
| A cauteloso | 5,3% | −0,3 p.p. |
| B genérico | 6,9% | −0,0 p.p. |
| C cuantificador | 8,6% | −0,0 p.p. |
| D vocal | **13,2%** | **+1,2 p.p.** |

La brecha cruda de D contra A es de 7,9 p.p.; dentro de sector queda en
1,5 p.p. **~81% del efecto es composición sectorial**, con un residuo
positivo chico que no se puede distinguir de un control imperfecto (SIC-2
es granularidad gruesa: dentro de "SIC 73 servicios de cómputo" conviven
perfiles de R&D muy distintos).

## Lectura conjunta

Lo que se sostiene:

- **Los arquetipos vocales (C y D) tienen más beta y volatilidad** — el
  hallazgo más limpio y estable de los tres documentos financieros.
- **D queda último en CAR ajustado por mercado** — el retorno aparente de
  la voz promocional desaparece al ajustar por riesgo.
- **El talk-vs-walk de revenue es mayormente composición sectorial**
  (0,049 → 0,019 dentro de sector-año), y la cruda ya no se distingue de
  cero.
- **La ventaja de R&D es mayormente sectorial** (~81%).

Y además:

- **C cotiza más caro que D en todos los múltiplos y le gana en CAR**
  (+0,87% contra −0,17% de mediana), lo que invierte la lectura original
  de que el mercado premiaba al grupo vocal. Premia al que cuantifica —
  con la salvedad de que el CAR de C se mueve con la población (ver
  arriba).

Ninguna conclusión de esta sección se dio vuelta, a diferencia de lo que
pasó en `07_...md` y `08_...md`. La razón es estructural y vale
registrarla: **este documento compara los cuatro arquetipos completos
(n de 129 a 497 empresas-año), no subgrupos chicos.** Los resultados
construidos sobre grupos grandes aguantaron; los de grupos de 4-20
empresas, no.

## Limitaciones

- El ajuste por beta usa un modelo de un factor (CAPM/mercado), no
  Fama-French de 3 factores completo (aunque `smb`/`hml` ya están
  descargados y disponibles para extender esto).
- Beta estimado con OLS simple sobre 252 días — sin shrinkage
  (Vasicek/Blume), sensible a outliers en la ventana de estimación.
- El control sectorial es SIC-2 (grueso) — un sector más granular
  podría atenuar aún más (o revelar variación adicional dentro de
  SIC-2) los resultados de `revenue_outcome` y R&D.
- Ratios contables con outliers extremos (algunas empresas con equity
  o EBITDA cercano a cero generan P/B o EV/EBITDA absurdos) —
  winsorizados al 2-98% para las tablas de arriba, pero no hay una
  regla de exclusión de outliers principiada, solo winsorización
  mecánica.
- No hay panel con efectos fijos de empresa en ninguna de las
  regresiones — el demeaning por sector-año no captura heterogeneidad
  a nivel empresa individual.
- Igual que el resto del proyecto: solo EE.UU., sin ponderar por
  `inclusion_weight`.
