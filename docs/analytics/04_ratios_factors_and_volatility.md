# Ratios contables, factores de mercado y estudio de volatilidad/beta (EE.UU.)

> **Recalculado 2026-09-05 con DEF 14A y 8-K.** Los ratios, factores y
> betas no cambiaron; las etiquetas de arquetipo sí (K-means re-ajustado
> sobre la población ampliada). En el camino se corrigió un bug de
> reproducibilidad en la vista `gold_ai_frames` — ver `01_...md`.


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
conclusión de `02_...md` (ver "Lectura conjunta" abajo) — no es un
descarte del hallazgo anterior, es una versión más precisa del mismo.

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

Medianas sobre empresas-año (n=1.363):

| Arquetipo | Margen bruto | Margen oper. | Margen neto | ROA | ROE | Rotación activos | Deuda/Equity | Current ratio | R&D/rev | Capex/rev |
|---|---|---|---|---|---|---|---|---|---|---|
| A cauteloso | 42,9% | 16,2% | 11,6% | 4,9% | 11,9% | 0,53 | 0,63 | 1,22 | 4,9% | 3,2% |
| B genérico | 49,2% | 16,4% | 12,6% | 4,8% | 12,8% | 0,52 | 0,68 | 1,21 | 7,0% | 3,2% |
| C cuantificador | 57,9% | **18,0%** | **14,5%** | 6,6% | **18,5%** | **0,54** | **0,58** | **1,50** | 9,0% | 3,2% |
| D vocal | **58,9%** | 16,8% | 13,0% | **6,9%** | 16,1% | 0,53 | 0,53 | 1,32 | **13,0%** | 2,7% |

C y D dominan, y se reparten los primeros puestos: D tiene el margen
bruto y el ROA más altos y triplica a A en R&D; C tiene el mejor margen
operativo, neto, ROE, rotación de activos y liquidez. **C convierte mejor
lo que gana en resultado final; D gasta más en I+D.** Es coherente con
sus perfiles de discurso: el que cuantifica muestra eficiencia, el que
promociona muestra inversión.

## Resultados: valuación por arquetipo

| Arquetipo | P/E | P/S | P/B | EV/Revenue | EV/EBITDA | Market cap |
|---|---|---|---|---|---|---|
| A cauteloso | 20,5 | 3,24 | 2,50 | 3,76 | 17,6 | $28,7B |
| B genérico | 22,0 | 2,75 | 2,54 | 3,58 | 17,0 | $39,2B |
| C cuantificador | **28,3** | **5,01** | **5,74** | **5,63** | **18,9** | $43,8B |
| D vocal | 23,5 | 3,21 | 4,16 | 3,90 | 16,4 | $43,2B |

**C cotiza más caro que D en todos los múltiplos.** Es una inversión
respecto de la versión anterior, donde D era el más caro. El mercado paga
la prima por el grupo que cuantifica, no por el que promociona — un dato
directamente relevante para la pregunta de AI-washing, aunque no permite
separar "paga por la sustancia del disclosure" de "paga por el sector".

## Resultados: beta y volatilidad — el hallazgo más limpio de esta sección

| Arquetipo | Beta | Vol. pre-filing (60d) | Vol. post-filing (60d) | Momentum 12-1 |
|---|---|---|---|---|
| A cauteloso | 0,79 | 24,4% | 30,8% | 7,1% |
| B genérico | 0,82 | 26,1% | 32,1% | 8,8% |
| C cuantificador | **1,04** | **31,1%** | **34,2%** | **13,3%** |
| D vocal | 1,03 | 28,4% | 32,6% | 12,8% |

C y D empatan arriba (beta 1,04 y 1,03) contra A y B en 0,79-0,82.
**Hablar mucho de IA —cuantificando o promocionando— va de la mano de más
riesgo sistemático**, con la causalidad sin identificar. El patrón
cualitativo se mantuvo respecto de la versión anterior; lo que cambió es
que C ya no es el de beta más bajo sino el más alto, consistente con su
nueva composición (semis y datos en vez de utilities).

## Resultados: retorno ajustado por mercado (CAR) — invierte el signo de `02_...md`

CAR [-1, +5] ajustado por mercado:

| Arquetipo | Media | Mediana | n |
|---|---|---|---|
| A cauteloso | +0,47% | +0,13% | 297 |
| B genérico | +0,20% | +0,08% | 477 |
| C cuantificador | **+0,54%** | **+0,88%** | 148 |
| D vocal | **−0,18%** | −0,16% | 383 |

Una vez descontado el movimiento del mercado, **D es el único arquetipo
con retorno anormal negativo**, y queda último tanto en media como en
mediana. C lidera. El orden se mantuvo respecto de la versión anterior en
lo esencial (A > D), y ahora con C separándose todavía más.

Las magnitudes son chicas y la dispersión grande, así que es direccional,
no concluyente. Pero apunta consistentemente en la misma dirección que
`08_...md`: la voz promocional viene con más riesgo y sin retorno extra.

## Resultados: talk-vs-walk CONTROLANDO por sector

Correlación `behavior_share_revenue_outcome` (t) vs. `next_revenue_yoy`:

| Especificación | r | n |
|---|---|---|
| Cruda | 0,071 | 887 |
| Dentro de sector-año (SIC-2 × año, ambas variables demeaned) | **0,033** | 887 |

Igual que en la versión anterior (0,09 → 0,03), **más de la mitad de la
correlación cruda es composición sectorial**. Lo que queda dentro de
sector-año es 0,033, indistinguible de ruido según el test de permutación
de `05_...md` (p=0,32).

## Resultados: intensidad de R&D dentro de sector

| Arquetipo | R&D/revenue (mediana) | Desvío vs. mediana de su sector |
|---|---|---|
| A cauteloso | 4,9% | −0,5 p.p. |
| B genérico | 7,0% | −0,0 p.p. |
| C cuantificador | 9,0% | +0,4 p.p. |
| D vocal | **13,0%** | **+0,8 p.p.** |

La brecha cruda de D contra A es de 8,1 p.p.; dentro de sector queda en
1,3 p.p. **~84% del efecto es composición sectorial**, con un residuo
positivo chico que no se puede distinguir de un control imperfecto (SIC-2
es granularidad gruesa: dentro de "SIC 73 servicios de cómputo" conviven
perfiles de R&D muy distintos).

## Lectura conjunta y actualización de `02_...md`

Sobreviven al cambio de población, sin cambios cualitativos:

- **Los arquetipos vocales (C y D) tienen más beta y volatilidad** — el
  hallazgo más limpio y estable de los tres documentos financieros.
- **D queda último en CAR ajustado por mercado** — el retorno aparente de
  la voz promocional desaparece al ajustar por riesgo.
- **El talk-vs-walk de revenue es mayormente composición sectorial**
  (0,071 → 0,033 dentro de sector-año).
- **La ventaja de R&D es mayormente sectorial** (~84%).

Aparece con la población nueva:

- **C cotiza más caro que D en todos los múltiplos y le gana en CAR**, lo
  que invierte la lectura anterior de que el mercado premiaba al grupo
  vocal. Premia al que cuantifica.

Ninguna conclusión de esta sección se dio vuelta, a diferencia de lo que
pasó en `07_...md` y `08_...md`. La razón es estructural y vale
registrarla: **este documento compara los cuatro arquetipos completos
(n de 148 a 493 empresas-año), no subgrupos chicos.** Los resultados
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
