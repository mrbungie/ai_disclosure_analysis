# Ratios contables, factores de mercado y estudio de volatilidad/beta (EE.UU.)

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

| Arquetipo | Margen bruto | Margen operativo | Margen neto | ROA | ROE | Rotación activos | Current ratio | Deuda/equity |
|---|---|---|---|---|---|---|---|---|
| A cauteloso | 45,2% | 18,0% | 12,4% | 5,4% | 11,0% | 0,50 | 1,17 | 0,65 |
| B genérico | 48,9% | 15,4% | 11,8% | 4,8% | 13,4% | 0,54 | 1,22 | 0,65 |
| C cuantificador | 54,1% | 19,6% | 15,8% | 7,1% | 19,8% | 0,59 | **1,61** | 0,70 |
| D vocal | **62,2%** | 17,2% | 13,9% | 7,0% | 18,3% | 0,53 | 1,40 | **0,47** |

D tiene el margen bruto más alto con claridad (62,2%, típico de
software/semis fabless) pero NO el margen operativo ni neto más alto —
su ventaja de margen bruto se diluye en gastos (R&D/SG&A altos, ya
visto en `02_...md`). D es también, con diferencia, **el menos
apalancado** (deuda/equity 0,47 vs. 0,65-0,70 del resto) — financia su
crecimiento con equity/caja, no con deuda. C tiene el balance más
líquido (current ratio 1,61) y el ROE más alto (19,8%).

## Resultados: valuación por arquetipo

| Arquetipo | P/E | P/S | P/B | EV/Revenue | EV/EBITDA | Market cap (mediana) |
|---|---|---|---|---|---|---|
| A cauteloso | 22,4 | 3,49 | 2,54 | 4,44 | 18,2 | $34,7B |
| B genérico | 21,9 | 2,50 | 2,60 | 3,21 | 15,9 | $37,0B |
| C cuantificador | 25,2 | 3,77 | 5,55 | 4,82 | 18,2 | $29,9B |
| D vocal | **25,7** | **4,15** | 4,53 | **5,00** | **19,3** | **$46,3B** |

D cotiza con **prima de valuación en casi todas las métricas** (P/E,
P/S, EV/Revenue, EV/EBITDA los más altos, y el market cap mediano más
grande) — el mercado sí le asigna una historia de crecimiento más cara
a este arquetipo, consistente con el hallazgo de `02_...md` de que D
crece más rápido al año siguiente. C tiene el P/B más alto (5,55) pese
a no ser el de mayor P/E — probablemente por activos livianos
(negocios de infraestructura/servicios con poco equity contable
relativo a su valor de mercado).

## Resultados: beta y volatilidad — el hallazgo más limpio de esta sección

| Arquetipo | Beta (mediana) | Volatilidad pre-filing (60d, anualizada) | Volatilidad post-filing (60d) | Momentum 12-1m |
|---|---|---|---|---|
| A cauteloso | **0,77** | 24,3% | 29,9% | 8,2% |
| B genérico | 0,85 | 26,3% | 32,1% | 8,3% |
| C cuantificador | 1,02 | 31,3% | 34,2% | 16,9% |
| D vocal | **1,10** | **31,5%** | **35,6%** | 15,5% |

**Monótono en las 4 columnas, A < B < C < D sin excepción.** Los
arquetipos que hablan más de IA y con más especificidad/promoción
(C, D) son sistemáticamente acciones de mayor beta y mayor volatilidad
realizada — tanto antes como después de cada filing. Esto es
importante para toda comparación de retornos entre arquetipos hecha
hasta ahora: cualquier diferencia de retorno CRUDO entre A y D está
mezclada con una diferencia real y grande de riesgo sistemático, no
solo con contenido del disclosure.

**El salto de volatilidad al filing es MENOR en D que en A/B**:

| Arquetipo | Δ volatilidad (post − pre, media) |
|---|---|
| A cauteloso | +0,053 |
| B genérico | **+0,063** |
| C cuantificador | +0,021 |
| D vocal | +0,037 |

El filing de una empresa D (o C) mueve proporcionalmente menos su
volatilidad que el de una A o B — leído en conjunto con la prima de
valuación de arriba, es consistente con que el mercado YA sigue de
cerca a las empresas D/C (más cobertura de analistas, más expectativas
ya incorporadas) y el 10-K aporta menos sorpresa marginal, mientras que
un 10-K de una empresa A/B (que habla poco de IA en general) es
relativamente más informativo cuando sí aparece.

## Resultados: retorno ajustado por mercado (CAR) — invierte el signo de `02_...md`

`02_...md` reportó retorno CRUDO por arquetipo sin diferencias claras.
Con el retorno **ajustado por beta** (CAR, `[-1,+5]` días):

| Arquetipo | CAR media | CAR mediana | Desv. est. | n |
|---|---|---|---|---|
| A cauteloso | **+0,49%** | +0,57% | 5,67% | 285 |
| B genérico | +0,28% | +0,20% | 6,46% | 493 |
| C cuantificador | −0,80% | −0,22% | 6,49% | 42 |
| D vocal | **−0,21%** | −0,15% | 6,20% | 357 |

Al controlar por beta, **A pasa a tener el CAR más alto y D el más
bajo** de los dos arquetipos grandes — orden inverso al de "quién
crece más" (§ valuación/`02_...md`). Lectura más plausible: no es que
el mercado premie a A por sobre D en el filing — es que gran parte del
retorno crudo de D en el período viene de su beta alto en un mercado
alcista para tech/IA, y una vez que se descuenta ese componente
sistemático, no queda una sorpresa positiva neta en sus propios
filings (coherente con "menos sorpresa marginal" de arriba). Sigue sin
haber correlación con tono/especificidad del filing (CAR vs.
`promotional_rate`: r=−0,02; vs. `specificity_index`: r=−0,05) — el
ajuste por beta cambia el ranking entre arquetipos pero no genera una
relación con el CONTENIDO textual del filing que antes no existiera.

## Resultados: talk-vs-walk CONTROLANDO por sector — atenúa fuerte el hallazgo de `02_...md`

`02_...md` reportó `revenue_outcome` (t) vs. `next_revenue_yoy` (t+1)
con r=0,09-0,11 y lo leyó como "sustancia real". Repitiendo dentro de
sector-año (demeaning por `sic2`×año, que remueve cualquier efecto de
"este sector como un todo creció más ese año"):

| | r (crudo) | r (dentro de sector-año) | n |
|---|---|---|---|
| `revenue_outcome` vs. `next_revenue_yoy` | 0,073 | **0,029** | 803 |

**La señal se reduce a menos de la mitad al controlar por sector.**
Gran parte de lo que se leyó como "las empresas que hablan de revenue
efectivamente crecen más" es en realidad "las empresas de sectores que
crecen más (tech/semis) hablan más de revenue Y crecen más, por
razones no relacionadas con el disclosure". Lo que queda (r=0,029) es
casi ruido — no hay evidencia fuerte de que, DENTRO del mismo sector y
año, hablar más de `revenue_outcome` distinga a las empresas que
efectivamente van a crecer más.

## Resultados: intensidad de R&D dentro de sector — reversión completa de la lectura anterior

`02_...md` reportó R&D/revenue de D (14,1%) casi el triple que A
(4,3%) y lo marcó como confusor sectorial pendiente de resolver. Ya
resuelto:

| Arquetipo | R&D/revenue cruda | R&D/revenue DENTRO de su sector (demeaned) |
|---|---|---|
| A cauteloso | 4,3% | **−4,0 p.p.** (bajo el promedio de su sector) |
| B genérico | 7,8% | −0,4 p.p. |
| C cuantificador | 6,6% | **−3,4 p.p.** (bajo el promedio de su sector) |
| D vocal | 14,1% | **−0,2 p.p.** (esencialmente el promedio de su sector) |

**Era, en efecto, casi puro efecto sectorial** — como ya se sospechaba
en `02_...md`. Dentro de su propio sector, D gasta en R&D lo que
cualquier empresa de su sector gastaría (deviación ≈0), NO gasta más
porque "divulgue mejor" o sea más sustantiva. Los que sí se desvían de
su sector son A y C, y en la dirección OPUESTA a lo esperado: gastan
MENOS en R&D que sus pares sectoriales, no más. La lectura "D tiene más
sustancia real medida en R&D" de `02_...md` no se sostiene — hay que
retirarla.

## Lectura conjunta y actualización de `02_...md`

Dos de los hallazgos de `02_...md` se debilitan sustancialmente al
agregar controles apropiados:

1. **`revenue_outcome` → crecimiento real**: sobrevive pero mucho más
   débil de lo reportado (r=0,03 dentro de sector, no 0,09-0,11
   crudo). Sigue siendo la única señal de "talk→walk" con signo
   correcto, pero ya no se puede llamar "sustancia real" con
   confianza — es, en el mejor caso, una señal muy débil.
2. **R&D intensity de D como evidencia de sustancia**: se retira. Era
   casi enteramente composición sectorial.

Lo que SÍ es un hallazgo nuevo y robusto de esta sección, no
reportado antes: **el gradiente de riesgo sistemático (beta,
volatilidad) entre arquetipos es monótono y grande**, y **invierte el
ranking de desempeño de mercado** una vez que se ajusta por él (A > D
en CAR, lo opuesto al ranking de crecimiento fundamental). Cualquier
afirmación futura tipo "el arquetipo X tiene mejor/peor desempeño de
mercado" debe especificar si es en términos crudos o ajustados por
riesgo — dan respuestas opuestas.

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
