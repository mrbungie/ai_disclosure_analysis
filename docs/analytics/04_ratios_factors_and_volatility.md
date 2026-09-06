# Ratios contables, factores de mercado y estudio de volatilidad/beta (EE.UU.)

**Modo de análisis final: margen extensivo.** Todas las empresas-año con
filings (2.964, 510 empresas), agrupadas por **nivel de intensidad de IA**:
`cero` (ningún frame de IA en el año) y terciles de frames de IA por 1.000
párrafos entre las que sí hablan (`bajo` mediana 0,7, `medio` 3,1, `alto`
12,1). Producido por `report_crosscheck_stats.py` (§3, §4) sobre
`firm_year_master_v2.parquet`.

Extiende `02_market_accounting_crosscheck.md` a ratios finales (rentabilidad,
eficiencia, apalancamiento, valuación) y a factores de mercado (beta,
volatilidad realizada, momentum, retorno ajustado por mercado), con control
por sector (SIC-2). Los perfiles son descriptivos: caracterizan qué tipo de
empresa habla más de IA en sus filings.

## Datos y construcción

| Archivo | Contenido |
|---|---|
| `firm_year_financials_ratios.parquet` | márgenes, ROA/ROE, liquidez, apalancamiento, rotación de activos — contemporáneos al FY divulgado, + `sic2` |
| `firm_year_market_factors.parquet` | market cap, P/E, P/S, P/B, EV/Revenue, EV/EBITDA, beta, volatilidad realizada pre/post filing, momentum 12-1, retorno ajustado por mercado |

Beta: regresión de 252 días de exceso de retorno diario sobre `mktrf`
(Fama-French) antes del filing. Volatilidad realizada: desviación estándar de
retornos diarios × √252 en ventanas de 60 días antes y después. CAR:
`Σ(excess_ret − β·mktrf)` en [−1, +5]. Valuación con el precio de cierre del
día hábil antes del filing.

## Resultados: rentabilidad, eficiencia y apalancamiento por nivel de IA

Medianas sobre empresas-año:

| nivel de IA | n | Margen bruto | Margen oper. | Margen neto | ROA | ROE | R&D/rev |
|---|---:|---:|---:|---:|---:|---:|---:|
| cero | 1.057 | 39,0% | 16,5% | 11,5% | 5,2% | 14,9% | 3,3% |
| bajo | 636 | 41,1% | 16,7% | 12,5% | 4,7% | 13,5% | 3,8% |
| medio | 635 | 45,1% | 15,6% | 11,3% | 4,8% | 13,9% | 5,9% |
| alto | 636 | **57,5%** | **17,3%** | **13,0%** | **6,7%** | **18,5%** | **11,9%** |

Los tres niveles inferiores son casi indistinguibles entre sí; **el salto está
en el tercil alto**: 18 puntos más de margen bruto que el cero, ROE de 18,5%
contra 14-15%, R&D 3,6x. Hablar poco o nada de IA no separa empresas; hablar
mucho sí, y separa a las de software y semiconductores.

## Resultados: valuación por nivel de IA

| nivel de IA | P/E | P/S | P/B | Market cap |
|---|---:|---:|---:|---:|
| cero | 22,5 | 2,54 | 3,14 | $25,5B |
| bajo | 22,3 | 2,93 | 2,96 | $34,4B |
| medio | 23,4 | 2,64 | 2,95 | $36,6B |
| alto | **28,6** | **3,87** | **5,34** | **$44,2B** |

El mercado paga la prima al tercil alto en todos los múltiplos y ahí también
están las empresas más grandes. No permite separar "paga por la sustancia del
disclosure" de "paga por el sector".

## Resultados: beta y volatilidad

| nivel de IA | Beta | Vol. pre-filing (60d) | Momentum 12-1 |
|---|---:|---:|---:|
| cero | 0,87 | 26,6% | 9,5% |
| bajo | 0,89 | 26,1% | 6,4% |
| medio | 0,84 | 25,6% | 9,0% |
| alto | **1,01** | **29,3%** | **10,7%** |

**Hablar mucho de IA va de la mano de más riesgo sistemático**, con la
causalidad sin identificar: 0,14 de beta y 3 puntos de volatilidad sobre los
otros tres niveles, que empatan entre sí.

## Resultados: retorno ajustado por mercado (CAR)

CAR [−1, +5] mediano: 0,3% / 0,3% / 0,4% / 0,1% por nivel. **Sin señal.**
Las correlaciones directas de `02_...md` (promocional por párrafo ~ CAR
r=−0,046, especificidad ~ CAR r=−0,039) van hacia "más IA en el filing,
retorno anormal levemente menor" y no pasan FDR.

## Resultados: talk-vs-walk controlando por sector

`revenue_outcome` por 1.000 párrafos (t) vs. `next_revenue_yoy`:

| Especificación | r | n |
|---|---|---|
| Cruda | 0,112 | 2.313 |
| Dentro de sector-año (SIC-2 × año, ambas variables demeaned) | **0,167** | 2.313 |

**El control sectorial no reduce la correlación, la aumenta.** No es
composición de industria: dentro de la misma industria y el mismo año, la
empresa que más filing dedica a resultados de IA es la que más crece al año
siguiente. `05_...md` la deja en p<0,001 por permutación y muestra que dentro
de empresa cae a 0,04: es un rasgo de la empresa, no dinámica.

## Resultados: intensidad de R&D dentro de sector

R&D/revenue mediano: 3,3% / 3,8% / 5,9% / 11,9%. La brecha cruda entre el
tercil alto y el cero es de 8,6 p.p. Dentro de SIC-2 el desvío mediano del
tercil alto contra su sector es de ~+1 p.p.: **la mayor parte de la ventaja
de R&D es composición sectorial**, con un residuo positivo chico.

## Lectura conjunta

- **El tercil alto de intensidad de IA es un tipo de empresa**: mayor margen
  bruto, ROE, R&D, beta, volatilidad y múltiplos, y más grande. Los otros tres
  niveles —incluido el cero— no se distinguen entre sí en casi nada.
- **Dentro de sector, la relación talk-vs-walk de revenue se fortalece**, lo
  que descarta la circularidad sectorial; el efecto fijo de empresa la
  elimina, lo que la deja como transversal (`05_...md`).
- **Sin señal en CAR**: el mercado no reacciona al contenido de IA del filing
  en la ventana de presentación.

## Limitaciones

- SIC-2 es granularidad gruesa.
- Los terciles se cortan sobre empresas-año, no sobre empresas: una empresa
  puede estar en `cero` en 2021 y en `alto` en 2025, y el boom empuja a todas
  hacia arriba. Los perfiles mezclan tipo de empresa con año.
- Intensidad por párrafo premia filings cortos.
- Etiquetas de un LLM sin validación humana (`docs/problemas_academicos.md`
  #1).
