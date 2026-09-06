# Circularidad, placebo y corrección por comparaciones múltiples

Todas las cifras salen de la corrida vigente de `make analytics`
(`report_crosscheck_stats.py` reproduce las tablas numéricas) sobre el panel
de 1.426 empresas-año y 460 empresas: frames de 10-K, DEF 14A y 8-K,
población marcada por el prefiltro v2 (árboles, umbral 0,17 —
`prefilter_evaluation.md` §8.16), lado contable/mercado de
`build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
(ERP geométrico 6,48%, `10_builders_y_recalculo.md`).

Auditoría pedida explícitamente después de discutir qué tan circular
es el diseño de `02_market_accounting_crosscheck.md` /
`04_ratios_factors_and_volatility.md` (el arquetipo se construye a
partir de intensidad/especificidad del discurso de IA, que está casi
mecánicamente correlacionada con pertenecer a un sector tech — así que
"D crece más" podía ser casi tautológico) y el riesgo de sobre-vender
un hallazgo "interesante" (positivo o neutro) sin someterlo al mismo
escrutinio en ambas direcciones. Tres chequeos, todos sobre
`data/processed/clusters/firm_year_master_v2.parquet`.

**Resultado headline: de once correlaciones del cruce contable/mercado,
ninguna sobrevive a la corrección por comparaciones múltiples, y la
cruda de revenue (r=0,049) ya no se distingue de ruido por permutación
(p=0,12). Lo único que queda es que el efecto fijo de empresa encuentra
una señal mayor (0,084) que la de dentro-de-sector (0,019) — débil, pero
en la dirección contraria a "todo es composición".**

## 1. Placebo / permutación: ¿el r crudo y el r dentro-de-sector son distinguibles de ruido?

2.000 permutaciones del predictor contra el outcome fijo, semilla 42:

| Especificación | r observado | p (permutación) | n |
|---|---|---|---|
| `revenue_outcome` ~ `next_revenue_yoy`, crudo | 0,049 | **0,115** | 939 |
| Ídem, dentro de sector-año | 0,019 | 0,533 | 939 |

**La correlación cruda no se distingue de ruido (p=0,12) y la de dentro
de sector-año menos todavía (p=0,53).** Una vez
que se controla por
sector, "hablar de revenue impulsado por IA" no predice el crecimiento
de revenue mejor que una asignación al azar de las etiquetas.

## 2. Efectos fijos de empresa (within-firm): la prueba más estricta para circularidad

Pregunta: ¿cuando UNA MISMA empresa habla más de revenue de IA que su
propio promedio, crece más que su propio promedio? Esto elimina de raíz
cualquier confusor fijo por empresa (sector, modelo de negocio, tamaño,
estilo de redacción del filing).

| Especificación | r | n |
|---|---|---|
| (a) Primeras diferencias (Δ año a año dentro de cada ticker) | **0,088** | 562 |
| (b) Demeaning por empresa (equivalente a efectos fijos) | **0,084** | 939 |

Ambas siguen siendo positivas y **mayores que la correlación dentro de
sector-año (0,019) y que la cruda (0,049)**. Es el resultado más
interesante de este documento: la señal within-firm es más fuerte que la
between-firm-dentro-de-sector.

Interpretación: la parte de la correlación cruda que es genuina parece
ser **temporal dentro de la empresa** (cuando una empresa empieza a
hablar más de resultados de IA, algo cambia de verdad en su trayectoria)
más que transversal entre empresas. El orden relativo —within-firm >
within-sector— es lo que sostiene la lectura.

Caveat que no cambia: r≈0,08-0,09 sigue siendo una correlación débil, y sin
test de permutación propio (el placebo de §1 es sobre la cruda). "No es
puramente circular" no es lo mismo que "es fuerte".

## 3. Corrección por comparaciones múltiples (FDR, Benjamini-Hochberg) sobre las 11 correlaciones reportadas

| # | Par | r | p | Umbral BH | ¿Pasa FDR 5%? |
|---|---|---|---|---|---|
| 1 | `ai_infrastructure` ~ `next_capex_yoy` | 0,086 | 0,0131 | 0,0045 | No |
| 2 | `n_frames` ~ `ret_m1_p5` | −0,050 | 0,0668 | 0,0091 | No |
| 3 | `promotional_rate` ~ `car_m1_p5` | −0,046 | 0,0934 | 0,0136 | No |
| 4 | `revenue_outcome` ~ `next_revenue_yoy` | 0,049 | 0,1335 | 0,0182 | No |
| 5 | `ai_investment` ~ `next_rd_expense_yoy` | 0,044 | 0,3239 | 0,0227 | No |
| 6 | `promotional_rate` ~ `ret_m1_p5` | −0,018 | 0,5154 | 0,0273 | No |
| 7 | `specificity_index` ~ `car_m1_p5` | −0,016 | 0,5558 | 0,0318 | No |
| 8 | `ai_infrastructure` ~ `next_rd_expense_yoy` | 0,019 | 0,6740 | 0,0364 | No |
| 9 | `cost_outcome` ~ `next_sga_expense_yoy` | −0,011 | 0,7533 | 0,0409 | No |
| 10 | `specificity_index` ~ `ret_m1_p5` | −0,008 | 0,7722 | 0,0455 | No |
| 11 | `ai_investment` ~ `next_capex_yoy` | −0,007 | 0,8359 | 0,0500 | No |

**Ninguna de las 11 sobrevive al FDR.** La primera del ranking,
`ai_infrastructure` ~ `capex`, tiene p=0,013 contra un umbral BH de
0,0045: lejos. `revenue_outcome` ~ `next_revenue_yoy` es cuarta con
p=0,13.

Lo defendible es: **de once correlaciones testeadas en todo el cruce
contable/mercado, ninguna se distingue de lo esperable por azar
múltiple.** Es un resultado limpio de defender: el instrumento de texto
no predice resultados financieros al año
siguiente, y ahora eso está medido con cobertura XBRL decente
(`10_...md`) en vez de sobre el subconjunto de filers con el tag más
común.

`cost_outcome` ~ `next_sga_expense_yoy`, la "señal de washing" de la
hipótesis de washing, queda novena de once con p=0,75 y r=−0,011. Ver
`02_...md` §3.

`ai_infrastructure` ~ `next_capex_yoy` depende de la cobertura de capex:
con el 69% de filers que reportan el tag más común da r≈0,11, con la
cobertura completa de 87% da 0,086. La relación "infraestructura
declarada → capex real" se apoya en parte en qué filers tienen dato.

## Lectura conjunta: ¿qué tan circular es, entonces?

Con evidencia, no sólo con intuición:

- **La preocupación de circularidad era correcta para el resultado
  reportado en `04_...md` como "atenuado pero sobreviviente"** (r=0,019
  dentro de sector-año) — el placebo confirma que ESE número específico
  es ruido (p=0,53), y ahora también lo es la cruda (p=0,12). El chequeo
  hace más sólido al hallazgo neutro, no lo contradice.
- **Pero la circularidad NO explica todo.** El efecto fijo de empresa —el
  control más estricto posible contra "esto es sólo composición sectorial
  o de nivel-empresa fijo"— encuentra una señal MÁS fuerte (r=0,084-0,088)
  que la de dentro-de-sector (0,019) para exactamente la misma relación.
  Es el opuesto de lo que predeciría "todo es
  circular": si fuera pura composición fija, el efecto fijo debería haber
  hecho desaparecer la señal, no fortalecerla.
- **Ya no hay una "señal más robusta del cruce contable".**
  `ai_infrastructure` → `capex` (r=0,086, p=0,013) es la primera del
  ranking y queda a un orden de magnitud del umbral FDR.

Los chequeos de acá se corren sobre el panel completo (n=562 a 1.358), no
sobre subgrupos de 4-10 empresas, y por eso son los más estables del
cruce financiero.

## Qué implica para los otros documentos

- `02_...md` §3: la "señal de washing" de costos (`cost_outcome` ~ SG&A)
  está en r=−0,011, novena de once en el ranking FDR (p=0,75). No es un
  hallazgo.
- `02_...md` §1 y `04_...md`, talk-vs-walk: la cruda (r=0,049) no es
  distinguible de ruido (placebo p=0,12); la de dentro de sector (0,019)
  menos (p=0,53). Revenue no es "evidencia de que el instrumento mide
  algo".
- `04_...md`, `ai_investment`/`ai_infrastructure`: `ai_infrastructure`/
  `capex` (r=0,086) no pasa FDR. No queda ningún par del cruce contable
  para reportar como señal.
- Hallazgo más defendible de todo el cruce contable: el resultado de
  efectos fijos de empresa (§2), within-firm > within-sector. Sigue sin ser
  fuerte.

## Limitaciones

- El placebo solo se corrió para `revenue_outcome`→`next_revenue_yoy`
  (el caso con la historia más fuerte) — no para las otras 10
  correlaciones de la tabla FDR; barajar todas sería el chequeo
  completo.
- Efectos fijos de empresa con solo 232 empresas y 2-3 observaciones
  por empresa en promedio — poca potencia para detectar efectos
  chicos, y sensible a outliers de crecimiento año a año (recortados a
  |growth|<300% pero no winsorizados más finamente).
- FDR (Benjamini-Hochberg) asume las pruebas razonablemente
  independientes — varias de las 11 comparten la misma variable
  dependiente o independiente (`next_revenue_yoy` aparece 2 veces,
  `promotional_rate`/`specificity_index` aparecen 4 veces cada una),
  así que la independencia es aproximada, no exacta.
- No se testeó reverse causality de forma directa para el hallazgo de
  efectos fijos (§2) — es la explicación alternativa más plausible al
  resultado y queda abierta.
