# Cruce con mercado y contabilidad (EE.UU.)

**Modo de análisis final: margen extensivo.** El panel son **todas las
empresas-año con filings** (10-K, 10-Q, DEF 14A, 8-K): 2.964 filas, 510
empresas, 2021-2026. Lo que la empresa dice de IA se mide como **intensidad
por 1.000 párrafos** de sus filings del año, con **cero** cuando no habla de
IA. Nada condiciona a hablar de IA: la empresa que no menciona IA es un cero,
no una fila que falta. El 36% de las empresas-año no tiene ningún frame de IA
(64% en 2021, 6% en 2026); sólo 17 empresas de 510 no hablan de IA en ningún
año.

Producido por `build_firm_panels.py` (`firm_year_master_v2.parquet`, sobre
`ai_intensity.py`) y `report_crosscheck_stats.py`, que reproduce cada tabla
numérica de este documento y de `04`, `05` y `08`. Lado contable/mercado de
`build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
(`10_builders_y_recalculo.md`).

Cruce de la divulgación de IA con datos externos al texto: XBRL (¿la
sustancia declarada se refleja en los números?) y precios (¿el mercado
reacciona al contenido del filing?). Responde al hueco de la revisión de
literatura de `docs/thesis_proposal.md` (Eisfeldt et al., Basnet et al.).

## Datos y construcción

| Archivo | Contenido |
|---|---|
| `firm_year_master_v2.parquet` | todas las empresas-año con filings: intensidades de IA por 1.000 párrafos (`frames_per_1k`, `promo_per_1k`, `revenue_outcome_per_1k`, …), `any_ai`, ratios contables, factores de mercado, crecimiento t+1; etiquetas del panel condicionado (`archetype`, tasas) donde la empresa habló de IA |
| `firm_year_full_crosscheck.parquet` | lo mismo con niveles XBRL crudos y el retorno de ventana |

**Financials**: XBRL alineado por fecha real —el `period_end` más reciente
antes del `filing_date` es el FY divulgado; el siguiente de ese ticker es el
FY futuro— con cadenas de fallback de conceptos (cobertura de capex 87%,
SG&A 80%, `net_margin` 98%). `next_*_yoy` se anula cuando el gap fiscal sale
de [340, 380] días. Crecimientos con |YoY| > 300% se descartan.

**Retornos**: `adj_close` del día hábil anterior al `filing_date` del 10-K
contra el quinto día hábil posterior (`ret_m1_p5`); `car_m1_p5` resta el
mercado ajustado por beta (`04_...md`).

**Texto**: frames de `gold_ai_frames` contados por documento y divididos por
los párrafos puntuables del conjunto de filings del año. `revenue_outcome`,
`ai_investment`, `ai_infrastructure`, `cost_outcome` son frames con ese
concepto por 1.000 párrafos.

## Resultados: "talk vs. walk"

### 1. `revenue_outcome` (lo que dicen) vs. crecimiento de revenue real al año fiscal siguiente

| especificación | r | n | p (permutación) |
|---|---:|---:|---:|
| cruda | **0,112** | 2.313 | <0,001 |
| dentro de sector-año (SIC-2 × año) | **0,167** | 2.313 | <0,001 |
| dentro de empresa (efectos fijos) | 0,038 | 2.313 | |
| primeras diferencias dentro de empresa | 0,114 | 1.820 | |
| cruda, sólo empresas-año que hablan de IA | 0,201 | 1.345 | |

**Cuanto más de su filing dedica una empresa a resultados de IA, más crece
su revenue al año siguiente**, y la relación es más fuerte dentro de
sector-año que cruda: no es composición de industria. Pasa el FDR (abajo).

Dos lecturas que la cifra tiene que cargar:

- **Es transversal, no temporal.** Dentro de empresa la correlación cae a
  0,04. Dedicar más filing a IA identifica un TIPO de empresa que crece más;
  no predice que la misma empresa crezca más el año en que habla más. Para
  la pregunta de washing esto importa: el instrumento separa empresas, no
  detecta cambios de conducta.
- **El margen extensivo puro va al revés.** Hablar de IA en absoluto
  (`any_ai`) tiene r=−0,07 con el crecimiento: las empresas-año sin ningún
  frame de IA crecen más (12,0% contra 8,8% de mediana), porque en 2021-2022
  quien no hablaba de IA era la empresa chica en expansión y quien hablaba,
  la grande. Hablar de IA no es señal de nada; cuánto del filing se dedica a
  resultados de IA, sí.

Crecimiento de revenue del FY siguiente por nivel de intensidad de IA del año
(cero = ningún frame; terciles entre quienes hablan):

| nivel de IA | frames por 1.000 párrafos (mediana) | next FY revenue YoY (mediana) | n |
|---|---:|---:|---:|
| cero | 0 | 8,1% | 1.057 |
| bajo | 0,7 | 5,7% | 636 |
| medio | 3,1 | 5,6% | 635 |
| alto | 12,1 | **8,7%** | 636 |

La forma es de U: los que no hablan crecen, los que hablan mucho crecen, los
que hablan poco no. El extremo alto es software/semis; el cero, 2021-2022.

### 2. `ai_investment` / `ai_infrastructure` vs. capex y R&D reales

| declarado (t), por 1.000 párrafos | vs. crecimiento real (FY t+1) | r | p | pasa FDR |
|---|---|---:|---:|---|
| `ai_infrastructure` | `capex_yoy` | **0,107** | <0,0001 | **sí** |
| `ai_infrastructure` | `rd_expense_yoy` | 0,059 | 0,059 | no |
| `ai_investment` | `capex_yoy` | 0,042 | 0,057 | no |
| `ai_investment` | `rd_expense_yoy` | 0,044 | 0,158 | no |

La asimetría se mantiene: decir "invertimos en IA" no predice el capex del
año siguiente; decir "construimos infraestructura de IA" sí, y es la segunda
correlación que sobrevive al FDR.

### 3. `cost_outcome` vs. SG&A real

`cost_outcome` por 1.000 párrafos contra crecimiento de SG&A: **r = −0,013**
(p=0,57, n=2.038 con dato). La hipótesis de washing —quienes más enmarcan la
IA como ahorro de costos muestran SG&A creciendo MÁS— no aparece. Cero.

### 4. Intensidad de R&D por nivel de IA (contemporánea)

| nivel de IA | R&D / revenue (mediana) |
|---|---:|
| cero | 3,3% |
| bajo | 3,8% |
| medio | 5,9% |
| alto | **11,9%** |

Monótono y de 3,6x entre extremos. Es lo que predeciría la composición
sectorial del nivel alto (software/semis); `04_...md` mide cuánto queda
dentro de sector.

## Resultados: reacción de mercado al filing

Retorno crudo [−1, +5 días hábiles] alrededor del 10-K, mediana por nivel de
IA: 0,1% / 0,1% / 0,3% / 0,1%. CAR ajustado por mercado: 0,3% / 0,3% / 0,4%
/ 0,1%. Correlaciones directas sobre las 2.746 empresas-año con retorno:

| | r | p |
|---|---:|---:|
| promocional por 1.000 párrafos ~ retorno | −0,025 | 0,19 |
| promocional por 1.000 párrafos ~ CAR | −0,046 | 0,018 |
| especificidad por 1.000 párrafos ~ CAR | −0,039 | 0,041 |
| frames de IA por 1.000 párrafos ~ retorno | −0,003 | 0,89 |
| `any_ai` ~ CAR | −0,053 | 0,006 |

**Ninguna sobrevive al FDR.** Lo poco que hay va en una sola dirección: más
promoción de IA, más IA en absoluto, retorno anormal levemente menor en la
ventana del filing. Es direccional, chico, y no pasa la corrección.

## Lectura conjunta

De los tres tipos de cruce, **dos relaciones contables sobreviven al FDR**
—dedicar más filing a resultados de IA predice más revenue, dedicar más a
infraestructura de IA predice más capex, ambas r≈0,11— y **ninguna de
mercado**. Las dos sobreviven al control de sector-año (la de revenue sube
a 0,17) y se van dentro de empresa (0,04): **son rasgos de empresa, no
respuestas a lo que la empresa hizo ese año.** El instrumento de texto
identifica qué empresas están construyendo con IA; no detecta cuándo una
empresa cambia.

Para la tesis, eso acota el uso del cruce financiero: sirve para
caracterizar segmentos (`04`, `07`, `08`), no para medir washing como
desviación temporal entre lo dicho y lo hecho. Ese trabajo lo hace la brecha
entre canales (`14_...md`).

## Limitaciones (leer antes de citar cualquier número de esta sección)

- **Sin control por sector más fino que SIC-2.** Dentro de "SIC 73 servicios
  de cómputo" conviven perfiles de I+D muy distintos; el residuo dentro de
  sector-año puede ser sub-sector.
- **Correlaciones, no un panel con dinámica.** El efecto fijo de empresa
  elimina la señal, lo que dice que es entre empresas; un modelo con rezagos
  y controles de tamaño sería el siguiente paso.
- **`capex` con 87% de cobertura XBRL y R&D con 45%**: los cruces con R&D
  tienen la mitad de la muestra.
- **Ventana de retorno fija en 5 días hábiles**, sin robustez con otras
  ventanas.
- **Denominador de párrafos.** La intensidad por 1.000 párrafos premia a la
  empresa con filings cortos; `frames_per_1k` entra como control en los
  perfiles de `04` y como covariable en `09`.
- Igual que el resto del proyecto: solo EE.UU., etiquetas de un LLM sin
  validación humana (`docs/problemas_academicos.md` #1), población filtrada
  por el prefiltro (recall 0,96-0,98 según formulario).
