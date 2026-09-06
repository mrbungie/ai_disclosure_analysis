# Cohorte 2021: empresas presentes desde el primer año del panel

Todas las cifras salen de la corrida vigente de `make analytics`
(`report_crosscheck_stats.py` reproduce las tablas numéricas) sobre el panel
de 1.426 empresas-año y 460 empresas: frames de 10-K, DEF 14A y 8-K,
población marcada por el prefiltro v2 (árboles, umbral 0,17 —
`prefilter_evaluation.md` §8.16), lado contable/mercado de
`build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
(ERP geométrico 6,48%, `10_builders_y_recalculo.md`).

Resumen autocontenido de las 99 empresas cuyo primer año en el panel
empresa-año (`data/processed/clusters/firm_year_archetype_behaviors.parquet`)
es 2021 — es decir, empresas con divulgación de IA detectable desde el
inicio de la ventana de datos (2021-2026). 66 de ellas tienen fila en los
6 años; el resto cae bajo el umbral de volumen mínimo de frames en algún
año (82-99 empresas por año). 537 filas empresa-año en total.

```python
p = pd.read_parquet('data/processed/clusters/firm_year_archetype_behaviors.parquet')
first_year = p.groupby('ticker')['year'].min()
cohort = first_year[first_year == 2021].index
c = p[p['ticker'].isin(cohort)]
```

## Composición por arquetipo, año a año

| Año | A cauteloso | B genérico | C cuantificador | D vocal |
|---|---|---|---|---|
| 2021 | 7,1% | 22,2% | 28,3% | 42,4% |
| 2022 | 4,6% | 29,9% | 20,7% | 44,8% |
| 2023 | 7,3% | 28,0% | 13,4% | 51,2% |
| 2024 | 17,6% | 23,1% | 9,9% | 49,5% |
| 2025 | 14,0% | 30,1% | 8,6% | 47,3% |
| 2026 | 8,2% | 43,5% | 9,4% | 38,8% |

**D se mantiene entre 42% y 51% durante cinco años y cae a 38,8% en
2026**, el año parcial (85 empresas, filings hasta mitad de año), así que
esa caída no se puede separar todavía de la estacionalidad del año
incompleto. Lo que se mueve con claridad es el resto: C cae de 28,3% a
9,4% y B sube de 22,2% a 43,5%.

Esa caída de C dentro de una cohorte FIJA es el dato más incómodo del
documento. No es composición: son las mismas 99 empresas. **Más de un
cuarto de la cohorte cuantificaba sus afirmaciones sobre IA en 2021 y
sólo un décimo lo hace en 2026**, mientras la proporción que habla en
registro genérico se duplica. Es consistente con la fuga C→D de 33,3% en
la matriz de transición de abajo.

## Comportamiento agregado, año a año

| Año | n empresas | n frames | % promocional | especificidad | % hipotético | % deployed | % riesgo |
|---|---|---|---|---|---|---|---|
| 2021 | 99 | 967 | 13,8% | 0,22 | 3,2% | 50,2% | 13,4% |
| 2022 | 87 | 1.040 | 15,4% | 0,20 | 3,1% | 49,7% | 13,1% |
| 2023 | 82 | 1.363 | 14,5% | 0,18 | 3,8% | 45,3% | 19,7% |
| 2024 | 91 | 2.819 | 13,3% | 0,15 | 5,1% | 39,9% | 25,6% |
| 2025 | 93 | 3.641 | 12,2% | 0,15 | 4,9% | 39,0% | 27,8% |
| 2026 | 85 | 4.011 | 11,6% | 0,15 | 6,3% | 34,1% | 31,2% |

**El promocional dentro de la cohorte fija baja poco: 13,8%→11,6%, 2,2
p.p., con un repunte en 2022.** Mirando sólo el 10-K la caída es del doble
(la serie 10-K puro de `01_...md`), y la diferencia es composición
documental.

La explicación es directa y ya está medida: la DEF 14A tiene 14,7% de
frames promocionales contra 7,1% del 10-K (`01_...md` #8). A medida que
avanzan los años, la proporción de frames que viene del proxy crece, y
eso sostiene el promocional agregado que el 10-K por sí solo dejaba caer.

Consecuencia para la tesis: **"las empresas se volvieron menos
promocionales sobre IA" era, en buena medida, un artefacto de mirar sólo
el 10-K.** No se volvieron menos promocionales; se volvieron menos
promocionales *en el documento que lee el regulador*. Lo que le dicen al
accionista en el proxy no siguió esa trayectoria. Eso es una hipótesis de
AI-washing más filosa que la original, y está a un análisis de distancia:
separar la serie por formulario dentro de la cohorte fija.

El resto de las tendencias sobrevive: especificidad 0,22→0,15, `deployed`
50,2%→34,1%, riesgo 13,4%→31,2%, hipotético 3,2%→6,3%.

## Estabilidad de arquetipo dentro de la cohorte

Persistencia año-a-año: **60,7%** de los 417 pares consecutivos.

| De \ A | A cauteloso | B genérico | C cuantificador | D vocal |
|---|---|---|---|---|
| **A cauteloso** | 42,2% | 40,0% | 6,7% | 11,1% |
| **B genérico** | 12,1% | 54,3% | 6,0% | 27,6% |
| **C cuantificador** | 4,5% | 16,7% | 45,5% | **33,3%** |
| **D vocal** | 3,2% | 17,4% | 5,3% | **74,2%** |

D es el más estable (74,2%). **C pierde un tercio de sus empresas hacia
D cada año**, contra un flujo inverso D→C de sólo 5,3% — una fuga neta
de más de 6 a 1. Combinado con la caída de C en la composición, el
movimiento dominante de esta cohorte es de cuantificar a promocionar.

La cohorte 2021 es apenas más estable que el panel completo (60,7% vs.
58,8%).

## D vocal: intensidad de `deployed` año a año

| Año | % deployed (media, solo empresas en D) | n empresas-D |
|---|---|---|
| 2021 | 53,1% | 42 |
| 2022 | 60,5% | 39 |
| 2023 | 50,8% | 42 |
| 2024 | 47,6% | 45 |
| 2025 | 41,4% | 44 |
| 2026 | 37,8% | 33 |

Dentro del subgrupo de empresas-en-D de esta cohorte, `deployed` cae de
53,1% a 37,8% — 15,3 p.p., con un número de empresas-D estable hasta
2025 (39-45) y 33 en el 2026 parcial.
Las mismas empresas, con voz igual de vocal, describiendo cada vez menos
despliegue efectivo. Es la versión más limpia del hallazgo central: **la
voz no baja, la sustancia sí.**

## Talk vs. walk

- `revenue_outcome` (año *t*) vs. crecimiento real de revenue del FY
  siguiente: **r = 0,044** (n=425).
- `cost_outcome` (año *t*) vs. crecimiento real de SG&A del FY siguiente:
  **r = −0,016** (n=389): cero, igual que en `02_...md`.

Crecimiento de revenue del FY siguiente, por arquetipo:

| Arquetipo | Mediana | Media | n |
|---|---|---|---|
| A cauteloso | 7,0% | 8,8% | 45 |
| B genérico | 8,9% | 12,5% | 117 |
| C cuantificador | 6,9% | 11,7% | 67 |
| D vocal | **10,9%** | **13,4%** | 196 |

A es el más bajo y D el más alto, en mediana y en media. Con n=45 en A,
esto no aguanta mucho peso.

## Reacción de mercado al filing

Retorno crudo [-1, +5 días hábiles] alrededor del 10-K, por arquetipo:

| Arquetipo | Media | Mediana | Desv. est. | n |
|---|---|---|---|---|
| A cauteloso | −0,43% | 0,30% | 6,98% | 51 |
| B genérico | −0,15% | −0,22% | 7,34% | 155 |
| C cuantificador | 0,33% | 1,14% | 6,69% | 72 |
| D vocal | −0,60% | −0,32% | 6,82% | 230 |

Correlación retorno vs. `promotional_rate`: **r = −0,018**.
Correlación retorno vs. `specificity_index`: **r = −0,013**.

No hay señal: medias dentro de ±0,7 p.p. con desviaciones de 7%, y con
n=51 en A cualquier media cambia de signo al cambiar la muestra.

## Notas

- Cohorte definida por primer año en el panel empresa-año, con el
  mismo umbral de volumen (≥3 frames/año) de `01_ai_disclosure_analytics.md`
  — no todas las 99 empresas tienen fila en los 6 años (66 sí; el resto
  cae bajo el umbral algunos años).
- Guardado en `data/processed/clusters/cohort_2021_crosscheck.parquet`.
- Solo EE.UU., sin ponderar por `inclusion_weight`.
