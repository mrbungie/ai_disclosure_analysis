# Cohorte 2021: empresas presentes desde el primer año del panel

Resumen autocontenido de las 77 empresas cuyo primer año en el panel
empresa-año (`data/processed/clusters/firm_year_archetype_behaviors.parquet`)
es 2021 — es decir, empresas con divulgación de IA detectable desde el
inicio de la ventana de datos (2021-2026), presentes en 6 de 6 años
posibles (58-77 empresas por año, variando por umbral de volumen mínimo
de frames). 411 filas empresa-año en total.

```python
p = pd.read_parquet('data/processed/clusters/firm_year_archetype_behaviors.parquet')
first_year = p.groupby('ticker')['year'].min()
cohort = first_year[first_year == 2021].index
c = p[p['ticker'].isin(cohort)]
```

## Composición por arquetipo, año a año

| Año | A cauteloso | B genérico | C cuantificador | D vocal |
|---|---|---|---|---|
| 2021 | 7,8% | 16,9% | 10,4% | 64,9% |
| 2022 | 5,9% | 23,5% | 5,9% | 64,7% |
| 2023 | 3,0% | 24,2% | 4,5% | 68,2% |
| 2024 | 7,1% | 32,9% | 2,9% | 57,1% |
| 2025 | 1,4% | 40,3% | 2,8% | 55,6% |
| 2026 | 5,2% | 36,2% | 1,7% | 56,9% |

Esta cohorte arranca con dos tercios de sus empresas en D (líder vocal)
en 2021 y se mantiene mayoritariamente ahí (55-68%) los 6 años. B crece
de forma sostenida (16,9%→36,2%), a expensas casi exclusivamente de C
(que cae de 10,4% a 1,7%). A se mantiene chico y sin tendencia clara
(entre 1,4% y 7,8%, sin monotonía).

## Comportamiento agregado, año a año

| Año | n empresas | n frames | % promocional | especificidad | % hipotético | % deployed | % riesgo |
|---|---|---|---|---|---|---|---|
| 2021 | 77 | 676 | 11,5% | 0,199 | 4,7% | 50,0% | 17,0% |
| 2022 | 68 | 699 | 11,5% | 0,189 | 4,8% | 49,7% | 18,5% |
| 2023 | 66 | 951 | 9,2% | 0,165 | 3,6% | 49,1% | 22,9% |
| 2024 | 70 | 1.886 | 8,0% | 0,140 | 8,3% | 40,0% | 33,4% |
| 2025 | 72 | 2.471 | 7,2% | 0,141 | 7,0% | 37,6% | 37,2% |
| 2026 | 58 | 2.391 | 6,4% | 0,140 | 10,0% | 33,9% | 42,5% |

Dentro de esta cohorte fija, el % promocional cae a poco más de la
mitad entre 2021 y 2026 (11,5%→6,4%), la especificidad cae de 0,199 a
0,140, y `deployed` cae de 50,0% a 33,9% — mientras el volumen de
frames por empresa se multiplica más de 5 veces (676→2.391 con menos
empresas activas en 2026 que en 2021). El % de riesgo sube de forma
sostenida (17,0%→42,5%) y el % hipotético repunta claramente desde
2024 (3,6%→10,0%).

## Estabilidad de arquetipo dentro de la cohorte

Persistencia año-a-año: **74,6%** de los pares consecutivos se quedan
en el mismo arquetipo.

| De \ A | A cauteloso | B genérico | C cuantificador | D vocal |
|---|---|---|---|---|
| **A cauteloso** | 29,4% | 52,9% | 0,0% | 17,6% |
| **B genérico** | 6,2% | 72,9% | 1,0% | 19,8% |
| **C cuantificador** | 0,0% | 11,1% | 27,8% | 61,1% |
| **D vocal** | 2,0% | 11,8% | 3,0% | **83,3%** |

D es el arquetipo más estable de la cohorte (83,3% se queda en D al año
siguiente). A es el menos estable (29,4%; más de la mitad sube a B).
C, cuando sale, termina mayoritariamente en D (61,1%).

## D vocal: intensidad de `deployed` año a año

| Año | % deployed (media, solo empresas en D) | n empresas-D |
|---|---|---|
| 2021 | 58,2% | 50 |
| 2022 | 59,3% | 44 |
| 2023 | 56,0% | 45 |
| 2024 | 48,1% | 40 |
| 2025 | 46,9% | 40 |
| 2026 | 43,9% | 33 |

Dentro del subgrupo de empresas-en-D de esta cohorte, `deployed` cae de
58,2% a 43,9% en el período — una caída de 14,3 p.p. con un número de
empresas-D relativamente estable (33-50 por año, sin gran entrada de
empresas nuevas al subgrupo).

## Talk vs. walk

- `revenue_outcome` (año *t*) vs. crecimiento real de revenue del FY
  siguiente: **r = 0,082** (n=331).
- `cost_outcome` (año *t*) vs. crecimiento real de SG&A del FY
  siguiente: **r = −0,042** (n=145).

Crecimiento de revenue del FY siguiente, por arquetipo:

| Arquetipo | Mediana | Media | n |
|---|---|---|---|
| A cauteloso | 11,3% | 13,2% | 17 |
| B genérico | 9,8% | 12,0% | 96 |
| C cuantificador | 8,0% | 6,5% | 17 |
| D vocal | 11,2% | 15,0% | 201 |

## Reacción de mercado al filing

Retorno crudo [-1, +5 días hábiles] alrededor del 10-K, por arquetipo:

| Arquetipo | Media | Mediana | Desv. est. | n |
|---|---|---|---|---|
| A cauteloso | 0,36% | 0,12% | 4,12% | 19 |
| B genérico | −0,26% | −0,35% | 7,45% | 118 |
| C cuantificador | −2,85% | −3,80% | 8,81% | 17 |
| D vocal | −0,67% | −0,41% | 7,08% | 239 |

Correlación retorno vs. `promotional_rate`: **r = −0,002**.
Correlación retorno vs. `specificity_index`: **r = −0,086**.

## Notas

- Cohorte definida por primer año en el panel empresa-año, con el
  mismo umbral de volumen (≥3 frames/año) de `01_ai_disclosure_analytics.md`
  — no todas las 77 empresas tienen fila en los 6 años (caen bajo el
  umbral algunos años).
- Guardado en `data/processed/clusters/cohort_2021_crosscheck.parquet`.
- Solo EE.UU., sin ponderar por `inclusion_weight`.
