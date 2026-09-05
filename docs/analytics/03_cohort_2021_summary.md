# Cohorte 2021: empresas presentes desde el primer año del panel

> **Recalculado 2026-09-05 con DEF 14A y 8-K.** La cohorte pasó de 77 a
> 85 empresas y de 411 a 465 filas empresa-año: con proxies y 8-K, más
> empresas superan el umbral de ≥3 frames ya en 2021. Los insumos
> financieros no cambiaron; las etiquetas de arquetipo sí. En el camino se
> corrigió un bug de reproducibilidad en `gold_ai_frames` — ver `01_...md`.


Resumen autocontenido de las 85 empresas cuyo primer año en el panel
empresa-año (`data/processed/clusters/firm_year_archetype_behaviors.parquet`)
es 2021 — es decir, empresas con divulgación de IA detectable desde el
inicio de la ventana de datos (2021-2026), presentes en 6 de 6 años
posibles (76-85 empresas por año, variando por umbral de volumen mínimo
de frames). 465 filas empresa-año en total.

```python
p = pd.read_parquet('data/processed/clusters/firm_year_archetype_behaviors.parquet')
first_year = p.groupby('ticker')['year'].min()
cohort = first_year[first_year == 2021].index
c = p[p['ticker'].isin(cohort)]
```

## Composición por arquetipo, año a año

| Año | A cauteloso | B genérico | C cuantificador | D vocal |
|---|---|---|---|---|
| 2021 | 4,7% | 20,0% | 27,1% | 48,2% |
| 2022 | 1,3% | 27,6% | 23,7% | 47,4% |
| 2023 | 4,1% | 27,4% | 21,9% | 46,6% |
| 2024 | 11,7% | 26,0% | 11,7% | 50,6% |
| 2025 | 8,8% | 28,7% | 12,5% | 50,0% |
| 2026 | 4,1% | 32,4% | 14,9% | 48,6% |

**D se mantiene notablemente plano: 46-51% los seis años.** Las empresas
que hablaban de IA desde el principio y con voz de líder siguen ahí. Lo
que se mueve es el resto: C cae de 27,1% a 14,9% y B sube de 20,0% a
32,4%.

Esa caída de C dentro de una cohorte FIJA es el dato más incómodo del
documento. No es composición: son las mismas 85 empresas. **Un cuarto de
la cohorte cuantificaba sus afirmaciones sobre IA en 2021 y sólo un
séptimo lo hace en 2026**, mientras la proporción que habla en registro
genérico crece. Es consistente con la fuga C→D de 39,2% en la matriz de
transición de abajo.

## Comportamiento agregado, año a año

| Año | n empresas | n frames | % promocional | especificidad | % hipotético | % deployed | % riesgo |
|---|---|---|---|---|---|---|---|
| 2021 | 85 | 878 | 14,7% | 0,21 | 3,2% | 49,3% | 14,0% |
| 2022 | 83 | 976 | 15,6% | 0,20 | 3,6% | 49,3% | 14,0% |
| 2023 | 81 | 1.281 | 14,8% | 0,18 | 3,7% | 45,9% | 20,0% |
| 2024 | 82 | 2.643 | 13,7% | 0,17 | 5,2% | 39,8% | 25,7% |
| 2025 | 81 | 3.383 | 12,7% | 0,17 | 4,9% | 39,3% | 27,6% |
| 2026 | 76 | 3.707 | 12,1% | 0,17 | 6,0% | 34,2% | 30,6% |

**Este es el resultado que más cambia con la actualización.** La versión
anterior (10-K solo) mostraba el promocional cayendo casi a la mitad
dentro de la cohorte fija: 11,5%→6,4%. Con DEF 14A y 8-K incorporados, la
caída es de 14,7%→12,1%: **2,6 p.p. en vez de 5,1**, y con un repunte en
2022.

La explicación es directa y ya está medida: la DEF 14A tiene 16,3% de
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

El resto de las tendencias sobrevive: especificidad 0,21→0,17, `deployed`
49,3%→34,2%, riesgo 14,0%→30,6%, hipotético 3,2%→6,0%.

## Estabilidad de arquetipo dentro de la cohorte

Persistencia año-a-año: **61,1%** de los 380 pares consecutivos (antes
74,6%).

| De \ A | A cauteloso | B genérico | C cuantificador | D vocal |
|---|---|---|---|---|
| **A cauteloso** | 37,5% | 45,8% | 12,5% | 4,2% |
| **B genérico** | 8,0% | 59,0% | 5,0% | 28,0% |
| **C cuantificador** | 1,4% | 9,5% | 50,0% | **39,2%** |
| **D vocal** | 2,7% | 17,0% | 10,4% | **69,8%** |

D es el más estable (69,8%). **C pierde casi el 40% de sus empresas hacia
D cada año**, contra un flujo inverso D→C de sólo 10,4% — una fuga neta
de casi 4 a 1. Combinado con la caída de C en la composición, el
movimiento dominante de esta cohorte es de cuantificar a promocionar.

La cohorte 2021 es más estable que el panel completo (61,1% vs. 56,8%):
las empresas que hablaban de IA desde el principio tienen una identidad
de disclosure más definida que las que entraron después.

## D vocal: intensidad de `deployed` año a año

| Año | % deployed (media, solo empresas en D) | n empresas-D |
|---|---|---|
| 2021 | 54,8% | 41 |
| 2022 | 59,0% | 36 |
| 2023 | 52,5% | 34 |
| 2024 | 42,8% | 39 |
| 2025 | 43,9% | 40 |
| 2026 | 37,2% | 36 |

Dentro del subgrupo de empresas-en-D de esta cohorte, `deployed` cae de
54,8% a 37,2% — 17,6 p.p., con un número de empresas-D estable (34-41).
Las mismas empresas, con voz igual de vocal, describiendo cada vez menos
despliegue efectivo. Es la versión más limpia del hallazgo central: **la
voz no baja, la sustancia sí.**

## Talk vs. walk

- `revenue_outcome` (año *t*) vs. crecimiento real de revenue del FY
  siguiente: **r = 0,065** (n=365), antes 0,082.
- `cost_outcome` (año *t*) vs. crecimiento real de SG&A del FY siguiente:
  **r = +0,029** (n=167), antes −0,042 — cambia de signo, igual que en
  `02_...md`.

Crecimiento de revenue del FY siguiente, por arquetipo:

| Arquetipo | Mediana | Media | n |
|---|---|---|---|
| A cauteloso | 6,6% | 10,5% | 24 |
| B genérico | 9,9% | 12,2% | 99 |
| C cuantificador | 8,0% | **14,9%** | 69 |
| D vocal | **10,9%** | 12,7% | 173 |

A es el más bajo y D el más alto en mediana; C tiene la media más alta,
arrastrada por su cola derecha (semis en años de expansión). Con n=24 en
A, esto no aguanta mucho peso.

## Reacción de mercado al filing

Retorno crudo [-1, +5 días hábiles] alrededor del 10-K, por arquetipo:

| Arquetipo | Media | Mediana | Desv. est. | n |
|---|---|---|---|---|
| A cauteloso | 0,47% | 0,69% | 6,35% | 26 |
| B genérico | −0,47% | −0,56% | 7,14% | 123 |
| C cuantificador | 0,19% | −0,05% | 7,19% | 78 |
| D vocal | −0,92% | −0,18% | 6,65% | 211 |

Correlación retorno vs. `promotional_rate`: **r = −0,009**.
Correlación retorno vs. `specificity_index`: **r = −0,064**.

Sigue sin haber señal. El −2,85% de media de C en la versión anterior
salía de 17 observaciones; con 78 queda en +0,19%, dentro del ruido.

## Notas

- Cohorte definida por primer año en el panel empresa-año, con el
  mismo umbral de volumen (≥3 frames/año) de `01_ai_disclosure_analytics.md`
  — no todas las 85 empresas tienen fila en los 6 años (caen bajo el
  umbral algunos años).
- Guardado en `data/processed/clusters/cohort_2021_crosscheck.parquet`.
- Solo EE.UU., sin ponderar por `inclusion_weight`.
