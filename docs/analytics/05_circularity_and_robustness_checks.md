# Circularidad, placebo y corrección por comparaciones múltiples

**Modo de análisis final: margen extensivo** — todas las empresas-año con
filings (2.964, 510 empresas), intensidad por 1.000 párrafos con ceros
(`02_...md`). Todo sale de `report_crosscheck_stats.py`.

Auditoría del diseño de `02_market_accounting_crosscheck.md` /
`04_ratios_factors_and_volatility.md`: la intensidad de IA está casi
mecánicamente correlacionada con pertenecer a un sector tech, así que "las
que hablan de IA crecen más" podía ser tautológico. Tres chequeos, todos sobre
`firm_year_master_v2.parquet`.

**Resultado headline: de once correlaciones del cruce contable/mercado, dos
sobreviven a la corrección por comparaciones múltiples (revenue e
infraestructura, r≈0,11), ambas sobreviven al control de sector-año —la de
revenue sube a 0,17— y ambas desaparecen dentro de empresa (0,04). Son
señales transversales, no circularidad sectorial ni dinámica.**

## 1. Placebo / permutación: ¿el r crudo y el r dentro-de-sector son distinguibles de ruido?

Se permuta la variable de texto 2.000 veces y se mide qué tan seguido el azar
produce un |r| igual o mayor.

| Especificación | r observado | p (permutación) | n |
|---|---|---|---|
| `revenue_outcome` por 1.000 párrafos ~ `next_revenue_yoy`, crudo | 0,112 | **<0,001** | 2.313 |
| Ídem, dentro de sector-año | 0,167 | **<0,001** | 2.313 |

Ni la cruda ni la de dentro de sector-año son ruido. Y el control sectorial
la AUMENTA: dentro de la misma industria y el mismo año, la empresa que
dedica más filing a resultados de IA crece más que sus pares. No es
composición de sector.

## 2. Efectos fijos de empresa (within-firm): la prueba más estricta para circularidad

Pregunta: ¿cuando UNA MISMA empresa dedica más filing a resultados de IA que
su propio promedio, crece más que su propio promedio?

| Especificación | r | n |
|---|---|---|
| (a) Primeras diferencias (Δ año a año dentro de cada ticker) | 0,114 | 1.820 |
| (b) Demeaning por empresa (equivalente a efectos fijos) | **0,038** | 2.313 |

El demeaning por empresa se lleva la señal: 0,038. Las primeras diferencias
la conservan (0,114), pero en un panel que arranca en 2021 con dos tercios de
ceros las primeras diferencias son en buena parte "empezó a hablar de IA"
contra "no habló", que es el margen extensivo otra vez.

Lectura: **la relación es un rasgo de la empresa**, no una respuesta a lo que
la empresa dijo ese año. Eso descarta la circularidad sectorial (§1) pero no
otra: la empresa que construye con IA y la que lo escribe en su 10-K son la
misma empresa por razones anteriores a ambas cosas. El cruce no identifica
dirección.

## 3. Corrección por comparaciones múltiples (FDR, Benjamini-Hochberg) sobre las 11 correlaciones reportadas

Las 11 correlaciones del cruce contable/mercado, en intensidad por 1.000
párrafos, ordenadas por p:

| # | Par | r | p | Umbral BH | ¿Pasa FDR 5%? |
|---|---|---|---|---|---|
| 1 | `revenue_outcome` ~ `next_revenue_yoy` | 0,112 | <0,0001 | 0,0045 | **Sí** |
| 2 | `ai_infrastructure` ~ `next_capex_yoy` | 0,107 | <0,0001 | 0,0091 | **Sí** |
| 3 | promocional ~ `car_m1_p5` | −0,046 | 0,0175 | 0,0136 | No |
| 4 | especificidad ~ `car_m1_p5` | −0,039 | 0,0414 | 0,0182 | No |
| 5 | `ai_investment` ~ `next_capex_yoy` | 0,042 | 0,0571 | 0,0227 | No |
| 6 | `ai_infrastructure` ~ `next_rd_expense_yoy` | 0,059 | 0,0591 | 0,0273 | No |
| 7 | `ai_investment` ~ `next_rd_expense_yoy` | 0,044 | 0,1583 | 0,0318 | No |
| 8 | promocional ~ `ret_m1_p5` | −0,025 | 0,1860 | 0,0364 | No |
| 9 | `cost_outcome` ~ `next_sga_expense_yoy` | −0,013 | 0,5683 | 0,0409 | No |
| 10 | especificidad ~ `ret_m1_p5` | −0,008 | 0,6611 | 0,0455 | No |
| 11 | frames de IA ~ `ret_m1_p5` | −0,003 | 0,8874 | 0,0500 | No |

**Dos de once pasan, con holgura** (p<0,0001 contra umbrales de 0,0045 y
0,0091). El tercero está cerca (promocional contra CAR, p=0,018 contra
0,014) y va en la dirección de "más promoción, peor retorno anormal", pero no
pasa.

Los dos que pasan son los dos contables de "lo que dicen que hacen contra lo
que hicieron": resultados de IA contra revenue, infraestructura de IA contra
capex. Los de mercado no pasan ninguno; el de costos —la hipótesis de
washing más directa del cruce— está en cero.

## Lectura conjunta: ¿qué tan circular es, entonces?

- **No es circularidad sectorial.** El control de sector-año sube la
  correlación de revenue de 0,11 a 0,17 y la permutación la deja en <0,001.
  Dentro de la misma industria, la empresa que más habla de resultados de IA
  es la que más crece.
- **Pero tampoco es dinámica.** El efecto fijo de empresa la baja a 0,04. La
  señal separa empresas; no dice que la empresa que empezó a hablar más
  empezó a crecer más.
- **El margen extensivo puro no aporta**: `any_ai` correlaciona negativo con
  el crecimiento (−0,07) por composición temporal (2021-2022: las que no
  hablaban eran las chicas en expansión). Lo que predice es cuánto del
  filing se dedica a resultados de IA, no si se menciona.

## Qué implica para los otros documentos

- `02_...md`: reporta las dos correlaciones que pasan FDR con la advertencia
  de que son transversales. La "señal de washing" de costos no existe
  (r=−0,013).
- `04_...md`: los perfiles por nivel de IA son descriptivos de un tipo de
  empresa; el control de sector-año hay que leerlo como en §1.
- `14_...md`: el washing como desviación temporal entre lo dicho y lo hecho
  no se mide en este cruce; se mide entre canales, dentro de la empresa y el
  ejercicio.

## Limitaciones

- SIC-2 es granularidad gruesa; el residuo dentro de sector-año puede ser
  sub-sector.
- El panel arranca en 2021 con 64% de ceros, así que las primeras
  diferencias mezclan el margen extensivo con el intensivo.
- Los `next_*_yoy` dependen de la cobertura XBRL (capex 87%, R&D 45%).
- Etiquetas de un LLM sin validación humana (`docs/problemas_academicos.md`
  #1); el prefiltro tiene recall 0,96-0,98 y precisión 0,73 en proxy/8-K.
