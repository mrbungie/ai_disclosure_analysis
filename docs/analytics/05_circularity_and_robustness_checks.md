# Circularidad, placebo y corrección por comparaciones múltiples

> **Recalculado con builders versionados (ver `10_builders_y_recalculo.md`).**
> El lado contable/mercado ya no viene de un script perdido: lo producen
> `build_firm_financials.py`, `build_market_factors.py` y `build_roic_wacc.py`
> (`make analytics`). Con mejor cobertura XBRL y el ERP corregido, varias
> cifras de este documento se movieron — las tablas de abajo son las de la
> corrida anterior; los deltas están listados en `10_...md`.

> **Recalculado 2026-09-05 con DEF 14A y 8-K.** Placebo, efectos fijos y
> FDR re-corridos sobre el panel ampliado (1.363 empresas-año). Las
> conclusiones cualitativas se mantienen; cambia cuáles correlaciones
> sobreviven al FDR.


Auditoría pedida explícitamente después de discutir qué tan circular
es el diseño de `02_market_accounting_crosscheck.md` /
`04_ratios_factors_and_volatility.md` (el arquetipo se construye a
partir de intensidad/especificidad del discurso de IA, que está casi
mecánicamente correlacionada con pertenecer a un sector tech — así que
"D crece más" podía ser casi tautológico) y el riesgo de sobre-vender
un hallazgo "interesante" (positivo o neutro) sin someterlo al mismo
escrutinio en ambas direcciones. Tres chequeos, todos sobre
`data/processed/clusters/firm_year_master_v2.parquet`.

**Resultado headline: uno de los chequeos más rigurosos (efectos fijos
de empresa) encuentra una señal MÁS fuerte que la reportada en
`02_...md`/`04_...md`, no más débil — y un hallazgo que había marcado
"sin señal" en `04_...md` resulta ser el único que sobrevive corrección
FDR. Este documento corrige ambos.**

## 1. Placebo / permutación: ¿el r crudo y el r dentro-de-sector son distinguibles de ruido?

2.000 permutaciones del predictor contra el outcome fijo, semilla 42:

| Especificación | r observado | p (permutación) | n |
|---|---|---|---|
| `revenue_outcome` ~ `next_revenue_yoy`, crudo | 0,071 | **0,035** | 887 |
| Ídem, dentro de sector-año | 0,033 | 0,321 | 887 |

Mismo veredicto que en la versión anterior: **la correlación cruda es
apenas distinguible de ruido (p≈0,035, al filo de 0,05) y la de dentro de
sector-año no lo es en absoluto (p=0,32).** Una vez que se controla por
sector, "hablar de revenue impulsado por IA" no predice el crecimiento
de revenue mejor que una asignación al azar de las etiquetas.

## 2. Efectos fijos de empresa (within-firm): la prueba más estricta para circularidad

Pregunta: ¿cuando UNA MISMA empresa habla más de revenue de IA que su
propio promedio, crece más que su propio promedio? Esto elimina de raíz
cualquier confusor fijo por empresa (sector, modelo de negocio, tamaño,
estilo de redacción del filing).

| Especificación | r | n |
|---|---|---|
| (a) Primeras diferencias (Δ año a año dentro de cada ticker) | **0,078** | 531 |
| (b) Demeaning por empresa (equivalente a efectos fijos) | **0,064** | 887 |

Ambas siguen siendo positivas y **mayores que la correlación dentro de
sector-año (0,033)**. Es el resultado más interesante de este documento y
sobrevivió al cambio de población: la señal within-firm es más fuerte que
la between-firm-dentro-de-sector.

Interpretación: la parte de la correlación cruda que es genuina parece
ser **temporal dentro de la empresa** (cuando una empresa empieza a
hablar más de resultados de IA, algo cambia de verdad en su trayectoria)
más que transversal entre empresas. Las magnitudes bajaron un poco
respecto de la versión anterior, pero el orden relativo —within-firm >
within-sector— se mantiene, que es lo que sostiene la lectura.

Caveat que no cambia: r≈0,06-0,08 sigue siendo una correlación débil.
"No es puramente circular" no es lo mismo que "es fuerte".

## 3. Corrección por comparaciones múltiples (FDR, Benjamini-Hochberg) sobre las 11 correlaciones reportadas

| # | Par | r | p | Umbral BH | ¿Pasa FDR 5%? |
|---|---|---|---|---|---|
| 1 | `ai_infrastructure` ~ `next_capex_yoy` | 0,114 | 0,0045 | 0,0045 | Empate exacto |
| 2 | `revenue_outcome` ~ `next_revenue_yoy` | 0,091 | 0,0070 | 0,0091 | **Sí** |
| 3 | `n_frames` ~ `ret_m1_p5` | −0,054 | 0,0520 | 0,0136 | No |
| 4 | `promotional_rate` ~ `car_m1_p5` | −0,047 | 0,0919 | 0,0182 | No |
| 5 | `specificity_index` ~ `ret_m1_p5` | −0,030 | 0,2773 | 0,0227 | No |
| 6 | `specificity_index` ~ `car_m1_p5` | −0,028 | 0,3147 | 0,0273 | No |
| 7 | `ai_investment` ~ `next_rd_expense_yoy` | 0,036 | 0,4427 | 0,0318 | No |
| 8 | `promotional_rate` ~ `ret_m1_p5` | −0,015 | 0,5915 | 0,0364 | No |
| 9 | `ai_investment` ~ `next_capex_yoy` | 0,016 | 0,6983 | 0,0409 | No |
| 10 | `ai_infrastructure` ~ `next_rd_expense_yoy` | 0,014 | 0,7683 | 0,0455 | No |
| 11 | `cost_outcome` ~ `next_sga_expense_yoy` | 0,011 | 0,8144 | 0,0500 | No |

**Entre 1 y 2 de 11 sobreviven al FDR**, y el matiz importa:
`ai_infrastructure` ~ `capex` cae exactamente sobre su umbral
(p=0,00450 contra 0,00450), o sea justo en el borde de la decisión.
`revenue_outcome` ~ `next_revenue_yoy` pasa con algo más de aire
(p=0,0070 contra 0,0091).

Ninguna de las dos debería reportarse como "establecida". Lo defendible
es: **de once correlaciones testeadas en todo el cruce contable/mercado,
como mucho dos se distinguen de lo esperable por azar múltiple, y ambas
están cerca del límite.**

`cost_outcome` ~ `next_sga_expense_yoy`, la "señal de washing" de la
primera versión, queda última con p=0,81 y r=+0,011. Ver `02_...md` §3.

Cambio respecto de la versión anterior: `ai_infrastructure` ~
`next_rd_expense_yoy` pasó de r=0,084 a r=0,014 y se desplomó al puesto
10 de 11. Su contraparte con capex, en cambio, se mantuvo (0,113 →
0,114). La relación "infraestructura declarada → capex real" es estable;
"infraestructura declarada → I+D real" no existía.

## Lectura conjunta: ¿qué tan circular es, entonces?

Con evidencia, no sólo con intuición:

- **La preocupación de circularidad era correcta para el resultado
  reportado en `04_...md` como "atenuado pero sobreviviente"** (r=0,033
  dentro de sector-año) — el placebo confirma que ESE número específico
  es ruido (p=0,32). El chequeo hace más sólido al hallazgo neutro, no
  lo contradice.
- **Pero la circularidad NO explica todo.** El efecto fijo de empresa —el
  control más estricto posible contra "esto es sólo composición sectorial
  o de nivel-empresa fijo"— encuentra una señal MÁS fuerte (r=0,064-0,078)
  que la de dentro-de-sector (0,033) para exactamente la misma relación.
  Es el opuesto de lo que predeciría "todo es circular": si fuera pura
  composición fija, el efecto fijo debería haber hecho desaparecer la
  señal, no fortalecerla.
- **La señal más robusta del cruce contable sigue siendo
  `ai_infrastructure` → `capex`** (r=0,114, p=0,0045, exactamente sobre
  el umbral FDR). Chica en magnitud y en el borde de la significancia,
  pero es la única que se sostuvo entre dos poblaciones distintas.

Lo que agrega la actualización con DEF 14A y 8-K: **este documento es el
que mejor aguantó el cambio de población.** Los tres chequeos
—permutación, efectos fijos, FDR— dan las mismas conclusiones
cualitativas con números levemente distintos. Y explica por qué otros
documentos no aguantaron: los chequeos de acá se corren sobre el panel
completo (n=531 a 1.305), no sobre subgrupos de 4-10 empresas.

## Qué actualizar en los documentos anteriores

Estado tras la actualización de 2026-09-05 (ya aplicado en cada doc):

- `02_...md` §3: la "señal más cercana a washing" (`cost_outcome` ~
  SG&A) pasó de r=−0,027 a r=+0,011 y quedó última en el ranking FDR
  (p=0,81). **Retirada como hallazgo.**
- `04_...md`, talk-vs-walk dentro de sector: el r=0,033 no es
  distinguible de ruido (placebo p=0,32) — refuerza lo ya escrito.
- `04_...md`, `ai_investment`/`ai_infrastructure`: mantener "señal real
  pero económicamente chica" sólo para `ai_infrastructure`/`capex`. El par
  `ai_infrastructure`/`R&D` se desplomó de r=0,084 a r=0,014 y ya no es
  candidato a nada.
- `07_...md` y `08_...md`: sus hallazgos titulares (washing destruye
  valor, sustancia callada crea 3-6x más) **no sobrevivieron** al cambio
  de población. Ver esos documentos.
- Hallazgo más defendible de todo el cruce contable, sin cambios: el
  resultado de efectos fijos de empresa (§2). Sobrevivió el control más
  estricto que se le podía aplicar, y sobrevivió también un cambio de
  muestra.

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
