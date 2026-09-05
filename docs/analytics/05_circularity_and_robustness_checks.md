# Circularidad, placebo y corrección por comparaciones múltiples

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

## 1. Placebo / permutación: ¿el r=0,09 crudo y el r=0,03 dentro-de-sector son distinguibles de ruido?

Método: barajar `behavior_share_revenue_outcome` aleatoriamente entre
empresas-año (2.000 permutaciones), recalcular la correlación con
`next_revenue_yoy` cada vez — cruda y dentro de sector-año — y comparar
el valor observado contra esa distribución nula.

```python
for _ in range(2000):
    xp = rng.permutation(x)
    null_raw.append(np.corrcoef(xp, y)[0,1])
    # ... mismo demeaning por sector-año que en 04_...md, sobre los datos barajados
```

| Especificación | r observado | Media nula | DE nula | IC 95% nulo | p (permutación) |
|---|---|---|---|---|---|
| Cruda | 0,073 | 0,001 | 0,037 | [−0,062, 0,078] | **0,043** |
| Dentro de sector-año | 0,029 | 0,001 | 0,040 | [−0,067, 0,085] | **0,458** |

El r crudo (0,073) queda justo en el borde de lo distinguible de ruido
puro (p=0,043) — apenas. El r dentro-de-sector (0,029), que en
`04_...md` se presentó como "señal débil que sobrevive, atenuada" —
**no es distinguible de una recolocación aleatoria de las etiquetas de
comportamiento** (p=0,46). Confirma con evidencia directa la sospecha
de circularidad: una vez que se remueve el efecto sector-año, lo que
queda de esa correlación específica es ruido, no señal atenuada.
**Corrección a `04_...md`: la frase "lo que queda (r=0,029) es casi
ruido" era la lectura correcta — este chequeo lo confirma
formalmente, no hace falta suavizarla más.**

## 2. Efectos fijos de empresa (within-firm): la prueba MÁS estricta para circularidad — y la que encuentra la señal más fuerte

El demeaning por sector-año (`04_...md`) remueve shocks de sector-año,
pero no la heterogeneidad de nivel-empresa (una empresa puede ser
sistemáticamente más grande, más rentable, o hablar más de IA que el
resto de su sector, de forma constante en el tiempo, y eso seguiría
contaminando la correlación). El control que SÍ elimina eso: mirar
solo la variación DENTRO de la misma empresa a través de los años
— la pregunta pasa de "¿las empresas que hablan más de
`revenue_outcome` crecen más?" a "¿CUANDO una empresa dada habla más
de `revenue_outcome` de lo que habla usualmente, ESA MISMA empresa
crece más de lo que crece usualmente?" — una pregunta mucho más difícil
de responder con un efecto puramente composicional/sectorial.

```python
# (a) primeras diferencias: cambio año a año dentro de cada ticker
c['d_x'] = c.groupby('ticker')['behavior_share_revenue_outcome'].diff()
c['d_y'] = c.groupby('ticker')['next_revenue_yoy'].diff()

# (b) demeaning por empresa (equivalente a efectos fijos de empresa)
c2['x_fe'] = c2['behavior_share_revenue_outcome'] - c2.groupby('ticker')[...].transform('mean')
c2['y_fe'] = c2['next_revenue_yoy'] - c2.groupby('ticker')[...].transform('mean')
```

| Especificación | r | n (obs) | n (empresas) | p |
|---|---|---|---|---|
| Primeras diferencias | **0,100** | 473 | 232 | **0,029** |
| Efectos fijos de empresa (demeaned) | **0,098** | 705 | 232 | **0,010** |

**Esto es más fuerte, no más débil, que la correlación cruda (0,073) y
mucho más fuerte que la de sector-año (0,029, no significativa).**
Dicho en palabras: no hay evidencia de que "las empresas que hablan de
`revenue_outcome` en general crecen más" (eso sí podía ser sector) —
pero SÍ hay evidencia, robusta a dos especificaciones distintas, de que
**cuando una empresa específica intensifica su propio discurso de
`revenue_outcome` por sobre su propio promedio histórico, esa misma
empresa tiende a crecer más de lo que crece usualmente al año
siguiente**. Es el resultado individual más convincente de todo el
cruce contable/mercado — sobrevive exactamente el control que estaba
diseñado para destruir un hallazgo circular, y no lo destruye.

**No es causal** (podría ser que la empresa empiece a hablar de
`revenue_outcome` PORQUE ya sabe internamente que el próximo año va
bien — reverse causality/anticipación, no que el discurso "cause" el
crecimiento), pero descarta la lectura más simple de circularidad
(que todo sea composición sectorial fija).

## 3. Corrección por comparaciones múltiples (FDR, Benjamini-Hochberg) sobre las 11 correlaciones reportadas en `02_...md`/`04_...md`

```python
def corr_p(r, n):
    t = r * np.sqrt(n-2) / np.sqrt(1-r**2)
    return 2*(1 - stats.t.cdf(abs(t), df=n-2))
# BH: ordenar por p, comparar p_(k) contra (k/m)*0.05
```

| Test | r | n | p | Rank | Umbral BH | Sobrevive FDR 5% |
|---|---|---|---|---|---|---|
| `ai_infrastructure` vs `next_capex_yoy` | 0,113 | 563 | **0,0073** | 1 | 0,0045 | **Sí** |
| `revenue_outcome` vs `next_revenue_yoy` (crudo) | 0,093 | 786 | **0,0091** | 2 | 0,0091 | **Sí** |
| `ret_m1_p5` vs `specificity_index` | −0,054 | 1.184 | 0,063 | 3 | 0,0136 | No |
| `ai_infrastructure` vs `next_rd_expense_yoy` | 0,084 | 418 | 0,086 | 4 | 0,0182 | No |
| `car_m1_p5` vs `specificity_index` | −0,049 | 1.184 | 0,092 | 5 | 0,0227 | No |
| `revenue_outcome` vs `next_revenue_yoy` (sector-año) | 0,029 | 803 | 0,412 | 6 | 0,0273 | No |
| `ai_investment` vs `next_capex_yoy` | 0,034 | 563 | 0,421 | 7 | 0,0318 | No |
| `car_m1_p5` vs `promotional_rate` | −0,020 | 1.184 | 0,492 | 8 | 0,0364 | No |
| `cost_outcome` vs `next_sga_expense_yoy` | −0,027 | 419 | 0,582 | 9 | 0,0409 | No |
| `ret_m1_p5` vs `promotional_rate` | −0,013 | 1.184 | 0,655 | 10 | 0,0455 | No |
| `ai_investment` vs `next_rd_expense_yoy` | 0,009 | 418 | 0,854 | 11 | 0,0500 | No |

Solo **2 de 11** correlaciones sobreviven corrección FDR al 5%: el
`revenue_outcome` crudo (ya sabíamos que era la señal principal) y
**`ai_infrastructure` vs. `next_capex_yoy` (r=0,113)** — que en
`04_...md` se reportó textualmente como "sin señal" porque la
magnitud de r parecía chica. **Es un error de lectura que corrijo
aquí: r=0,113 con n=563 SÍ es estadísticamente significativo, incluso
tras controlar por 11 comparaciones simultáneas — de hecho es la
correlación más significativa del conjunto (rank 1).** La conclusión
de `04_...md` ("hablar de infraestructura de IA no predice nada del
capex real") debe leerse como "la relación es débil en magnitud
económica, no que sea estadísticamente inexistente" — hay una relación
real y detectable, solo que pequeña.

## Lectura conjunta: ¿qué tan circular es, entonces?

Con evidencia, no solo con intuición:

- **La preocupación de circularidad era correcta para el resultado
  reportado en `04_...md` como "atenuado pero sobreviviente" (r=0,029
  dentro de sector)** — el placebo confirma que ESE número específico
  es ruido. Esa parte del hallazgo neutro se sostiene, y el chequeo la
  hace más sólida, no la contradice.
- **Pero la circularidad NO explica todo.** El efecto fijo de empresa
  (el control más estricto posible contra "esto es solo composición
  sectorial/de nivel-empresa fijo") encuentra una señal MÁS fuerte
  (r≈0,10, p<0,03) para exactamente la misma relación
  (`revenue_outcome` → crecimiento real). Esto es el opuesto de lo que
  predeciría "todo es circular" — si fuera pura composición fija, el
  efecto fijo de empresa debería haber hecho desaparecer la señal, no
  fortalecerla.
- **Sobre preferir positivo a neutro**: este chequeo, hecho
  honestamente en ambas direcciones, terminó revirtiendo el sesgo
  contrario — encontró que había estado siendo DEMASIADO conservador
  con `ai_infrastructure`→capex (marcado "sin señal" cuando sí lo hay,
  aunque chica) y confirmando (no relajando) el hallazgo neutro de
  `revenue_outcome`-dentro-de-sector. El resultado neto es una imagen
  MÁS matizada, no más positiva ni más negativa: hay una señal real y
  robusta (dentro de empresa, para `revenue_outcome`), una señal real
  pero económicamente chica (`ai_infrastructure`→capex), y una señal
  que resultó ser puro artefacto sectorial una vez puesta a prueba
  (`revenue_outcome` cruzado con sector-año en vez de con la propia
  empresa).

## Qué actualizar en los documentos anteriores

- `04_ratios_factors_and_volatility.md`, sección "talk-vs-walk
  CONTROLANDO por sector": agregar que el r=0,029 no es distinguible
  de ruido (placebo p=0,46) — refuerza la conclusión ya escrita, no la
  cambia.
- `04_...md`, sección de `ai_investment`/`ai_infrastructure`: corregir
  "sin señal" → "señal estadísticamente real pero económicamente
  pequeña" para el par `ai_infrastructure`/`next_capex_yoy`
  específicamente (los otros 3 pares de esa tabla sí quedan sin señal
  tras FDR).
- Nuevo hallazgo a incorporar en la narrativa de tesis: el resultado
  de efectos fijos de empresa (§2) es actualmente el hallazgo
  individual más defendible de todo el cruce contable — sobrevivió el
  control más estricto que se le podía aplicar.

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
