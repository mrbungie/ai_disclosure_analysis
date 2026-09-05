# AI-washing como score continuo: reemplazo del cruce de clusters

Este documento reemplaza la definición operativa de AI-washing de
`06_voice_vs_behavior_clustering.md` ("arquetipo de voz D × cluster de
comportamiento 1"). Esa definición no era medible, y la sección siguiente
explica por qué con números antes de proponer nada.

Producido por `scripts/analytics/washing_score.py`. Determinístico, sin
LLM, sobre `gold_ai_frames`.

## Por qué el cruce de clusters no servía

Los 4 arquetipos de voz se construyen sobre 9 **tasas** cuyo denominador
va de 5 a 500 frames según la empresa, y K-means trata una tasa estimada
con 5 frames como igual de confiable que una estimada con 500. Medido
sobre este corpus:

| Diagnóstico | Valor |
|---|---|
| Tasa promocional del corpus | 9,6% |
| P(cero frames promocionales por azar) con 4 frames | **67%** |
| Ídem con 50 frames | 0,6% |
| Empresas-año con 3-5 frames y tasa promocional exactamente 0 | **84,6%** |
| Ídem con 50+ frames | 1,1% |
| Mediana de frames totales, arquetipo A / B / C / D | 17 / 26 / 36 / 59 |
| Estabilidad de la etiqueta año a año, 3-5 frames | 55,9% |
| Ídem, 50+ frames | 74,2% |

**La escalera A<B<C<D es también una escalera de volumen de texto.** Las
nueve métricas son mayoritariamente ceros estructurales con denominador
chico, y esos ceros empujan a las empresas de poco texto hacia los
rincones extremos del espacio de features, que es donde viven A y C.

El caso que lo deja fuera de discusión: **cuatro empresas con CERO frames
promocionales quedaban etiquetadas "líderes vocales de IA".**

| ticker | frames | promocionales | cuantificados | deployed | arquetipo |
|---|---:|---:|---:|---:|---|
| CMS | 5 | **0** | 0 | 0 | D |
| FE | 6 | **0** | 0 | 1 | D |
| CHD | 8 | **0** | 0 | 3 | D |
| TFC | 9 | **0** | 0 | 4 | D |

Pasa porque D no se define sólo por promoción: también por
`realized_share` alto y `risk_share` bajo. Una empresa con 5 frames que
resultan ser todos afirmaciones en pasado y ninguna de riesgo cae en el
rincón D sin haber dicho una palabra promocional.

Y el grupo de washing resultante mezclaba dos cosas incompatibles: seis
empresas de ≤9 frames cuya etiqueta era artefacto puro, y cinco (AAPL,
ADI, JPM, ALL, ECL) con lenguaje promocional genuino. No era una
categoría.

**Ningún umbral de volumen arregla esto**, y conviene decirlo porque fue
el primer arreglo que se intentó: exigir ≥10 frames/año selecciona a las
tecnológicas grandes, que son justamente el *resto* de D, y elimina por
construcción a las empresas sobre las que trata la hipótesis. Es un
filtro de selección disfrazado de filtro de calidad.

## El modelo

En vez de estimar una tasa por empresa y compararla contra una frontera,
se modelan los **conteos** y se pregunta si el exceso de lenguaje
promocional de una empresa supera lo que explica el muestreo.

1. **Nivel frame**: regresión logística de `rhetoric_promotional` sobre el
   índice de comportamiento de su empresa. Cada frame es una observación,
   así que una empresa con 500 frames pesa 100 veces más que una con 5 —
   el peso que corresponde.
2. **Nivel empresa**: test binomial **exacto** de *k* frames promocionales
   observados contra `Binomial(n, p̂)`, donde p̂ es lo que el modelo predice
   dado el comportamiento de esa empresa.
3. **Benjamini-Hochberg** al 5% sobre las 446 empresas, porque son 446
   tests simultáneos.

Con *n* chico el test simplemente no rechaza. **No hace falta ningún
umbral arbitrario: la potencia estadística se encarga**, y la ausencia de
evidencia queda registrada como ausencia de evidencia en vez de como una
etiqueta.

### Control de la dependencia mecánica

Voz y comportamiento se miden sobre los mismos frames, y la dependencia a
nivel frame es grande:

- P(describe comportamiento | frame promocional) = **92,6%**
- P(describe comportamiento | frame NO promocional) = **54,7%**

Por eso el índice de comportamiento se calcula **sólo sobre los frames no
promocionales** de cada empresa. Eso rompe el lazo sin sesgar el
predictor: sigue midiendo cuánto comportamiento describe la empresa, en la
parte de su texto que no está en discusión.

**Robustez**: con la especificación ingenua (comportamiento sobre todos
los frames) salen 16 empresas en la cola de washing; con la limpia, 22, y
**las 16 están contenidas en las 22**. El resultado no depende de esa
decisión.

El modelo ajustado, especificación limpia:

```
logit(P(promocional)) = -3,78 + 2,63 × índice_comportamiento
```

La pendiente positiva importa: **a más comportamiento descrito, más
lenguaje promocional.** Promoción y sustancia van juntas en general, no
son sustitutos. El washing no es "promocionar en vez de hacer" sino
"promocionar más de lo que incluso esa relación positiva predice".

## Resultados

Sobre 446 empresas con ≥5 frames, FDR 5%:

| | empresas |
|---|---:|
| Habla MÁS de lo que su comportamiento justifica (washing) | **22** |
| Habla MENOS (sustancia callada) | **10** |
| Indistinguibles del modelo | 414 |

### Cola de washing (por exceso de frames promocionales)

| ticker | frames | promo. observados | esperados | tasa obs. | tasa esperada |
|---|---:|---:|---:|---:|---:|
| GOOGL | 561 | 108 | 42,5 | 19,3% | 7,6% |
| ADBE | 486 | 97 | 69,5 | 20,0% | 14,3% |
| MSFT | 595 | 102 | 68,4 | 17,1% | 11,5% |
| PANW | 263 | 72 | 33,5 | 27,4% | 12,7% |
| CRM | 348 | 66 | 36,5 | 19,0% | 10,5% |
| CRWD | 204 | 57 | 26,0 | 27,9% | 12,7% |
| INTU | 237 | 53 | 22,5 | 22,4% | 9,5% |
| AMZN | 436 | 52 | 29,2 | 11,9% | 6,7% |
| CDNS | 180 | 51 | 21,6 | 28,3% | 12,0% |
| WDAY | 291 | 47 | 25,3 | 16,2% | 8,7% |
| IBM | 238 | 43 | 23,9 | 18,1% | 10,0% |
| EFX | 250 | 41 | 25,5 | 16,4% | 10,2% |
| ADP | 216 | 40 | 18,3 | 18,5% | 8,5% |
| IQV | 190 | 39 | 21,5 | 20,5% | 11,3% |
| LUMN | 112 | 25 | 12,0 | 22,3% | 10,7% |

### Cola de sustancia callada (por déficit)

| ticker | frames | promo. observados | esperados | tasa obs. |
|---|---:|---:|---:|---:|
| META | 308 | 13 | 28,9 | 4,2% |
| MSI | 197 | 6 | 21,4 | 3,0% |
| AVGO | 141 | 9 | 22,7 | 6,4% |
| MSCI | 175 | 4 | 15,8 | 2,3% |
| STX | 94 | **0** | 13,7 | 0,0% |
| FTNT | 112 | 4 | 15,7 | 3,6% |
| AXP | 131 | **0** | 10,5 | 0,0% |
| KLAC | 89 | **0** | 7,2 | 0,0% |
| NET | 77 | **0** | 8,1 | 0,0% |
| ICE | 48 | **0** | 6,4 | 0,0% |

Cinco empresas con volumen sustancial de divulgación de IA y **cero
frames promocionales**: STX (94 frames), AXP (131), KLAC (89), NET (77),
ICE (48). Describen despliegue de IA sin una sola afirmación superlativa.

## Cómo leer esto, y qué NO dice

**El resultado se invirtió respecto del cruce de clusters, y la razón es
metodológica, no sustantiva.** El método viejo señalaba empresas
pequeñas y no-tech (CMS, FE, CHD, TFC); este señala grandes tecnológicas
(GOOGL, MSFT, ADBE, AMZN). No es que antes estuviera mirando al grupo
equivocado y ahora al correcto en un sentido sustantivo: es que **sólo se
puede detectar un exceso donde hay suficientes frames para detectarlo.**

La tabla de potencia lo hace explícito:

| frames por empresa | empresas | washing detectados | callada |
|---|---:|---:|---:|
| 5-10 | 76 | 0 | 0 |
| 11-25 | 131 | 2 | 0 |
| 26-50 | 113 | 2 | 1 |
| 51-100 | 72 | 2 | 3 |
| 100+ | 54 | 16 | 6 |

**No se puede concluir que las tecnológicas grandes hagan más
AI-washing.** Se puede concluir que son las únicas empresas sobre las que
este corpus permite afirmar algo. De las 207 empresas con ≤25 frames, el
test no rechaza en ninguna dirección para 205 — y eso es la respuesta
correcta, no una falla: no hay evidencia suficiente.

Lo que sí se sostiene:

- **Entre las empresas medibles, el exceso promocional es real y grande.**
  GOOGL usa lenguaje promocional en 19,3% de sus frames cuando su propio
  perfil de comportamiento predice 7,6% — 2,5x, sobre 561 frames.
- **La cola opuesta existe y es igual de nítida.** Cinco empresas con
  decenas de frames y cero promoción.
- **Promoción y comportamiento son complementarios, no sustitutos**
  (pendiente +2,63). La intuición de "el que habla no hace" es falsa como
  regla general en este corpus; el washing es una desviación de esa
  relación, no la relación misma.

## Limitaciones

- **Voz y comportamiento salen del mismo texto.** Se acota midiendo el
  comportamiento en frames no promocionales, y el resultado es robusto a
  esa decisión, pero no se elimina. Un diseño limpio mediría el
  comportamiento contra una fuente externa —capex, I+D, contrataciones—
  que es lo que intentan `02_...md` y `04_...md` con resultados débiles.
- **Potencia concentrada en empresas de mucho texto.** Ver arriba. Es una
  limitación del corpus, no del estimador.
- **`rhetoric_promotional` es una etiqueta de un LLM** (qwen/qwen3.7-flash,
  ver `docs/classification_model.md`), no una medida validada contra
  anotación humana a escala. Todo el score hereda esa dependencia.
- **Sin dimensión temporal.** El score es pooled por empresa. La versión
  empresa-año es directa (mismo test por año) y no está hecha; permitiría
  preguntar si el exceso promocional de una empresa cambia tras el
  escrutinio de la SEC, que es la pregunta de `01_...md` §SEC 2024.
- **Sin control sectorial.** La cola de washing es casi toda software; la
  de sustancia callada mezcla semis, seguros y redes. Repetir el ajuste
  con efectos fijos de SIC-2 diría si el exceso es sobre el promedio del
  corpus o sobre el de su propia industria.
