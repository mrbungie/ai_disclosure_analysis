# Voz × comportamiento como factores, no como cruce de clusters

Producido por `scripts/analytics/voice_behavior_factors.py` sobre
`gold_ai_frames` (corpus final: 10-K, DEF 14A y 8-K, prefiltro v2 con umbral
0,17). Reemplaza el cruce de clusters "arquetipo de voz × cluster de
comportamiento" que este documento contenía: los arquetipos A/B/C/D no eran
reproducibles (`11_segmentacion.md`) y la matriz 2D dependía de denominadores
de 5 a 500 frames. La segmentación vigente es `11_...md`, la grilla
`12_...md`, el score de washing `09_...md`.


`scripts/analytics/voice_behavior_factors.py` reemplaza la matriz 2D de
etiquetas por dos bloques de factores sobre tasas con encogimiento
empírico-Bayes, cruzados con **correlación canónica** — que es la forma de
contestar con un número la premisa de este documento: ¿cuánto comparten "cómo
habla" y "qué dice que hace"?

### Cómo se mide la conducta importa más que con qué se la agrupa

Antes de cruzar nada: el bloque de conducta se agregaba como **tasas** (fracción
de frames de la empresa con cada concepto), y eso destruía la señal.
Confiabilidad split-half (Spearman-Brown) de la MISMA información según cómo se
la resuma (`scripts/analytics/behavior_block_eval.py`):

| representación | confiabilidad mediana |
|---|---:|
| tasa | 0,528 |
| presencia (¿lo menciona alguna vez?) | 0,648 |
| **log-conteo** | **0,845** |

`ai_infrastructure` pasa de 0,53 a 0,85, `proprietary_ai` de 0,43 a 0,83,
`customer_outcome` de 0,36 a 0,77. Una tasa es una forma pésima de resumir un
concepto que aparece en el 2% de los frames: tres menciones sobre 300 frames y
una sobre 100 dan la misma tasa con evidencia muy distinta. **La conducta estaba
medida; la agregación la borraba.**

El conteo crudo, en cambio, mide en buena parte cuánto habla la empresa — el
volumen explica 78-86% del log-conteo de `deployed` y de los dominios, pero sólo
15-46% de los conceptos raros. Por eso el bloque final son **log-conteos
residualizados contra `log(n_frames)`**: "cuánta conducta describe más de la que
su volumen predice".

### Con esa medición, voz y conducta SÍ se separan

| bloque de conducta | correlación canónica con la voz | varianza compartida |
|---|---:|---:|
| tasas (lo anterior) | 0,908 | 82% |
| log-conteos crudos | 0,892 | 80% |
| **log-conteos residualizados por volumen** | **0,770** | **59%** |

(bootstrap remuestreando frames: 0,72-0,76). Y el bloque pasa de tener un solo
factor —donde todo cargaba junto porque todo medía volumen— a **tres
interpretables**: producto vs. interno, construcción de capacidad, y despliegue
vs. inversión temprana.

El comportamiento explica ahora **34,1%** de la varianza del eje de voz, no
55,7%: hay mucho más residuo con el que trabajar, que es exactamente donde vive
la pregunta de AI-washing.

### El eje compartido (con la medición corregida)

Qué es lo que sí comparten:

| lado "riesgo hipotético" | lado "despliegue afirmado" |
|---|---|
| `risk_share` +0,83 | `promotional_rate` −0,58 |
| `hypothetical_share` +0,52 | `quantified_rate` −0,42 |
| `gov_share` +0,40 | `realized_share` −0,40 |
| | `behavior_share_deployed` −0,81 |
| | `behavior_share_productivity_outcome` −0,69 |

**La lectura corregida**: hay un eje compartido fuerte —riesgo/gobernanza
hipotéticos en un extremo, despliegue con resultados y orientación a producto en
el otro— pero **no agota la variación**. Con la medición vieja (tasas) los dos
bloques compartían 82% y parecía que la matriz voz×conducta no tenía contenido;
con log-conteos residualizados comparten 59%, y el 41% restante es la separación
que este documento venía buscando.

### El residuo es lo que queda para "washing"

Regresando el eje de voz sobre TODO el bloque de comportamiento, el residuo
—"habla distinto de lo que su comportamiento declarado predice"— es una medida
continua y corregida por confiabilidad, en vez de un cruce de dos etiquetas de
cluster con denominadores de 5 a 500 frames.

Cuadrantes (mediana de cada eje):

| cuadrante | empresas | frames (mediana) |
|---|---:|---:|
| voz y conducta altas | 139 | 56 |
| voz y conducta bajas | 139 | 25 |
| **voz alta, conducta baja** | 85 | 43 |
| **conducta alta, voz baja** | 85 | 16 |

**Signo del residuo**: el eje de voz va de riesgo/hipotético (+) a
despliegue afirmado (−), así que el residuo se reporta con el signo dado vuelta.
**Positivo = habla en registro de despliegue más de lo que su conducta descrita
predice**, que es la dirección de AI-washing. Negativo = conducta por delante
del discurso.

Residuo positivo (dirección de washing): AMZN, ALLE, USB, ANSS, IQV, ADP, APH,
RSG, PANW, HON. Residuo negativo (describen conducta y hablan poco, o hablan en
registro de riesgo): DLTR, DPZ, ETN, VTR, KMI, SPG, VRTX, WEC, LW, ROST —
retail, utilities y REITs, que mencionan IA sobre todo como riesgo.

Es la misma pregunta que responde `09_washing_score.md` con conteos y un test
exacto; esta versión no tiene potencia estadística por empresa, pero sí ordena a
las 457 en una escala continua en vez de marcar 8. Las dos deberían leerse
juntas: el score dice **dónde hay evidencia**, el residuo dice **dónde mirar**.

Salida: `data/processed/clusters/firm_voice_behavior_factors.parquet`.
