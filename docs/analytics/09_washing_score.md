# AI-washing como score continuo: exceso promocional sobre comportamiento y mezcla documental

Reemplaza la definición de `06_voice_vs_behavior_clustering.md` ("arquetipo de
voz D × cluster de comportamiento 1"), que no era medible. Producido por
`scripts/analytics/washing_score.py` y validado por
`scripts/analytics/validate_washing_score.py`. Determinístico, sin LLM.

## Por qué el cruce de clusters no servía

Los 4 arquetipos de voz se construyen sobre 9 **tasas** cuyo denominador va de
5 a 500 frames según la empresa, y K-means trata una tasa estimada con 5 frames
como igual de confiable que una estimada con 500.

| Diagnóstico | Valor |
|---|---|
| Tasa promocional del corpus | 9,4% |
| P(cero frames promocionales por azar) con 4 frames | **67%** |
| Empresas-año con 3-5 frames y tasa promocional exactamente 0 | **84,6%** |
| Ídem con 50+ frames | 1,1% |
| Mediana de frames totales, arquetipo A / B / C / D | 17 / 26 / 36 / 59 |

**La escalera A<B<C<D es también una escalera de volumen de texto.** El caso que
lo deja fuera de discusión: cuatro empresas con CERO frames promocionales (CMS,
FE, CHD, TFC, todas con 5-9 frames) quedaban etiquetadas "líderes vocales de
IA", porque D también se define por `realized_share` alto y `risk_share` bajo.

Ningún umbral de volumen lo arregla: exigir ≥10 frames/año selecciona a las
tecnológicas grandes, que son el *resto* de D, y elimina por construcción a las
empresas sobre las que trata la hipótesis.

## El modelo

En vez de estimar una tasa por empresa y compararla contra una frontera, se
modelan los **conteos** y se pregunta si el exceso promocional de una empresa
supera lo que explican el muestreo, su comportamiento declarado y **la mezcla
de documentos en la que habla**.

1. **Unidad: frame único.** `gold_ai_frames` expande cada texto clasificado a
   una fila por instancia de párrafo, así que un boilerplate repetido entra
   varias veces: 24.328 instancias → **22.643 frames únicos**.
2. **Nivel frame**: logística de `rhetoric_promotional` sobre el índice de
   comportamiento de la empresa **más efectos fijos de formulario**. Cada frame
   es una observación, así que una empresa con 500 frames pesa 100 veces más
   que una con 5.
3. **Nivel empresa**: el conteo nulo es una **Poisson-binomial** (cada frame
   tiene su propia probabilidad según su formulario), corregida por
   **dependencia intra-documento**.
4. **Benjamini-Hochberg** al 5% sobre las 449 empresas, en las dos colas.

Modelo ajustado:

```
logit(P(promocional)) = −4,297 + 2,913 × índice_comportamiento
                        + 0,970 × [DEF 14A] + 0,054 × [8-K]
```

**El coeficiente del proxy es la mitad del efecto que un score sin control de
formulario atribuiría a la empresa.** La DEF 14A tiene 14,7% de frames
promocionales contra 7,1% del 10-K,
así que una empresa cuyo proxy aporta la mitad de sus frames parecía
promocional por composición documental. Sin ese control la cola de washing
tiene 41% de frames de proxy contra 23% del resto del corpus.

La pendiente del comportamiento sigue siendo **positiva y grande**: a más
comportamiento descrito, más lenguaje promocional. Promoción y sustancia van
juntas; el washing es una desviación de esa relación, no la relación misma.

### Dependencia entre frames, y a qué nivel corregirla

El test binomial supone frames independientes. No lo son: los de un mismo
filing comparten párrafo, sección y decisiones de redacción. Se estima la
correlación intra-documento con un ANOVA de una vía sobre los residuos del
modelo (**ICC = 0,043**, mediana de 5,3 frames por filing → efecto de diseño
1,18) y se testea con una beta-binomial equivalente.

**El nivel del clustering es una decisión sustantiva.** Estimar la dispersión a
nivel EMPRESA (`--dispersion firm`) trata la heterogeneidad entre empresas como
ruido — y esa heterogeneidad es el fenómeno que se quiere medir. Con esa
especificación el test **no marca a nadie**, en ninguna de las dos colas. No
es que no haya washing: es que ese estimador se lo comió.

### Control de la dependencia mecánica voz-comportamiento

P(describe comportamiento | frame promocional) = 92,6%, contra 54,7% en los no
promocionales. Por eso el índice de comportamiento se calcula **sólo sobre los
frames no promocionales** de cada empresa: rompe el lazo sin sesgar el
predictor.

## Resultados

449 empresas con ≥5 frames (23.690 frames únicos de 10-K, DEF 14A y 8-K,
población del prefiltro v2 con umbral 0,17), FDR 5%: **8 en la cola de
washing, 3 en la de sustancia callada, 438 indistinguibles.** Coeficiente de
comportamiento 3,15, DEF 14A +0,88, ICC de documento 0,036.

| ticker | frames | promo. obs. | esperados | tasa obs. | tasa esperada | z |
|---|---:|---:|---:|---:|---:|---:|
| GOOGL | 497 | 98 | 46,4 | 19,7% | 9,3% | 8,0 |
| CRWD | 172 | 48 | 18,0 | 27,9% | 10,5% | 7,5 |
| PANW | 236 | 71 | 35,4 | 30,1% | 15,0% | 6,5 |
| CDNS | 170 | 50 | 22,2 | 29,4% | 13,1% | 6,3 |
| INTU | 229 | 51 | 22,7 | 22,3% | 9,9% | 6,2 |
| ADP | 200 | 37 | 15,4 | 18,5% | 7,7% | 5,7 |
| CRM | 340 | 66 | 35,9 | 19,4% | 10,6% | 5,3 |
| YUM | 38 | 9 | 2,0 | 23,7% | 5,2% | 5,1 |

Sustancia callada: **MSI** (5 promocionales de 175, esperados 21,6), **MSCI**
(4 de 187, esperados 20,0) y **STX** (0 de 73, esperados 10,0).

**Con la especificación ingenua (instancias, sin control de formulario, sin
dispersión) serían 23 y 9.** Salen AAPL, ADBE, AMZN, IBM, EFX, IQV, LUMN, SNPS,
MSFT, NVDA, QCOM, WDAY, CHRW, JCI, ACN — casi todas por composición
documental — y de la cola callada META, AXP, KLAC, NET, SQ y FTNT.

### Potencia

| frames por empresa | empresas | washing | callada |
|---|---:|---:|---:|
| 5-10 | 77 | 0 | 0 |
| 11-25 | 133 | 0 | 0 |
| 26-50 | 114 | 1 | 0 |
| 51-100 | 70 | 0 | 1 |
| 100+ | 55 | 7 | 2 |

**No se puede concluir que las tecnológicas grandes hagan más AI-washing.** Se
puede concluir que son las únicas empresas sobre las que este corpus permite
afirmar algo. De 210 empresas con ≤25 frames, el test no rechaza en ninguna.

## Validación

`validate_washing_score.py`, cinco chequeos. No hay gold standard de "esta
empresa hace washing", así que se valida como cualquier instrumento sin
criterio externo: calibración, confiabilidad, persistencia y sensibilidad.

### 1. Placebo — ¿está bien calibrado?

Permutando la etiqueta promocional **dentro de cada formulario** (preserva la
tasa de cada forma y la mezcla documental de cada empresa, destruye sólo la
asociación empresa-retórica), 5 corridas:

**0 falsos positivos en las 5, en ambas colas**, contra 8 y 3 observados. El
test no está inflando: si acaso, es conservador.

### 2. Split-half — ¿es un rasgo de la empresa o ruido?

Partiendo los frames de cada empresa en dos mitades al azar (268 empresas con
≥10 frames en ambas): **Spearman(z) = 0,453** (p=5e-15), solapamiento del decil
superior **54%**.

Hay señal real y estable, pero **moderada**: la mitad del ordenamiento no se
reproduce con otra mitad de los mismos datos. El score ordena bien los
extremos y mal el medio.

### 3. Persistencia — filings ≤2023 contra ≥2024

98 empresas con ≥10 frames en ambas épocas: **Spearman(z) = 0,370**
(p=2e-04), solapamiento del decil superior 33%. **Una sola empresa (CRWD)
queda marcada en el pool y en las dos épocas por separado.**

Es la misma prueba que hundió la definición de clusters, y este score la pasa
mejor pero no la pasa con holgura: el exceso promocional es persistente como
ordenamiento, no como etiqueta binaria.

### 4. Sensibilidad de especificación

| unidad | controles | dispersión | washing | callada | Jaccard vs. referencia |
|---|---|---|---:|---:|---:|
| único | comportamiento+forma | documento | **8** | **3** | — |
| único | comportamiento+forma | ninguna | 15 | 4 | 53% |
| único | comportamiento | documento | 7 | 1 | 88% |
| único | comportamiento+forma+sector | documento | 3 | 5 | 38% |
| instancia | comportamiento | ninguna (v1) | 23 | 9 | 35% |

**Sólo tres empresas están en la cola bajo TODAS las especificaciones: CDNS,
CRWD y PANW.** Ese es el resultado honesto: hay tres casos que no dependen de
cómo se calcule, y cinco más que sí.

Agregar efectos fijos de sector deja 3 — la cola es casi toda software, así que
controlar por industria consume casi toda la variación. Cuál de las dos es la
pregunta correcta ("¿habla más que el corpus?" vs. "¿habla más que su propia
industria?") es una decisión de tesis, no técnica.

### 5. Criterio externo

Los únicos casos con evidencia independiente son las cartas de comentario de la
SEC (`01_...md`):

| ticker | percentil | z | promocionales | marcado | caso |
|---|---:|---:|---|---|---|
| WELL | 66,5 | −0,42 | — | no | carta SEC sobre disclosure de IA (abril 2025) |
| ANET | 84,6 | +0,96 | — | no | falso positivo (segmentos) |
| HPE | 29,0 | −1,11 | — | no | falso positivo (segmentos) |
| NVDA | 97,5 | +4,09 | — | no | falso positivo (ingresos) |

(La validación guarda percentil y z; los conteos de promocionales por caso
salen de `firm_washing_score.parquet`.)

**Welltower —el único caso real— no lo detecta el score, y no es un bug del
estimador sino una diferencia de constructo.** La SEC no le objetó su 10-K por
promocional: le objetó lo contrario, que la empresa dijera "industry-leading" y
"competitive advantage" en el earnings call y el press release **sin respaldo
proporcional en el 10-K**. Este score mide exceso promocional DENTRO del
filing; el mecanismo que persigue el regulador es una BRECHA entre canales.

Es la limitación más importante del instrumento y marca el próximo paso
concreto: medir el mismo score sobre transcripciones de earnings calls y
comparar el exceso de cada empresa entre canales. El corpus de earnings calls
ya está clasificado (16.270 frames de 403 empresas,
`scripts/analytics/earnings_calls_analysis.py`) y el diseño por canal está en
`14_brecha_entre_canales.md`: brecha promocional call − filing de +10,7 p.p.,
ortogonal a este score (Spearman −0,06).

## Limitaciones

- **Voz y comportamiento salen del mismo texto.** Se acota midiendo el
  comportamiento en frames no promocionales; no se elimina. Un diseño limpio lo
  mediría contra capex/I+D/contrataciones — lo que intentan `02_...md` y
  `04_...md`, hoy sin ninguna correlación que sobreviva FDR
  (`10_builders_y_recalculo.md`).
- **Potencia concentrada en empresas de mucho texto.** Es limitación del
  corpus, no del estimador, y la tabla de potencia la deja explícita.
- **`rhetoric_promotional` es una etiqueta de un LLM** (`qwen3.7-flash`), sin
  validación contra anotación humana a escala. Todo el score hereda esa
  dependencia. Es el hueco de medición más grande del proyecto. Lo único
  medido hasta ahora es acuerdo ENTRE LLMs en la etapa anterior (el prefiltro:
  κ=0,87 entre gemini y qwen, `prefilter_evaluation.md` §8.15) — la extracción
  de frames no tiene ni eso.
- **El prefiltro mide peor en DEF 14A que en 10-K** (con v2: F1 0,821 vs
  0,961; precisión 0,73 en el holdout de proxy/8-K — §8.16), y el score usa
  frames de los dos. El
  control de formulario absorbe la diferencia de TASA promocional entre
  formularios, no la diferencia de error de medición.
- **Sin dimensión temporal en el score publicado**: es pooled por empresa. La
  versión empresa-año es el mismo test por año y permitiría preguntar si el
  exceso cambia tras el escrutinio de la SEC.
- **La cola depende de si se controla por sector.** Con SIC-2 quedan 3
  empresas. Hay que elegir y declarar cuál es la pregunta.
