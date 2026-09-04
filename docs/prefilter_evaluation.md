# Evaluación del prefiltro: registro de evidencia

Todas las mediciones hechas sobre el corpus real, con el número de casos y las
advertencias de cada una. Sirve para no repetir experimentos ya hechos y para
poder citar de dónde sale cada decisión.

Corpus de referencia: **3.281.038 párrafos** (`us` 10-K 1.544.496 / 10-Q
1.736.542), 469.231.281 tokens. Modelo `BAAI/bge-m3`, 1024 dims.

**Base de cálculo: 3.016.097 párrafos puntuables** (`paragraphs.is_scorable`).
Se descartan 264.941 filas (8,1%) que la extracción deja como párrafo sin
contenido: viñetas sueltas, espacios de ancho cero, guiones. El texto más
"frecuente" del corpus es un `•` repetido 23.497 veces, y hay 139.157 filas de
un solo carácter. Promedian 10 caracteres contra 472 los puntuables.

No se borran, se marcan: `paragraph_index` tiene que seguir correspondiendo al
texto original o se rompen las llaves ya escritas en embeddings, scores y golden
set. Tampoco hace falta re-scorear — el prefiltro ya las manejaba bien (score
semántico máximo 0,5483, ninguna supera 0,6); lo que corregían era el
denominador. El golden set queda casi intacto: sólo 6 de 6.038 etiquetas cayeron
sobre filas no puntuables, ninguna positiva.

Los porcentajes de secciones anteriores calculados sobre 3.281.038 quedan ~9%
bajos. Los corregidos: léxico fuerte marca 8.347 (**0,277%**), el híbrido 13.829
(**0,459%**).

Fecha de las corridas: 2026-09-02.

---

## 0. Estado final y próximos pasos

### 0.1 Configuración vigente

Prefiltro de **dos etapas con trabajos distintos**, que es la conclusión
principal de todo el registro:

1. **Compuerta léxica** — términos con límite de palabra
   (`(^|[^a-z0-9])término([^a-z0-9]|$)`) sobre minúsculas, incluyendo las
   siglas (`ai`, `ml`, `llm`, `agi`, `nlp`). Hace el trabajo de **no perder
   nada**: recall 98,2%.
2. **Margen semántico** (`max_semantic_score − negative_similarity`) — **no**
   expande recall, y no puede: los párrafos que divulgan IA sin vocabulario de
   IA casi no existen en este corpus, porque las empresas que hablan de IA
   escriben "AI". Su trabajo es **precisión dentro** de lo que la compuerta ya
   tomó, separando disclosure sustantivo de mención al pasar (AUC 0,671; margen
   medio 0,0328 en `substantive` contra −0,0016 en `incidental`).

Veníamos usándola al revés todo el tiempo.

### 0.2 Números finales

Sobre 3.016.097 párrafos puntuables y 6.032 etiquetas del golden set. **Sin
ponderar** — los pesos no son confiables hasta terminar el etiquetado (§4.2),
así que estas cifras describen el estrato muestreado, no el corpus.

| scorer | prec | recall | F1 | filas marcadas | % corpus |
|---|---|---|---|---|---|
| **logit sobre las 11 señales** | **0,736** | **0,946** | **0,828** | **15.282** | **0,507%** |
| solo compuerta léxica | 0,693 | 0,982 | 0,813 | 16.800 | 0,557% |
| léxico O (débil Y margen ≥ 0,06) | 0,684 | 0,983 | 0,807 | 17.294 | 0,573% |
| léxico, podando 10% de margen más bajo | 0,714 | 0,910 | 0,800 | 15.306 | 0,507% |
| léxico, podando 20% de margen más bajo | 0,739 | 0,838 | 0,786 | 13.613 | 0,451% |
| léxico, podando 30% de margen más bajo | 0,754 | 0,748 | 0,751 | 12.081 | 0,401% |

**Mejor clasificador: la regresión logística sobre las 11 señales** que el
pipeline ya calcula (7 scores por categoría + `negative_similarity` +
`semantic_margin` + los dos flags léxicos), validada out-of-fold con
`GroupKFold` agrupado por filing. F1 0,828 contra 0,813 de la compuerta sola, y
domina a las reglas de poda: mejor precisión Y mejor recall que podar 10% o 20%,
marcando el mismo volumen.

**Si preferís una regla explicable**, la compuerta léxica sola cuesta 1,5 puntos
de F1 y marca 16.800 párrafos con recall 98,2%. Es una decisión de tesis, no
técnica: coeficientes aprendidos contra un umbral que se puede escribir en una
oración.

### 0.3 Próximos pasos, en orden de valor

1. **Terminar el golden set** (3.862 pendientes: 3.472 sin intentar + 390
   fallidos por créditos). Es lo que desbloquea todo lo demás — sin
   `stage3_random` etiquetado no hay reponderación, y sin reponderación ninguna
   cifra proyectada al corpus es citable. Con el orden de etiquetado ya
   corregido a aleatorio, un corte parcial deja submuestra válida.
2. **Rehacer §5–§8 con la configuración actual.** Esas tablas se midieron contra
   la compuerta léxica rota y los anchors viejos. Las conclusiones cualitativas
   sobreviven; los números no.
3. **Decidir regla vs modelo** y congelarlo. Afecta cómo se describe el
   prefiltro en la tesis.
4. **Revisar la poda por margen como etapa separada.** Podar el 20% más bajo
   sube precisión de 0,693 a 0,739 perdiendo 14 puntos de recall. Si aguas abajo
   hay un clasificador LLM caro, ese intercambio puede convenir; si no, no.
5. **Extraer más secciones del filing.** Hoy hay 3 del 10-K (items 1, 1A, 7) y 2
   del 10-Q (1A, 2). Es la única palanca que sube el numerador de verdad — lo
   que se diga de IA en Item 5, 7A o en las notas no está en el corpus.
6. **Las otras cabezas de bge-m3** (`sparse_vecs`, `colbert_vecs`), sólo si 1–5
   ya están hechos. Requiere re-embeber y ColBERT no es viable a escala de
   corpus por almacenamiento; su uso natural es como reranker sobre lo que
   sobrevive al prefiltro.

**Lo que NO vale la pena reintentar**, ya medido y descartado en §7: BM25 con
vocabulario a mano (+0,003 F1), centrado del espacio de embeddings (−0,038), y
TF-IDF con vocabulario del corpus (+0,002).

---

## 1. El problema que justifica todo lo demás

Evaluar el prefiltro con una etiqueta léxica ("¿el párrafo dice *artificial
intelligence*?") es circular:

- Un scoring **híbrido** da precisión ~100% por construcción, porque el léxico
  es a la vez entrada y verdad.
- Un scoring **semántico puro** queda castigado justo donde aporta: sobre
  940.032 párrafos medidos, sólo el **0,38%** contiene un término fuerte, así
  que la etiqueta léxica declara negativo todo lo demás por definición.

De ahí el golden set con juez LLM (`docs/golden_set_sampling.md`).

---

## 2. Costo de cómputo del embedding

Autotune por presupuesto de tokens, medido en una RTX 5090 (31,4 GiB):

| precisión | B/token medidos | tok/s sostenidos | corpus completo | pico de memoria |
|---|---|---|---|---|
| fp32 | 42.344 | 73.600 | 1 h 46 min | 3,7 GiB |
| fp16 | 21.998 | 189.000–205.000 | **45 min** | 3,7 GiB |

El padding es despreciable: 469.231.281 tokens sin padding contra 469.291.118
con padding bajo el plan real de batches — **factor 1,000** en 30.142 batches,
porque el empaquetado ordena por largo antes de agrupar.

Llenar la GPU no ayuda: el throughput satura en ~10k tokens/batch (85k tok/s a
9.344 tokens contra 81k a 145k tokens en fp32). Por eso el autotune corta en el
batch más chico que ya satura, y usa 3,7 de 31,4 GiB.

Salida: 60 partes, **6,0 GB** en `fixed_size_list<float16, 1024>` sin comprimir.
fp32 serían 13,5 GB para registrar precisión que el encoder en fp16 nunca tuvo.

### 2.1 Precisión numérica (6.000 párrafos reales)

| config | tok/s | speedup | pico | cos mín vs fp32 | \|Δscore\| p99 | cambia argmax |
|---|---|---|---|---|---|---|
| fp32 | 59.590 | 1,00x | 2,95 G | 1,000000 | 0 | 0% |
| fp32 + TF32 | 76.097 | 1,28x | 5,22 G | 0,999872 | 0,000177 | 0,10% |
| **fp16** | 172.664 | **2,90x** | 2,18 G | 0,999775 | 0,000521 | 0,28% |
| bf16 | 173.878 | 2,92x | 2,18 G | 0,999229 | 0,003408 | 2,62% |

fp16 es la elección: misma velocidad que bf16 con mucha mejor fidelidad, porque
bf16 cambia bits de mantisa por rango que este modelo no necesita.

### 2.2 int8 ONNX: descartado con número

`gpahal/bge-m3-onnx-int8` en CUDA rinde **1,6k tok/s contra 186k de fp16
(~100x más lento)**. Fue exportado con `AutoQuantizationConfig.avx512_vnni`, un
target de CPU x86, y su `requirements.txt` pide `onnxruntime` (build de CPU).
ONNX Runtime inserta **606 nodos Memcpy** porque los ops int8 no tienen kernel
CUDA: cada capa copia GPU→CPU→GPU. La GPU queda al 34%.

Para que int8 rindiera en esta GPU haría falta exportarlo con calibración
estática y correrlo por TensorRT.

---

## 3. Reescritura de los anchors (940.032 párrafos)

Los anchors originales eran **descripciones de la categoría** ("The company
relies on third-party artificial intelligence providers or models."). Sus
vecinos más cercanos en el corpus real eran boilerplate de dependencia de
proveedores: *"We are dependent on third-party suppliers."* quedaba en el
**puesto 12** de todo el corpus, por encima de disclosure de IA real.

Reescritos como **prosa de filing**, mismos vectores, sólo cambian los anchors:

| anchors | score | AUC | prec@1k | prec@10k | rank del falso positivo |
|---|---|---|---|---|---|
| viejos | crudo | 0,9459 | 37,0% | 15,3% | **12** |
| viejos | pos−neg | 0,9339 | 36,5% | 12,9% | 1.145 |
| **nuevos** | crudo | 0,9408 | **47,2%** | 14,5% | 24.396 |
| nuevos | pos−neg | 0,9212 | 41,1% | 15,4% | 938.529 |

Precisión en el top-1000: **37,0% → 47,2%**.

Advertencias: la etiqueta acá es el proxy léxico (§1), que castiga al semántico,
así que el 47,2% es un piso. Y el rank 938.529 es en parte tautológico: la frase
exacta se agregó como anchor negativo.

---

## 4. Golden set

`gemini-3.8-flash`, prompt v1, muestreo de `docs/golden_set_sampling.md`.

**Estado: incompleto.** De 9.900 muestreados hay **6.038 etiquetados** y 390
fallidos; la corrida se cortó con `429 — "Your prepayment credits are depleted"`.
Cubre `stage1_tech_oversample` (3.950, 1.172 positivos) y parte de
`stage2_max_variation` (2.088, 335 positivos). **`stage3_random` quedó sin
etiquetar**, y es el estrato que ancla la reponderación — por eso toda cifra
ponderada de este documento es provisoria en su valor absoluto, aunque el orden
entre reglas sea estable.

### 4.1 Calidad del juez

| chequeo | resultado |
|---|---|
| Acuerdo de `mentions_ai_explicitly` con el texto real | **98,21%** (piso: parte de los 108 desacuerdos son del regex de control) |
| Citas que fallan literalidad | 14 de 1.507 (0,93%) |
| Citas **realmente** inventadas | **0** — las 14 son escapes de markdown (`M\*Modal`) que el juez cita limpio |
| Coherencia interna | 100%: 0 positivos sin categoría, 0 negativos con categoría, 0 positivos sin cita, 0 negativos con cita |
| Largo del razonamiento | 196 chars promedio (84–420) |

Dos debilidades del esquema de salida:

- `is_ai_disclosure` es **100% redundante** con `relevance == "substantive"`
  (1.507 / 655 / 3.876, sin una sola discrepancia).
- `confidence` es un flag de tres niveles, no una probabilidad: 93,5% de las
  filas dicen exactamente 1,0, y hay 3 valores en todo el conjunto. **13 de las
  14 citas no literales tienen confianza 1,0**, o sea que no detecta el problema
  que uno esperaría. Lo que sí lo detecta es el chequeo de literalidad.
- Pendiente para un v2: poner `reasoning` **primero**. Hoy es el último campo, y
  con salida estructurada el modelo emite la decisión antes de razonar, así que
  el razonamiento es justificación post-hoc, no cadena de pensamiento.

---

## 4.2 ADVERTENCIA: las cifras ponderadas de §5–§8 no son válidas todavía

Dos defectos del muestreo, encontrados al intentar proyectar al corpus. Ambos
corregidos en `golden_set.py`, pero las etiquetas existentes se produjeron
antes:

1. **Los pesos de la etapa 1 estaban rotos.** El peso se buscaba en un índice
   armado con las claves de 7 partes de la etapa 2, y el estrato de la etapa 1
   tiene 3 partes: nunca acertaba y caía a un fallback de 1/n. Las 3.950 filas
   del sobremuestreo tech —donde están 1.172 de los 1.507 positivos— pesaban
   **36 párrafos de corpus en total** en vez de 424.051. Todo lo "ponderado" se
   calculó, en la práctica, con las 2.088 filas de la etapa 2.
   Pesos recalculados en `data/interim/golden_set/golden_set_weights_v2.parquet`.

2. **El etiquetado iba ordenado por estrato.** Al cortarse por créditos, lo
   etiquetado no es una submuestra aleatoria del diseño sino los estratos
   alfabéticamente primeros: `stage3_random` quedó con **0 etiquetas**. Los
   pesos suponen que se etiquetó la muestra completa, así que proyectar desde
   este subconjunto sobreestima. Ya corregido a orden aleatorio sembrado, para
   que cualquier corte parcial siga siendo submuestra válida.

Verificación que lo destapó: dos estimaciones que deberían coincidir no lo
hacen. Proyectando por diseño desde el golden set, el logit marcaría ~14.000
párrafos únicos; aplicado directamente a los 3.281.038, marca 8.280 (~4.200 en
únicos). **B/A = 0,30x.** La aplicación directa (B) no depende de los pesos y es
la cifra utilizable; la proyección (A) no lo será hasta terminar el etiquetado.

Hallazgo lateral, independiente de los defectos: el pool deduplicado por texto
exacto es de **1.651.191 párrafos**, la mitad del corpus. El resto es
boilerplate repetido literal entre filings.

**Las comparaciones RELATIVAS entre reglas siguen en pie** — todas se calcularon
con los mismos pesos, así que el orden se sostiene. Lo que no se puede citar son
los valores absolutos ponderados ni ninguna proyección al corpus.

---

## 5. Comparación de reglas (6.038 etiquetas)

`prec`/`recall`/`F1` sobre `is_ai_disclosure`. "Ponderado" usa
`inclusion_weight` para proyectar al corpus.

| regla | F1 estrato | prec pond. | recall pond. | **F1 pond.** |
|---|---|---|---|---|
| léxico fuerte | 0,731 | 0,545 | 0,884 | **0,674** |
| léxico fuerte o débil | — | 0,398 | 0,986 | 0,568 |
| semántico ≥ 0,60 | 0,542 | 0,027 | 0,839 | 0,053 |
| semántico ≥ 0,65 | 0,500 | 0,074 | 0,716 | 0,134 |
| margen ≥ 0,05 | 0,618 | 0,036 | 0,839 | 0,069 |
| fuerte O sem ≥ 0,70 | **0,770** | 0,353 | 0,929 | 0,511 |
| fuerte O margen ≥ 0,10 | 0,765 | 0,199 | 0,952 | 0,329 |
| fuerte O (débil Y margen ≥ 0,06) | 0,739 | 0,493 | **0,958** | 0,651 |
| fuerte O (bm25 ≥ 8 Y margen ≥ 0,04) | 0,730 | 0,535 | 0,919 | 0,677 |

**El resultado que más importa**: dentro del estrato tech el híbrido gana cómodo
(0,770 vs 0,731), pero **proyectado al corpus el semántico puro colapsa** — un
umbral de 0,60 marcaría ~44.000 párrafos para acertarle al 2,7%. En el estrato
uno de cada cuatro párrafos es positivo; en el corpus, uno de cada 250, y la
señal semántica no distingue a esa escala.

---

## 6. Métricas por categoría: tres lecturas, tres conclusiones distintas

Sobre las primeras 4.000 etiquetas. **Este es el resultado metodológicamente más
importante del registro**, porque las tres métricas ordenan las categorías de
forma casi opuesta.

| categoría | base | argmax "acierta" | ROC-AUC | AP | lift | P@50 |
|---|---|---|---|---|---|---|
| ai_risk | 10,4% | 93% | 0,960 | **0,768** | 7,4x | 92% |
| ai_use | 21,9% | 87% | 0,788 | 0,517 | 2,4x | 82% |
| ai_capability | 16,9% | **93%** | 0,764 | 0,392 | 2,3x | 66% |
| ai_strategy | 8,6% | 66% | 0,833 | 0,314 | 3,7x | 56% |
| ai_governance | 1,5% | **4%** | 0,924 | 0,246 | **16,4x** | 30% |
| ai_outcome | 4,5% | 86% | 0,764 | **0,135** | **3,0x** | 18% |
| ai_exploration | 0,9% | 9% | 0,871 | 0,125 | 13,9x | 10% |

1. **La precisión del argmax mide tasas base, no calidad de anclas.** El juez
   asigna 2,1 categorías por párrafo y `best_semantic_anchor` elige una sola.
   `ai_governance` acierta el 4% porque pierde siempre contra `ai_risk` — en
   filings "board oversight of AI risk" es honestamente ambas. Y `ai_capability`
   gana el 93% de sus argmax con de los peores AUC.
2. **El ROC-AUC se infla con desbalance.** Las bases van de 21,9% a 0,9%.
   `ai_governance` da 0,924 de AUC pero 0,246 de AP.
3. **AP + lift es la lectura correcta**, y revela la categoría realmente floja,
   invisible bajo las otras dos: **`ai_outcome`**, con lift 3,0x y 18% de
   precisión en el top-50. Sus anclas hablan de productividad, costos e ingresos
   — el lenguaje de medio MD&A tenga IA o no.

Corolario para el pipeline: `best_semantic_anchor` es una conveniencia, no una
clasificación. La asignación de categorías debe salir de los `score_<categoría>`
con umbral por categoría.

---

## 7. Mejoras intentadas que NO funcionaron

Las tres son transformaciones sin entrenamiento y las tres chocan contra el
mismo techo, lo cual ya es información: no falta afinar, falta señal.

| intento | resultado | por qué |
|---|---|---|
| BM25 con vocabulario a mano (IDF del corpus completo) | +0,003 F1 | El IDF va de 4,4 a 13,3 — sólo 3x. Un párrafo con "model" x5 supera a uno con "artificial intelligence" x1. |
| Centrado del espacio de embeddings (restar la media del corpus) | **−0,038 F1** | `sem − neg` ya *es* un centrado, con mejor referencia que la media. Hacer ambos es redundante. |
| TF-IDF con vocabulario aprendido del corpus (591.363 términos, ajustado en 600.000 párrafos) | +0,002 F1 | Los anchors son **malas queries léxicas**: escritos como prosa de filing, en espacio TF-IDF matchean medio corpus. Correlación con el margen semántico: 0,623. |

Dato útil de paso: la lista `weak` tiene términos muertos. Sobre 3.281.038
párrafos, `classifier` aparece en **5**, `nlp` en 6, `claude` en 6, `gpt-4` en 7.
Contra `model` en 38.801.

Punto de operación de alta precisión, por si alguna etapa posterior lo necesita:
`tfidf + semántico` da **precisión 0,741 con recall 0,592**, que ninguna otra
regla alcanzó.

---

## 8. Modelos aprendidos (sklearn, CV agrupada por filing)

6.038 etiquetados, 1.507 positivos (25,0%), 2.564 filings distintos.
`GroupKFold(5)` por `accession_number` — agrupar importa porque el mismo
boilerplate aparece en varios párrafos del mismo filing. Umbral elegido **dentro
del fold de entrenamiento**; todas las cifras son out-of-fold.

| modelo | F1 estrato | prec pond. | recall pond. | **F1 pond.** |
|---|---|---|---|---|
| léxico fuerte (referencia) | 0,731 | 0,545 | 0,884 | 0,674 |
| fuerte O (débil Y margen ≥ 0,06) | 0,739 | 0,493 | 0,958 | 0,651 |
| prototipo: centroide de positivos | 0,635 | 0,524 | 0,561 | 0,542 |
| **logit sobre las 11 señales del prefiltro** | 0,736 | 0,717 | 0,802 | **0,757** |
| logit sobre 1024 dims (C=0,1) | **0,824** | 0,622 | 0,812 | 0,704 |
| logit sobre 1024 dims (C=1,0) | 0,808 | 0,368 | 0,786 | 0,502 |
| logit sobre señales + 1024 dims | **0,856** | 0,690 | 0,812 | 0,746 |

**Gana el modelo chico**: regresión logística sobre las 11 señales que el
pipeline ya calcula (7 scores por categoría + `negative_similarity` +
`semantic_margin` + los dos flags léxicos). F1 ponderado **0,757 contra 0,674**
de la mejor regla: +8,3 puntos, y sube precisión de 0,545 a 0,717 perdiendo 8
puntos de recall.

### 8.1 Reajuste con el golden set completo (2026-09-04)

El golden set ya está completo: 9.884 etiquetas legibles (vs 6.038 arriba),
incluyendo por fin `stage3_random` — el estrato del que dependía toda
reponderación confiable (ver §9 punto 1 de la versión anterior de este
documento). Reajustando el mismo modelo (`scripts/verif/prefilter_logit_refit.py`,
mismas 11 señales, mismo `GroupKFold(5)` por `accession_number`, sin volver a
embeber nada) sobre el set completo:

| modelo | F1 estrato | prec pond. | recall pond. | **F1 pond.** |
|---|---|---|---|---|
| léxico fuerte (referencia, 9.884) | 0,793 | 0,523 | 0,995 | 0,686 |
| **logit sobre las 11 señales (9.884)** | 0,807 | 0,736 | 0,971 | **0,837** |

**La ventaja del modelo se hace más clara, no desaparece**: +8 puntos de F1
ponderado sobre la corrida anterior (0,837 vs 0,757) y +15 puntos sobre la
regla léxica en el mismo set completo (0,837 vs 0,686). Las cifras del §8
original quedan como referencia histórica, no como recomendación — **usar
0,837 / la fila de 9.884 etiquetas** para cualquier decisión de threshold o
comparación futura.

Advertencia metodológica encontrada al reajustar: `LogisticRegression(...,
class_weight="balanced")` sobre el corpus completo (muy desbalanceado)
colapsa el modelo en una copia casi exacta de `strong_lexical_match` —F1
ponderado idéntico a 3 decimales a la regla léxica pura, perdiendo toda la
señal semántica. `class_weight=None` (default de sklearn) es lo que
produce la fila de arriba; no está confirmado si el §8 original usó
`balanced` o no, así que sus cifras de 6.038 podrían estar afectadas por el
mismo efecto — otra razón para tratar §8 como histórico y no recalcularlo
hacia atrás.

### 8.2 Modelo final desplegado + funnel del corpus (2026-09-04)

`scripts/common/ai_prefilter_classify.py` es el paso que faltaba: §8.1 solo
evaluaba el modelo (out-of-fold), esto lo **despliega** — elige threshold,
reajusta sobre todo el golden set, y marca el corpus completo.

**Metodología** (evita el error clásico de elegir threshold y evaluar con
los mismos datos):
1. `GroupKFold(5)` por `accession_number` sobre las 9.884 etiquetas,
   guardando las PROBABILIDADES out-of-fold (no solo la predicción con
   corte 0,5 default).
2. El threshold (barrido en pasos de 0,01) es el que maximiza F1
   ponderado sobre esas probabilidades out-of-fold — nunca se mira una
   fila usada para elegir el corte con un modelo que la vio en
   entrenamiento.
3. Con threshold ya fijo, el modelo FINAL se reajusta sobre las 9.884
   etiquetas completas (estándar: CV es para validar/afinar, el modelo que
   se despliega usa todas las etiquetas disponibles) y se aplica a los
   3.281.038 párrafos del corpus.

**Resultado**: threshold **0,56**. CV out-of-fold: F1 estrato 0,788 | prec
pond. 0,750 | recall pond. 0,950 | **F1 pond. 0,838** (consistente con el
0,837 de §8.1, que no había optimizado el threshold).

**Funnel del corpus completo** (`data/interim/prefilter_predictions/`):

| etapa | párrafos | % del corpus |
|---|---|---|
| corpus completo | 3.281.038 | 100% |
| pasa léxico fuerte o débil (determinista) | 45.992 | 1,40% |
| pasa léxico fuerte (determinista, más estricto) | 16.802 | 0,51% |
| **positivo del modelo logit (clasificador final)** | **12.840** | **0,39%** |

0,39% coincide con la estimación de §5 ("en el corpus, uno de cada 250") —
consistencia entre dos análisis hechos en momentos distintos con métodos
distintos. Los 12.840 párrafos son la población real que
`scripts/common/ai_classify.py` (extracción de frames semánticos) debe
consumir — no la muestra del golden set, que era un sustituto provisorio
mientras este paso no existía.

### 8.3 ¿El modelo pierde positivos reales? Sí, un poco — y no vale la pena perseguirlo (2026-09-04)

Pregunta que motivó esto: por qué el modelo final marca MENOS párrafos
(12.840) que el léxico fuerte solo (16.802). Verificado
(`data/interim/prefilter_predictions/` cruzado con `strong_lexical_match`):
**el modelo nunca predice positivo cuando `strong_lexical_match=False`**
— su coeficiente (+5,31, ver §8.2) domina tanto la combinación lineal que
ninguna señal semántica alcanza a compensarlo. El modelo funciona como un
filtro de PRECISIÓN dentro de lo que el léxico ya atrapa (saca 3.962 falsos
positivos léxicos), pero no rescata nada fuera de él.

Contra el golden set: **39 de 1.731 positivos reales (2,25%) tienen
`strong_lexical_match=False`** — ~0,46% de la masa ponderada positiva. Son
casos que el modelo pierde garantizado.

**Dos formas de rescatarlos, probadas con CV anidado real**
(`scripts/verif/prefilter_rescue_eval.py` — el threshold de cada candidato
se elige SOLO con el fold de entrenamiento, nunca con el fold que después
se evalúa, a diferencia de §8.2 — ver la advertencia metodológica abajo):

| candidato | F1 estrato | prec pond. | recall pond. | **F1 pond.** |
|---|---|---|---|---|
| baseline (solo modelo principal) | 0,779 | 0,709 | 0,782 | **0,744** |
| A: regla (`weak_lexical` Y `semantic_margin`≥t) en el subgrupo sin léxico fuerte | 0,660 | 0,313 | 0,784 | 0,448 |
| B: modelo chico (mismas señales semánticas) en ese mismo subgrupo | 0,779 | 0,709 | 0,782 | 0,744 |

**Ninguno mejora sobre no hacer nada.** La regla (A) dispara demasiados
falsos positivos y hunde la precisión. El modelo chico (B) empata exacto
con el baseline — con solo ~31 positivos de entrenamiento por fold (39
en todo el golden set, repartidos en 5 folds), no hay señal suficiente
para aprender nada generalizable; en la práctica nunca dispara sobre el
fold de test. **Decisión: se mantiene el modelo de §8.2 tal cual**, sin
mecanismo de rescate — el hueco (0,46% de masa positiva) queda como
limitación conocida y documentada, no como algo a seguir persiguiendo con
tan pocos casos de entrenamiento disponibles.

**Advertencia metodológica encontrada de paso** (importante para leer §8.2
correctamente): ese threshold (0,56) se eligió agrupando las probabilidades
out-of-fold de los 5 folds y maximizando F1 ponderado sobre ESE MISMO
conjunto agrupado — una fuga sutil, porque el threshold queda optimizado
sobre los mismos datos con los que después se reporta la métrica. El CV
anidado de este experimento (threshold elegido dentro de cada fold de
entrenamiento, nunca sobre el fold de test) da una estimación más honesta:
**F1 ponderado 0,744, no 0,838**. La diferencia (~10 puntos) es el costo de
elegir threshold sin anidar el CV. §8.2 se deja como está (el modelo y el
threshold desplegados no cambian, ya están fijados y aplicados al corpus),
pero **0,744 es la cifra a citar** como estimación de performance
out-of-sample, no 0,838.

### 8.4 Entidades nombradas de IA: override determinista probado y descartado (2026-09-04)

Idea: una lista de nombres propios inequívocos de IA (OpenAI, ChatGPT,
Anthropic, Claude, Gemini, Copilot, Midjourney, Stable Diffusion, watsonx,
Hugging Face, etc. — `NAMED_AI_ENTITIES` en
`scripts/common/ai_prefilter_classify.py`) que fuercen `is_ai_prefiltered
=True` sin pasar por el modelo, para casos como un párrafo real encontrado
en §8.3 ("net losses from investments in OpenAI", `strong_lexical_match
=True` pero igual filtrado por el modelo).

**Probado contra el golden set, no ayuda por la métrica.** De 76 filas con
`named_entity_match=True`, 30 (39%) son negativos reales — ejemplos:
"Search and news advertising, comprising Bing and Copilot..." (nombre de
segmento de negocio, no divulgación) y una nota de estados financieros que
excluye "net gains and losses from investments in OpenAI" (contable, no
divulgación). CV anidado (mismo threshold/C que el modelo solo, entidad
OR-combinada solo en el fold de test): F1 pond. 0,813 vs 0,814 sin el
override — un empate/leve retroceso, no una mejora. **`ai_prefilter_classify.py`
decide automáticamente por métrica** (`use_named_entity = combined > baseline`)
y en esta corrida quedó descartado. El código queda listo por si algún
cambio futuro en los signals hace que sí ayude — se reevalúa cada vez que
el script corre, no es una decisión congelada.

### 8.5 El modelo no se entrenaba con `inclusion_weight` — corregido, mejora real (2026-09-04)

Encontrado al buscar más mejoras: `ai_prefilter_classify.py` usaba
`inclusion_weight` para EVALUAR (correcto, es como se leen todas las
cifras ponderadas de este documento) pero nunca para **entrenar** — el
`.fit()` corría sin `sample_weight`, aprendiendo sobre la proporción de
positivos de la muestra estratificada (deliberadamente inflada, ver
docs/golden_set_sampling.md), no la del corpus real.

`scripts/verif/prefilter_model_variants_eval.py` prueba, con el mismo CV
anidado de §8.3 (threshold y ahora también `C` elegidos solo con el fold
de entrenamiento):

| variante | F1 estrato | prec pond. | recall pond. | **F1 pond.** |
|---|---|---|---|---|
| 0. baseline (como en §8.3: sin `sample_weight`, C=1,0) | 0,780 | 0,708 | 0,786 | 0,745 |
| 1. `sample_weight=inclusion_weight` en el fit, C=1,0 | 0,633 | 0,721 | 0,744 | 0,732 |
| **2. (1) + grid de C** | 0,631 | 0,745 | 0,898 | **0,814** |
| 3. (2) + `max_semantic_score` como señal 12 | 0,611 | 0,882 | 0,726 | 0,796 |

**Gana (2) por 7 puntos sobre el baseline honesto de §8.3** (0,814 vs
0,745) — recall pond. sube de 0,786 a 0,898 con precisión similar (0,745
vs 0,708). Agregar `max_semantic_score` (variante 3) empeora: sube
precisión pero hunde recall, peor F1 pond. neto.

**Desplegado** (`ai_prefilter_classify.py` actualizado): `C` y threshold
de despliegue se eligen en una pasada final sobre TODO el golden set
(esto es selección de hiperparámetros, no la cifra de performance — esa
es la de la tabla, obtenida con datos que el modelo desplegado nunca
usó para elegir su propia configuración). Resultado en esta corrida:
**C=0,01, threshold=0,45**, **11.546 párrafos marcados IA-relevantes**
(0,35% del corpus — antes 12.840/0,39% con el modelo sin corregir).
Menos en cantidad bruta, pero mejor calibrado: la regularización fuerte
(C=0,01) con pesos correctos generaliza a la proporción real del corpus
en vez de sobreajustar el conteo de la muestra estratificada.

**F1 pond. 0,814 es la cifra a citar de acá en adelante**, reemplazando
el 0,744 de §8.3 (que a su vez ya había reemplazado el 0,838 leakeado de
§8.2). El pipeline completo (§8.2 → §8.3 → §8.5) queda como registro de
cómo se llegó ahí, no se reescribe retroactivamente.

**Extracción de frames sobre los 12.840 (2026-09-04)**: corrida completa vía
`scripts/common/ai_classify.py` (qwen/qwen3.7-flash por OpenRouter) —
13.116 párrafos clasificados con éxito (incluye el subconjunto ya hecho
antes vía el golden set, que quedó cubierto por el diseño aditivo), 22
con error persistente (reintentables, no bloqueantes), **18.511 frames
semánticos** extraídos en total. Costo: **US$1,56**.

**El modelo de 1024 dims sobreajusta al diseño muestral, no a los datos.**
Out-of-fold anda bien (0,824 en el estrato), pero al reponderar cae a 0,704, y
con menos regularización se desploma a 0,502. La curva de aprendizaje lo delata:

| fracción de entrenamiento | F1 estrato | F1 ponderado |
|---|---|---|
| 25% | 0,749 | 0,309 |
| 50% | 0,806 | 0,288 |
| 100% | 0,824 | 0,704 |

Eso no es una curva convergiendo. Con 6.000 casos el modelo grande aprende la
estructura del **estrato muestreado**, no la del corpus. Es un sobreajuste más
sutil que el clásico porque sólo aparece al reponderar.

El **prototipo** (centroide de los positivos como anchor aprendido) falla
(0,542): promedia siete categorías muy distintas — riesgo, uso, gobernanza — en
un vector que no se parece a ninguna.

---

## 9. Qué falta

1. ~~**Completar el golden set.**~~ Resuelto 2026-09-04 (9.884 etiquetas
   legibles, `stage3_random` incluido) — ver §8.1. Las cifras ponderadas ya
   no son provisorias.
2. **Decidir regla vs modelo.** Con el golden set completo el logit de 11
   señales gana por 15 puntos de F1 ponderado (0,837 vs 0,686), no 8 — la
   brecha se agrandó, no se achicó. Sigue siendo una decisión de tesis
   (coeficientes vs regla explicable), pero con el modelo más claramente
   adelante.
3. **Contexto en el embedding.** Hoy cada párrafo se puntúa aislado: un
   encabezado "Risks Related to Artificial Intelligence" y su contenido son dos
   filas sin relación. Concatenar el encabezado de sección antes de embeber es
   la mejora estructural pendiente; cuesta re-embeber (~45 min) y no se puede
   evaluar sin hacerlo.
4. **Las otras cabezas de bge-m3.** El modelo produce `sparse_vecs` (pesos
   léxicos aprendidos, la versión buena de lo que §7 intentó a mano) y
   `colbert_vecs` (interacción tardía, que atacaría la dilución del vector denso
   en párrafos largos). `SentenceTransformer` sólo expone la cabeza densa;
   requiere `FlagEmbedding.BGEM3FlagModel` y re-embeber. ColBERT a escala de
   corpus no es viable por almacenamiento (vectores por token); su uso natural
   es como reranker sobre los candidatos que sobreviven al prefiltro.
