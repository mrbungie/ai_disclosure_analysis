# Evaluación del prefiltro: registro de evidencia

Todas las mediciones hechas sobre el corpus real, con el número de casos y las
advertencias de cada una. Sirve para no repetir experimentos ya hechos y para
poder citar de dónde sale cada decisión.

Corpus de referencia: **3.281.038 párrafos** (`us` 10-K 1.544.496 / 10-Q
1.736.542), 469.231.281 tokens. Modelo `BAAI/bge-m3`, 1024 dims.

Fecha de las corridas: 2026-09-02.

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

1. **Completar el golden set.** Faltan 3.862 (3.472 sin intentar + 390 fallidos
   por créditos). Sin `stage3_random` no hay reponderación confiable, y todas
   las cifras ponderadas de acá son provisorias en valor absoluto.
2. **Decidir regla vs modelo.** El logit de 11 señales gana por 8 puntos pero es
   un modelo con coeficientes; la regla es explicable. Es una decisión de tesis,
   no técnica.
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
