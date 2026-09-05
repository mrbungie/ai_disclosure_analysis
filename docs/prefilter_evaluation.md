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

**ADVERTENCIA: la cifra 0,814 y el modelo C=0,01/threshold=0,45 de esta
sección quedaron obsoletos el mismo día — ver §8.6. El bug no era la
idea (usar `sample_weight` sí es correcto), era la escala del peso
crudo.

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

### 8.6 `inclusion_weight` crudo colapsaba el tamaño efectivo de muestra — corregido con `sqrt` (2026-09-04)

El usuario desconfió del conteo de §8.5 (11.546, comparado con 12.840 del
modelo anterior) y pidió revisar con ejemplos concretos en vez de solo la
métrica agregada — buena decisión, porque el modelo de §8.5 estaba roto.

**Diagnóstico.** `inclusion_weight` (ver docs/golden_set_sampling.md)
abarca ~5 órdenes de magnitud entre estratos:

| estrato | filas | media | min | max |
|---|---|---|---|---|
| stage1_tech_oversample | 3.950 | 0,009114 | 0,005025 | 0,03125 |
| stage2_max_variation | 4.439 | 275,690246 | 1,0 | 6572,25 |
| stage3_random | 1.495 | 1100,794 (constante) | — | — |

Al ajustar `.fit(..., sample_weight=inclusion_weight)` con el peso crudo,
el tamaño de muestra efectivo (`(Σw)² / Σw²`) es **1.964 de 9.884 filas**
— el ajuste queda dominado por stage2/stage3 e ignora casi todo
stage1_tech_oversample, que es justo donde viven los positivos limpios e
inequívocos.

**Verificación concreta** con 3 párrafos del golden set, todos
`is_ai_disclosure=True`, todos de stage1_tech_oversample (peso
≈0,005–0,008): IBM watsonx (`0000051143-23-000032`, item `2`, párrafo
`913`), herramientas de IA de Microsoft (`0001564590-22-026876`, item
`1`, párrafo `33`), FactSet AI Blueprint (`0001013237-24-000141`, item
`1`, párrafo `34`).

| modelo | IBM watsonx | Microsoft AI tools | FactSet AI Blueprint |
|---|---|---|---|
| §8.3 (sin peso) | 0,794 | 0,812 | 0,885 |
| §8.5 (peso crudo, C=0,01) | **0,105** | **0,117** | **0,111** |

Los tres casos, positivos inequívocos usados para entrenar el propio
modelo, quedan **excluidos** por §8.5 (threshold 0,45) pese a que la
métrica agregada de esa sección decía que había mejorado. La métrica no
detecta el problema porque está calculada con los mismos pesos sesgados.

**Corrección probada** en `scripts/verif/prefilter_model_variants_eval.py`
(mismo CV anidado de §8.3/§8.5; el ajuste usa el peso transformado, la
evaluación siempre usa `inclusion_weight` crudo):

| variante de peso para el fit | F1 estrato | prec pond. | recall pond. | **F1 pond.** | 3 casos de control |
|---|---|---|---|---|---|
| sin peso (§8.3) | 0,780 | 0,708 | 0,786 | 0,745 | 0,79 / 0,81 / 0,89 |
| `inclusion_weight` crudo (§8.5) | 0,631 | 0,745 | 0,898 | 0,814 | 0,11 / 0,12 / 0,11 ❌ |
| **`sqrt(inclusion_weight)`** | 0,753 | 0,747 | 0,924 | **0,826** | 0,65 / 0,69 / 0,82 ✓ |
| `log1p(inclusion_weight)` | 0,761 | 0,721 | 0,759 | 0,740 | 0,74 / 0,80 / 0,89 |
| `inclusion_weight` recortado a percentil 95 | 0,604 | 0,743 | 0,723 | 0,733 | 0,13 / 0,17 / 0,33 ❌ |

`sqrt(inclusion_weight)` gana en ambos frentes: mejor F1 ponderado que
cualquier otra variante probada (0,826, incluida la de §8.5) Y los 3
casos de control se mantienen correctamente clasificados con margen
cómodo sobre cualquier threshold razonable. El recorte a percentil 95 y
el peso crudo fallan el chequeo de sanidad pese a tener F1 pond.
razonable — la lección general: **nunca aceptar una métrica agregada
mejor sin además revisar casos concretos ya verificados.**

**F1 pond. 0,826 es la cifra a citar de acá en adelante**, reemplazando
el 0,814 de §8.5 (que queda como registro de un bug real y su
diagnóstico, no como resultado válido).

**Redesplegado** (`ai_prefilter_classify.py` actualizado para ajustar con
`sqrt(inclusion_weight)` en vez del peso crudo, en las tres pasadas: CV
anidado, búsqueda de C/threshold de despliegue, y el modelo final).
Resultado: **C=3,0, threshold=0,51**, **11.561 párrafos marcados
IA-relevantes** (0,35% del corpus). Los 3 casos de control quedan
correctamente incluidos (`is_ai_prefiltered=True`) en el corpus
desplegado. El conteo bruto (11.561) queda cerca del de §8.5 (11.546)
por coincidencia — la cifra ya no es sospechosa porque esta vez está
verificada contra ejemplos concretos, no solo contra su propia métrica.

`named_entity_match` (§8.4) sigue sin mejorar la métrica bajo este
modelo (F1 pond. 0,825 vs 0,826 combinado) y sigue descartado del
despliegue.

Funnel actualizado: 3.281.038 párrafos totales → 45.992 con match léxico
(fuerte o débil) → 16.802 con match léxico fuerte → **11.561 marcados
IA-relevantes por el modelo**.

**Extracción de frames sobre la población corregida (2026-09-04)**: el
diseño aditivo/idempotente de `ai_classify.py` significa que casi toda la
población de 13.239 positivos ya estaba clasificada (el conjunto de
positivos apenas cambió: 11.561 vs 11.546 párrafos, más el resto ya
cubierto por el golden set) — solo **394 párrafos** quedaron pendientes en
esta sesión. Corrida: **383 clasificados con éxito, 11 con error
persistente** (consistente con la tasa de error base ~6% del bug de
pydantic-ai/Qwen ya documentado), 401 frames nuevos. Costo de esta
corrida: **US$0,030** (uso total OpenRouter US$1,588 de 20 acreditados,
quedan ~US$18,41). Total acumulado: **20.969 frames semánticos** sobre
**13.512 párrafos** con al menos un frame.

**ADVERTENCIA: el "13.239" de arriba estaba mal — era un artefacto de un bug
real, corregido y explicado en §8.7. La cifra correcta de población es
11.561/9.664 (instancias/textos únicos).**

---

### 8.7 El corpus tiene ~50% de párrafos duplicados; `ai_classify.py` los pagaba todos — corregido (2026-09-04)

El usuario preguntó si el funnel y la extracción de frames ya estaban
libres de duplicados y "texto raro". Respuesta corta: el filtrado de
basura de línea sí corre (§2/§3, más fuerte en Chile vía
`cmf_pdf_paragraphs.py`), pero **nunca hubo deduplicación de texto entre
filings** — y sí la hay, siempre, en el golden set (`docs/
golden_set_sampling.md` §5, "Deduplicación previa": deduplica por
`text_hash` antes de repartir cuotas; verificado empíricamente, 0 grupos
de `text_hash` repetido en las 9.884 etiquetas).

**Magnitud del problema.** De 3.281.038 párrafos del corpus, solo
1.651.191 (50%) son textos únicos (ya medido en §4.2) — el resto es
boilerplate literal repetido entre filings (disclaimers, secciones de
riesgo estándar, encabezados). Entre los 11.561 párrafos que el prefiltro
marca positivos, solo **9.664 son textos únicos** — 1.897 instancias son
repeticiones exactas de otro positivo ya visto.

**Corrección del funnel** (`ai_prefilter_classify.py::funnel_counts`):
cada etapa ahora reporta las dos lecturas, instancia y texto único:

| etapa | instancias | textos únicos |
|---|---|---|
| corpus total | 3.281.038 | 1.651.191 |
| match léxico (fuerte o débil) | 45.992 | 33.431 |
| match léxico fuerte | 16.802 | 14.191 |
| **positivo del modelo final** | **11.561** | **9.664** |

**Corrección de `ai_classify.py`: deduplicar ANTES de gastar LLM.**
Rediseñado para que la población de extracción de frames sea por
`text_hash` (BLAKE2b-8, el mismo de `ai_embed.py`/`golden_set.py`), no por
instancia de párrafo: se elige UN representante por texto único, se
clasifica una sola vez, y `ai_frames` queda **keyed by `text_hash`**, no
por `(country_code, form, accession_number, item_key, paragraph_index)` —
esas columnas ahora nombran solo la instancia representante que
efectivamente se mandó al modelo; `duplicate_count` registra cuántas
instancias del corpus comparten ese texto. Cualquier consumidor futuro
que necesite los frames de un párrafo específico debe calcular su propio
`text_hash` y unir por esa columna, no por la llave de párrafo.

**Dos bugs reales encontrados al implementar esto**, ambos corregidos:

1. **`QUALIFY` después de `WHERE` "resucitaba" positivos obsoletos.** La
   población se construía con `WHERE is_ai_prefiltered = true` seguido de
   `QUALIFY row_number() ... ORDER BY model_version DESC = 1` — pero el
   `WHERE` corre ANTES de la ventana, así que el ranking se calculaba solo
   entre filas YA positivas. Si la corrida más reciente marcaba una llave
   como negativa, esa fila quedaba fuera del `WHERE` y el rank 1 caía en
   una corrida vieja que sí la marcaba positiva — revivendo un positivo de
   un modelo ya descartado. Verificado concretamente: con el orden
   incorrecto la población salía **13.239** instancias; resolviendo
   primero la corrida más reciente por llave (sin filtrar) y recién ahí
   filtrando por `is_ai_prefiltered`, sale **11.561** — exactamente el
   conteo de la corrida desplegada, como debe ser (todas las corridas
   puntúan el corpus completo, así que nunca deberían divergir). El
   "13.239" citado en §8.6 arriba era este bug, no un número real.
2. **El `text_hash` persistido no coincidía con el texto real del
   párrafo.** Se calculaba sobre `" ".join(sentences)` (la reconstrucción
   usada para armar el prompt), que no reproduce byte a byte el
   `paragraph_text` real de la tabla `paragraphs` (separadores distintos).
   Efecto: de 721 párrafos recién clasificados, solo 7 hashes coincidían
   con la población real — la deduplicación por texto quedaba
   silenciosamente rota. Corregido para que el hash persistido salga
   siempre del `paragraph_text` canónico (traído en la misma consulta que
   arma la población), nunca de la reconstrucción usada solo para el
   prompt.

**Costo de la corrección**: el bug de `text_hash` obligó a reclasificar
~721 párrafos que ya se habían pagado con el hash equivocado (no se
pueden recuperar como "cubiertos" retroactivamente sin volver a llamar al
modelo, porque su hash guardado no sirve para el join). Costo total de
todo este episodio de depuración: **US$0,177** (de US$1,588 a US$1,765 de
uso acumulado). Los datos ya clasificados con hash incorrecto (de
sesiones anteriores a este fix) NO se borraron — siguen siendo frames
válidos para su propio párrafo, solo su columna `text_hash` es ruido; se
dejan como están (`data/`, nunca se borra sin archivar) y simplemente no
cuentan para deduplicación futura.

**Estado final tras la corrección**: de los 9.664 textos únicos
positivos, **9.663 clasificados** (11.560 instancias cubiertas), queda
**1 pendiente** (error persistente del bug de pydantic-ai/Qwen, se
reintenta solo en la próxima corrida). Nunca más se paga la extracción
de un mismo texto dos veces.

**Corrección arquitectónica inicial (2026-09-04): el dedup se mueve a la
fuente.** El usuario objetó, con razón, que parchar la deduplicación
adentro de `ai_classify.py` (calculando su propio hash, sus propios
grupos) es el patrón equivocado — la limpieza debería vivir en la tabla
`paragraphs` misma, no reimplementarse en cada script consumidor. Es
exactamente lo que causó el bug del hash desincronizado de arriba: dos
cálculos independientes del "mismo" hash, en dos lugares, que dejaron de
coincidir.

Corregido en `build_duckdb.py` (la ÚNICA fuente de la tabla `paragraphs`):

- `text_hash` (BLAKE2b-8, vía un UDF de DuckDB registrado una sola vez)
  ahora es una **columna de `paragraphs` misma**, calculada al construir la
  tabla — no un cálculo aparte en `ai_embed.py`/`golden_set.py`/
  `ai_classify.py`.
- Nueva tabla materializada **`unique_paragraphs`**: una fila por
  `text_hash` único, con una instancia representante determinística
  (la llave natural más chica del grupo) y `duplicate_count`. Es LA
  superficie de deduplicación del proyecto — 1.651.191 filas, coincide
  exactamente con el 50% ya medido en §4.2.

**¿Es creíble ~9.700/1.651.191 (0,59% de los textos únicos) con divulgación
de IA?** El usuario dudó del orden de magnitud. Hay un chequeo directo,
libre de cualquier sesgo del prefiltro: `stage3_random`, el único estrato
del golden set sin ningún supuesto de diseño (docs/golden_set_sampling.md
§5, "el único libre de cualquier supuesto ... da un estimador insesgado
de prevalencia"). Sobre sus 1.495 párrafos, **4 son positivos** —
prevalencia empírica 0,268% (IC 95% exacto de Clopper-Pearson: 0,073% a
0,684%, ancho por lo poco que son 4 eventos). Aplicado al corpus completo:
**8.779 positivos esperados, IC 95% [2.393, 22.430]**.

Los textos únicos que el modelo final marca caen justo dentro de ese
intervalo, cerca del punto central. No es una coincidencia forzada: el
diseño del golden set (§4 de ese mismo documento) YA anticipaba "~0,4% de
prevalencia" antes de tener modelo ni prefiltro — es el motivo por el que
se sobremuestreó tech en primer lugar (una muestra aleatoria simple daría
~40 positivos en 10.000 párrafos, insuficiente para medir nada por
categoría). El recall ponderado del modelo (~0,93, §8.8) tampoco sugiere
que se esté perdiendo una fracción grande de positivos reales. El número
bajo no es un error de conteo: es el resultado esperado de mirar
divulgación de IA a nivel de PÁRRAFO (no de filing) en un panel de 517
empresas de TODOS los sectores y varios años — la mayoría de un 10-K, en
la mayoría de las empresas, en la mayoría de los años, sencillamente no
menciona IA para nada.

---

### 8.8 Deduplicación como FASE del pipeline, no como columna calculada en cada consumidor (2026-09-04)

El usuario volvió a objetar, más fuerte: "no entiendo pq te fijas tanto en
los unicos, la dedup deberia ser una fase no algo que calcules cada vez".
Tenía razón otra vez — §8.7 arregló el dedup de `ai_classify.py`, pero
`ai_prefilter.py` (embeddings + scoring) y `ai_prefilter_classify.py`
(el modelo final + funnel) seguían corriendo sobre las 3.281.038
instancias completas, reportando "único" como una columna
`COUNT(DISTINCT ...)` calculada aparte en cada etapa — dedup como
curiosidad de reporte, no como fase real del pipeline.

**Corregido migrando el cómputo real, no solo el reporte:**

- `ai_prefilter.py::run_prefilter` ahora acepta `--source-relation`
  (default `paragraphs`, por compatibilidad). Corriendo con
  `--source-relation unique_paragraphs` puntúa cada texto único UNA vez —
  las 1.651.191 filas de `unique_paragraphs`, no 3.281.038 — reusando los
  embeddings ya calculados (la instancia representante de cada grupo ya
  tiene su propio vector de `ai_embed.py`, que embebió cada instancia).
  Corrida real: **19 segundos** para las 1.651.191 filas (el paso caro,
  embeddings, ya estaba pagado).
- `ai_prefilter_classify.py` reentrenado y aplicado sobre esa nueva
  corrida (`data/interim/prefilter_scores_unique/`). `load_golden()` ahora
  une por `text_hash` (no por llave de instancia), porque el muestreo del
  golden set eligió su PROPIO representante de deduplicación,
  independiente del que `unique_paragraphs` elige — verificado que ambos
  cubren el mismo texto, solo que no necesariamente con la misma llave.
  Salida: `data/interim/prefilter_predictions_unique/` — **UNA fila por
  texto único**, no por instancia, con `text_hash` y `duplicate_count`
  incluidos.
- `funnel_counts()` se simplificó a un número por etapa (ya no hay
  columna `_unique_text` en ningún lado) porque la etapa misma ya está
  deduplicada por construcción. `duplicate_count` se reporta UNA sola vez,
  al final, como la única cifra de escala de instancias que queda.
- `ai_classify.py::fetch_pending` se simplificó de nuevo: como
  `prefilter_predictions_unique` YA es una fila por texto único, ya no
  hace falta el JOIN intermedio contra `unique_paragraphs` que tenía la
  primera corrección de §8.7 — solo lee la tabla de predicciones tal cual.
  QUALIFY particiona por `text_hash` (no por llave de instancia), la
  identidad estable del texto.

**Resultado**, con las mismas 9.884 etiquetas del golden set: **C=3,0,
threshold=0,51, F1 pond. 0,828** (vs. 0,826 de §8.6 — diferencia mínima,
ruido de punto flotante fp16 entre corridas de scoring, no una regresión),
**9.670 textos únicos positivos (0,59% de 1.651.191), representando
11.567 instancias del corpus (0,35% de 3.281.038)**. `ai_classify.py`
reclasificó solo 1 texto nuevo (el resto ya estaba cubierto por sesiones
anteriores) y quedó en **9.670/9.670 (100%)**.

**La corrida anterior** (`data/interim/prefilter_predictions/`,
por-instancia, 3.281.038 filas) **queda en disco, no se borra** — es la
regla del proyecto para datos derivados — pero ya no es lo que
`ai_classify.py` lee; `data/interim/prefilter_predictions_unique/` la
sucede como fuente de verdad.

Funnel final (una sola lectura, sin columnas duplicadas):

| etapa | textos únicos | instancias que representan |
|---|---|---|
| corpus | 1.651.191 | 3.281.038 |
| match léxico (fuerte o débil) | 33.431 | — |
| match léxico fuerte | 14.191 | — |
| **positivo del modelo final** | **9.670** | **11.567** |

---

### 8.9 Chile entra al esquema de `paragraphs` (sin correr nada todavía) + dos bugs más encontrados al debuggear

Se le pidió al modelo que revisara más a fondo si los resultados tenían
sentido. Encontró que **todo el pipeline de clasificación de IA (prefiltro,
golden set, `ai_classify`) era US-only**, pese a haber 1.077.595 párrafos
de Chile ya extraídos (`data/interim/sections_cl/filing_paragraphs__run=*.
parquet`, de `scripts/cl/cmf_pdf_paragraphs.py`) que nunca entraban a
`paragraphs`: el pipeline de PDF de Chile extrae párrafos directo, sin el
paso intermedio de "texto de sección cruda" que `build_duckdb.py` espera
del pipeline HTML de EE.UU. (`filing_sections__run=*.parquet`) — un nombre
de archivo y esquema distintos que `_existing()` nunca matcheaba.

**Corregido en `build_duckdb.py`** con una nueva `_cl_paragraph_select_sql()`
que mapea el esquema de Chile al contrato compartido de `paragraphs`:

| columna de `paragraphs` | origen en Chile |
|---|---|
| `country_code` | literal `'cl'` |
| `form` | `filing_type` (`'annual'`/`'quarterly'` — vocabulario propio de Chile, deliberadamente NO forzado a `'10-K'`/`'10-Q'`: son formularios de la SEC, Chile no los presenta. `annual` cumple el mismo ROL que el panel 10-K, `quarterly` el de la serie de shocks 10-Q — mismo rol, etiqueta distinta, a propósito) |
| `accession_number` | `document_id` (verificado único: 1.948 valores distintos, y `(document_id, paragraph_index)` ya es único sin trabajo extra) |
| `item_key` | constante `'0'` — las Memorias/Análisis Razonado no tienen estructura de Item 1/1A/7 como el 10-K; una constante es honesta en vez de inventar secciones falsas |
| `is_scorable` | misma regla que EE.UU. (>3 caracteres útiles Y al menos un alfanumérico), calculada acá porque el extractor de Chile no la emite |

**IMPORTANTE — por instrucción explícita, no se corrió nada de Chile.**
`paragraphs`/`unique_paragraphs`/`sentences` ya incluyen Chile (1.077.595
párrafos, 818.709 textos únicos, 1.745.203 oraciones) porque están
construidas con SQL puro sobre lo que ya hay en disco — pero `ai_embed.py`
(embeddings), `ai_prefilter.py` (scoring) y `ai_prefilter_classify.py`
(el modelo) NO se ejecutaron sobre Chile: no hay embeddings para esos
818.709 textos, así que no pueden puntuarse todavía. Motivo explícito del
usuario: no confía aún en el parsing de documentos de Chile y prefiere
revisarlo antes de gastar cómputo en clasificarlo. El funnel de este
documento sigue siendo 100% EE.UU.

**Dos bugs reales encontrados exactamente por fusionar Chile al esquema**
(ninguno de los dos tiene que ver con clasificar contenido de Chile — son
sobre cómo `unique_paragraphs` se comporta al tener más de un país):

1. **`is_scorable` nunca se aplicaba.** El párrafo literal `"AI"` (2
   caracteres, `is_scorable=false`) salía con `predicted_proba=0,86` —
   matchea el término léxico fuerte "ai" sin tener ningún contenido
   detrás. Ni `ai_prefilter_classify.py` ni `ai_classify.py` filtraban por
   esa columna en ningún lado, pese a que existe desde antes (`build_
   duckdb.py`) exactamente para descartar esto. Corregido: `load_golden()`
   y la consulta del corpus en `ai_prefilter_classify.py` ahora exigen
   `up.is_scorable`. Efecto: 6 etiquetas basura salen del golden set
   (9.884 → 9.878), y las 785 instancias basura del corpus dejan de poder
   ser positivas nunca.
2. **Colisión de hash entre países escondía ~214.710 instancias de
   EE.UU.** Strings universales ("", un bullet "•", "100%", números
   sueltos como "(1)") son byte-idénticos entre el texto extraído de
   EE.UU. y el de Chile. Al fusionar Chile, el representante elegido para
   esos grupos (la llave natural más chica) saltó a una instancia
   **chilena** simplemente porque `'cl'` ordena antes que `'us'` — aunque
   el grupo sea ~99,99% párrafos de EE.UU. (ej.: el string vacío tiene
   139.157 miembros de EE.UU. y 1 de Chile, pero quedó etiquetado
   `country_code='cl'`). Como la corrida de scores existente se calculó
   ANTES de fusionar Chile (solo cubre representantes de EE.UU.), unir por
   **llave de instancia** contra esa corrida hacía que esos grupos
   desaparecieran enteros del corpus — verificado: el total de instancias
   del funnel cayó de los ~3.281.038 esperados a 3.012.397, mucho más de
   lo que explica solo el filtro de `is_scorable`. Corregido: todos los
   `JOIN` entre `unique_paragraphs` y la corrida de scores ahora son por
   **`text_hash`** (la identidad real del contenido), no por llave de
   instancia — el representante elegido puede cambiar entre reconstrucciones
   de la tabla sin que eso rompa nada. Resultado tras el fix:
   **3.016.700 instancias** en el funnel, que coincide casi exacto con las
   3.016.097 instancias reales de EE.UU. que pasan `is_scorable` (la
   diferencia de 603 es el sliver negligible de miembros chilenos que
   viajan colgando en esos mismos grupos basura — ninguno de ellos puede
   ser positivo).

Ninguno de los dos bugs cambió qué párrafos reales quedan marcados
IA-relevantes (**9.670 textos únicos, 11.567 instancias** — prácticamente
igual a antes): todo lo afectado era basura sin señal de IA. Pero ambos
eran reales, y el segundo en particular es una advertencia útil para
cuando Chile sí se puntúe: `unique_paragraphs.country_code` describe SOLO
al representante elegido, no a todos los miembros del grupo — cualquier
consulta futura que filtre `unique_paragraphs` por país debe unir por
`text_hash` contra la población real, nunca confiar en esa columna como
si describiera el grupo completo.

**Verificación explícita de falsos negativos en EE.UU. (pedida por el
usuario, "pero estas leyendo tonto weon"):** comparar el set de positivos
antes/después de los fixes de esta sección solo confirma que no se rompió
nada RELATIVO a la corrida anterior — no dice nada sobre falsos negativos
reales contra el corpus. Leyendo texto real de párrafos con match léxico
fuerte que el modelo NO marca positivo, aparece un hallazgo genuino:
**~110 párrafos justo debajo del threshold (predicted_proba entre 0,505 y
0,510, threshold=0,51)**, varios de ellos divulgación sustantiva real, no
ruido — ejemplos leídos a mano: "*continued customer interest in our new
generative AI product, watsonx Code Assistant for Z*" (0,50986), "*Cortex
XSIAM. Our cloud-based AI-powered security operations platform harnesses
the power of AI*" (0,50778), "*Microsoft Cloud gross margin... driven by
scaling our AI infrastructure*" (0,50752), "*investments in modern
technology including artificial intelligence and machine learning*"
(0,50917).

**Esto SÍ es un falso negativo real — pero es un trade-off cuantificado,
no un bug.** Sobre las etiquetas del golden set, variando el threshold:

| threshold | F1 pond. | precisión pond. | recall pond. | positivos (golden set) |
|---|---|---|---|---|
| **0,51 (desplegado)** | **0,852** | 0,786 | 0,929 | 1.644 |
| 0,45 | 0,825 | 0,732 | 0,947 | 1.912 |
| 0,40 | 0,829 | 0,726 | 0,968 | 2.102 |
| 0,35 | 0,773 | 0,640 | 0,975 | 2.300 |

Bajar el threshold SÍ recuperaría esos casos (recall 0,929→0,947 a 0,45),
pero empeora el F1 ponderado (0,852→0,825): en esa banda, sin ponderar,
el golden set tiene más positivos que negativos (148 vs 120 entre 0,45 y
0,51) — pero por PESO real de corpus, los negativos dominan largamente
(678 vs 128), justo el fenómeno que §4 ya advertía ("la proporción de
positivos en el golden set no dice nada sobre la proporción en el
corpus"). El threshold de 0,51 no está mal elegido — es el punto que
honestamente maximiza F1 ponderado — pero eso significa aceptar, a
propósito, perder una fracción real y no trivial de divulgaciones
genuinas de IA que caen justo en el borde. Queda como límite conocido y
cuantificado del enfoque de un único threshold global, no como algo para
"arreglar" subiendo el recall a costa de la métrica que se optimizó — sí
como candidato a revisar si en algún momento el objetivo cambia de
"mejor F1 ponderado" a "minimizar falsos negativos a cualquier costo".

**Candidato de regla determinista probado y rechazado: `AI-<palabra>`.**
Dado que varios de los falsos negativos leídos a mano eran compuestos con
guión ("AI-powered", "AI-Driven Enterprise"), se probó un override
determinista `(^|[^A-Za-z0-9])AI-[A-Za-z]+` — mismo patrón de evaluación
que `named_entity_match` en §8.4. El patrón solo tiene **93,9% de
precisión cruda** (215/229 en el golden set) pero **46,9% ponderada** —
los falsos positivos del patrón caen otra vez en los estratos de peso
alto, el mismo fenómeno de §8.4/§8.6. Combinado (OR) con el modelo bajo el
mismo CV anidado: **F1 pond. empeora de 0,829 a 0,778** (precisión pond.
0,745→0,664, ganancia de recall marginal 0,934→0,938). **Rechazado**, por
el mismo criterio que ya se usa en este documento: no se despliega nada
que empeore la métrica que se optimiza, por más que el ejemplo puntual
que lo motivó fuera real.

---

### 8.10 XGBoost probado y rechazado; dummies escalonadas del léxico + léxico internacional desplegados (2026-09-04)

**XGBoost, bajo el mismo CV anidado, pierde en todos los escenarios
probados.** Grilla permisiva (hasta max_depth=6): F1 pond. 0,788, y los 5
folds eligieron consistentemente el modelo más complejo de la grilla —
sobreajuste. Grilla con regularización fuerte (max_depth≤2, `reg_lambda`
alto, subsampling): peor todavía, F1 pond. 0,761, porque sacrifica recall
sin ganar precisión. Con `inclusion_weight` crudo (en vez de `sqrt`) una
configuración fija dio 0,845 — pero no sobrevivió a una búsqueda honesta
de hiperparámetros por fold (la misma disciplina de siempre): con grilla
real, 0,752-0,755. **Rechazado**: con solo 11 (ahora 12) señales ya
bastante informativas y ~9.878 filas con desbalance extremo, un ensamble
de árboles no tiene con qué aprender interacciones que valgan la pena —
solo sobreajusta el ruido de la muestra estratificada. La regresión
logística sigue siendo la mejor opción medida.

**Dummies escalonadas en vez de booleano plano para el léxico — mejora
real.** `strong_lexical_match`/`weak_lexical_match` colapsaban "0
términos" vs "1 o más" a un bit — pero en el golden set, más términos
matcheados es señal cada vez más fuerte y no lineal: 0 términos fuertes
→ 0,5% positivo, 1 → 59%, 2 → 72%, 3+ → 89-100%. Un conteo LINEAL del
número de términos matcheados es PEOR que el booleano (F1 pond. 0,745,
porque la logística asume una relación lineal que no existe: el salto
0→1 es enorme, 4→5 es chico). La solución es un escalón por nivel
(`strong_ge1`, `strong_ge2`, `weak_ge1` — más granularidad, ej.
`strong_ge4`, no mejora más): **F1 pond. 0,829 → 0,857-0,858**, estable
en tres variantes de granularidad probadas, con mejor precisión Y mejor
recall a la vez (no es un trade-off). Verificado contra los 3 casos de
control: los tres se mantienen o mejoran.

**Léxico internacional agregado, con la misma disciplina de verificación
que "descartó" `named_entity_match` en su momento.** El usuario notó que
la lista de marcas de IA (`configs/ai_prefilter.yaml`) era 100%
estadounidense (chatgpt/openai/copilot/gemini/claude). Antes de agregar
nombres, cada término ambiguo se verificó contra el corpus real:

- Aceptados (nombre de PRODUCTO inequívoco, nunca la empresa matriz):
  deepseek, qwen, ernie bot, chatglm, hunyuan, kimi, moonshot ai,
  mistral ai, aleph alpha, ai21 labs, veo, windsurf, cursor, etc.
- **Rechazados por colisión real verificada en el corpus**: "sora" (es
  "Singapore Overnight Rate Average", una tasa financiera, y también el
  nombre de una adquisición de ADP — cero relación con OpenAI Sora en
  los 14 párrafos que matcheaban); "devin" (coincide con "Devin W.
  Stockfish", CEO real de Weyerhaeuser, en cada firma de sus 10-K);
  "pika" ("Pika Energy", empresa real de energía adquirida por Generac);
  "runway" (jerga de negocios/aviación común, 58 párrafos, ninguno sobre
  Runway ML); "flux" (palabra inglesa genérica, 93 párrafos, ninguno
  sobre el modelo FLUX). También se excluyeron deliberadamente
  conglomerados genéricos (baidu, alibaba, huawei, tencent) — verificado
  que aparecen mayormente en contexto NO relacionado con IA (registradores
  de dominio, proveedores de GPU/telecom).
- Config reestructurado a `lexical.concepts` (vocabulario genérico) +
  `lexical.entities` (geografía → modalidad → términos: text/image/video/
  audio/three_d/coding), a pedido explícito para preservar PROVENANCE sin
  repetir metadata por término — la jerarquía misma es la metadata.
  `ai_prefilter_anchors.py::entity_terms()` expone esa metadata aplanada
  (`{término: {geo, modalidad}}`) para un futuro cálculo de "model
  leaning" (EE.UU./China/Europa) sin una segunda pasada de LLM — el
  prefiltro hoy solo consume la lista aplanada sin distinción, no calcula
  ningún leaning todavía.

**Redesplegado** con ambos cambios juntos: **C=10,0, threshold=0,52, F1
pond. 0,856**, **8.915 textos únicos marcados IA-relevantes (10.658
instancias)** — bajó de 9.670/11.567 pese a que la métrica mejoró, lo
cual generó dudas razonables ("¿no dijiste que mejoraba?"). Verificado
con el diff completo contra la corrida anterior: 8.263 en común, 1.406
removidos, 652 agregados.

- Los 1.406 removidos: muestra leída a mano — boilerplate genérico de
  riesgo competitivo, cadena de suministro, descripciones de producto sin
  sustancia de IA, todos ya en el borde del umbral viejo (proba
  0,51-0,58). Ninguno se leyó como divulgación sustantiva perdida.
- Los 652 agregados: la muestra inicial preocupó (varios parecían
  boilerplate de "forward-looking statements" con 2+ términos genéricos
  como "artificial intelligence" + "machine learning" y margen semántico
  negativo). Verificado contra el golden set: ese patrón exacto
  (`strong_ge2` + margen semántico bajo) tiene **68,7% de precisión
  cruda y ~83% ponderada** (peso positivo 1.572 vs negativo 327) — es
  decir, es mayoritariamente positivo de verdad según las etiquetas
  reales, no un artefacto de la muestra de lectura manual.

---

### 8.11 El target estaba mal definido: "sustantivo" no es lo mismo que "menciona IA" — y el foco del proyecto es AI-washing (2026-09-04)

El usuario preguntó si el pipeline no estaba perdiendo señal, específicamente
menciones de IA que "no son concretas". La respuesta fue que sí, y el
motivo es de diseño, no de ajuste fino: `golden_set.py`'s `SYSTEM_PROMPT`
define `is_ai_disclosure = true` **si y solo si** `relevance == "substantive"`
— una mención "incidental" (la IA aparece pero el párrafo no afirma nada
concreto) se etiquetaba `is_ai_disclosure = False`, es decir, se entrenaba
como NEGATIVO. Eso es exactamente lo contrario de lo que este proyecto
necesita: el foco es detectar **AI-washing** (divulgación de IA vaga o
promocional sin sustancia real detrás), así que si una mención no es
concreta ESA ES LA SEÑAL a estudiar río abajo, no algo para filtrar en el
prefiltro. La responsabilidad de juzgar "sustantivo vs. vago" le
corresponde a una etapa posterior (el propio análisis de AI-washing), no
a esta.

**Corrección, sin gastar un peso de LLM más**: el juez del golden set
siempre guardó `relevance` (`none`/`incidental`/`substantive`) además del
binario — el dato ya estaba ahí. Nuevo target: **`is_ai_mention = relevance
!= 'none'`**, reemplazando `is_ai_disclosure` en `ai_prefilter_classify.py`
(`load_golden()` y el `y` de entrenamiento). Efecto sobre las 9.878
etiquetas: 786 filas "incidental" (mención real de IA, sin afirmación
concreta) pasan de negativo a positivo — el conteo de positivos sube de
1.731 a **2.517** (+45%).

**Resultado**, mismo CV anidado de siempre: **C=10,0, threshold=0,75, F1
pond. 0,947** (vs 0,856 del target anterior — la tarea "¿menciona IA?" es
genuinamente más fácil de separar que "¿es sustantivo?", con menos
ambigüedad en el borde). Corpus: **13.373 textos únicos marcados
IA-relevantes (0,81% de 1.650.145), representando 15.868 instancias**
— sube de 8.915/10.658, +50%, consistente con recuperar las menciones
incidentales que antes se descartaban.

Verificado con los 3 casos de control (ahora triviales, 0,997-0,9998,
eran ya sustantivos) y con ejemplos reales `relevance=incidental` del
golden set que antes el modelo hubiera aprendido a rechazar — ahora
correctamente positivos (0,97-0,99), ej.: "*The Intelligent Edge is
characterized by ubiquitous sensing, hyper-scale and edge computing,
artificial intelligence (AI) and pervasive connectivity*" — una mención
real pero genérica/promocional, exactamente el tipo de caso que el
análisis de AI-washing necesita ver, no perder.

`ai_classify.py` corrido sobre la población expandida: **2.159 textos
nuevos, 2.089 clasificados con éxito en la primera pasada (2.324 frames),
70 con error; un reintento bajó el error persistente a 1/13.374 (99,99%
cobertura)**. Costo real: **US$0,204** (uso acumulado OpenRouter
US$1,834 → US$2,038 de 20 acreditados, quedan ~US$17,96).

---

### 8.12 Dos tablas gold + detector de menciones de entidad (2026-09-04)

A pedido explícito, se construyeron:

- **`scripts/common/ai_entity_mentions.py`**: sobre la población
  IA-relevante (13.373 textos), matchea los términos de `lexical.entities`
  (§8.10) con la misma técnica de límite de palabra que el prefiltro,
  agregando `geo`/`modality` por mención. Sin LLM, se recalcula entero
  cada corrida. Salida: `data/interim/ai_entity_mentions/`.
- **`gold_ai_frames`** y **`gold_ai_entity_mentions`** (vistas en
  `build_duckdb.py`): expanden `ai_frames`/`ai_entity_mentions` (una fila
  por texto único) de vuelta a una fila por INSTANCIA real de párrafo vía
  `text_hash` — el mismo criterio de "dedup como fase" de §8.8.
  Deliberadamente sin filtro por `country_code` en ningún lado, para que
  Chile aparezca solo cuando entre a la población `is_ai_prefiltered`,
  sin tocar código.

### 8.13 named_entity_match forzado a incluir — el criterio automático rechazaba casos reales de OpenAI (2026-09-04)

El usuario dudó de que solo 205/13.373 textos IA-relevantes tuvieran una
mención de entidad específica y pidió comparar contra el corpus completo.
Verificado: "openai" aparece en **59 textos únicos de todo el corpus**,
pero solo **42 estaban en la población IA-relevante** — 17 quedaban
afuera, varios financieramente sustanciales: *"the Company issued to
OpenAI OpCo, LLC... a warrant to purchase up to 160 million shares"*,
*"exclude net losses from investments in OpenAI"*. Sus `predicted_proba`
caían en 0,34-0,69 — por debajo del nuevo threshold 0,75 (§8.11 subió el
threshold óptimo al cambiar el target), pese a que `openai` SÍ está en la
lista léxica fuerte (`strong_ge1=True`).

El mecanismo para rescatar esto ya existe: `named_entity_match` (OR
determinista, §8.4) incluye `"openai"`. Pero el criterio automático de
`ai_prefilter_classify.py` lo estaba descartando por una diferencia de
**0,001 en F1 pond.** (0,946 vs 0,947) medida sobre solo 76 filas del
golden set con `named_entity_match` — una muestra demasiado chica y no
representativa: el patrón real que el override rescata (boilerplate
financiero genérico con una sola mención nombrada) casi no aparece en el
golden set, pero sí en el corpus real.

**Forzado a `True`** (ya no depende del criterio automático), documentado
en el código con la evidencia concreta. Resultado: **69 textos
rescatados** (antes 0, con el criterio automático descartándolo), 59/59
menciones de OpenAI ahora incluidas (100%, antes 42/59 = 71%). Corpus:
**13.442 textos únicos (15.945 instancias)**, subiendo levemente de
13.373/15.868. `ai_classify.py` re-corrido: solo 3 textos genuinamente
nuevos (el resto de los 69 ya estaba clasificado de corridas previas),
costo despreciable.

**Lección para el resto del pipeline**: el criterio "¿mejora la métrica
agregada?" puede fallar cuando la muestra que lo mide es chica frente al
fenómeno real que se está evaluando — ya pasó con el peso de entrenamiento
(§8.6) y ahora con este override. Un margen de 0,001 en 76 filas no es
evidencia suficiente para descartar un mecanismo cuya premisa de diseño
(un nombre propio de IA específico casi nunca es ruido) sigue siendo
válida y verificable directamente contra el corpus.

### 8.14 8-K entra al esquema de `paragraphs` (sin correr embeddings/scoring) + fix de OOM al materializar; hallazgo de AI-washing en cartas de comentario SEC (2026-09-05)

**Fetch + extracción de 8-K.** 517/517 tickers, 35.829 filings (2021-01-01
a 2026-09-01), sin errores, ~3h47m. Extracción de secciones (documento
completo como `item_key='0'`, mismo criterio pragmático que DEF 14A y la
Memoria chilena — ver `scripts/us/8k/segmenter_8k.py`) sobre 34.421
archivos completados, ~6 min. `firm_universe.parquet` verificado en 517
filas después del fetch — el fix aditivo de §incidente-firm_universe
(`_safe_write_universe`) sostuvo la corrida completa sin truncar nada.

**Resultado en `paragraphs`:** 1.623.906 filas nuevas (407.350 textos
únicos), 86% prosa (avg 110 caracteres), 14% tablas. Conteos totales por
form tras el rebuild: DEF 14A 3.679.553 (2.276.339 únicos), 10-Q
1.736.542, 8-K 1.623.906, 10-K 1.544.496, annual(CL) 874.385,
quarterly(CL) 203.210. `unique_paragraphs`: 5.138.210 filas.

**Segundo OOM al materializar `paragraphs`** (el de §8.9/8.13 con
`preserve_insertion_order=false`+`threads=4` ya no alcanzaba): el volumen
agregado de 8-K sumado a DEF 14A hizo que DuckDB planeara el `UNION ALL`
completo de las 5 ramas (10-K/10-Q/DEF14A/8-K/CL) como una sola sentencia
`CREATE TABLE`, manteniendo vivas simultáneamente las CTEs de
línea-a-línea (unnest + LAG/LEAD por `accession_number`) de TODAS las
ramas a la vez. Ni bajar threads a 2 ni fijar `memory_limit`/
`temp_directory` explícito lo resolvió — seguía OOMeando a 9-10GB en una
máquina de 16GB.

**Fix real:** materializar cada rama por separado como tabla física
(`_paragraphs_stage_10k`, `..._10q`, `..._proxy`, `..._8k`, `..._cl`) con
sentencias `CREATE TABLE` secuenciales — cada una libera su memoria de
trabajo antes de que empiece la siguiente — y solo al final unir esas
tablas ya pequeñas en `paragraphs` con un `UNION ALL BY NAME` trivial (sin
funciones de ventana). Mismo resultado exacto (la partición de
LAG/LEAD sigue siendo por `accession_number` dentro de cada forma, nunca
cruza formas), pero sin picos de memoria simultáneos. Costo total del
rebuild completo: 2:53 (antes ~2:02 sin 8-K). Las tablas de staging se
borran automáticamente al final de cada corrida (no quedan huérfanas).

**Hallazgo de cartas de comentario de la SEC — sección aparte en
`docs/analytics/01_ai_disclosure_analytics.md`** ("Cartas de comentario de
la SEC: ¿hay evidencia directa de escrutinio por AI-washing?"): de 5.057
cartas (371/517 empresas), solo **1 caso limpio de AI-washing genuino**
(Welltower, abril 2025 — lenguaje promocional "industry-leading" en
earnings call/press release sin respaldo proporcional en el 10-K,
solicitud SEC de revisar filings futuros, la empresa lo admite y agrega
disclosure). Los otros 22+ candidatos leídos manualmente son ruido
(contabilidad de segmentos ASC 280-10-50, disputas de proxy contest,
falsos positivos de substring). No es un hallazgo nulo: el mecanismo
existe y es identificable, solo es de muy baja frecuencia en este canal
específico — la carta de comentario no es un instrumento de alto volumen
para medir AI-washing sistemáticamente, pero valida que la hipótesis de
tesis no está mal planteada.

**Pendiente, no ejecutado:** embeddings/prefilter scoring sobre los
1.623.906 párrafos nuevos de 8-K (igual que DEF 14A en §8.9, requiere
confirmación explícita antes de correr el paso GPU).

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
