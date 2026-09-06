# Golden set: diseño muestral y protocolo de etiquetado

Conjunto de párrafos etiquetados por LLM que sirve de referencia para **evaluar y
ajustar el prefiltro** (`scripts/common/ai_prefilter.py`): elegir su umbral, medir
precisión y recall por categoría, y comparar variantes de anchors.

Implementación: `scripts/common/golden_set.py`.

---

## 1. Por qué existe: el problema de circularidad

La tentación es evaluar el prefiltro con una etiqueta léxica — "¿el párrafo dice
literalmente *artificial intelligence*?". No sirve, y conviene entender por qué
antes de gastar en llamadas a un LLM.

El prefiltro combina coincidencia léxica sobre una lista de términos y similitud
semántica contra anchors. Si la etiqueta se construye con la primera:

- Un scoring **híbrido** da precisión ~100% por construcción. No mide nada.
- Un scoring **semántico puro** queda castigado justo donde aporta valor:
  encontrar disclosure que no usa las palabras de la lista. Medido sobre 940.032
  párrafos, sólo el **0,38%** contiene un término fuerte, así que la etiqueta
  léxica declara negativo todo lo demás por definición.

La etiqueta tiene que venir de fuera de ambas señales. De ahí el LLM.

---

## 2. Regla dura: el muestreo no puede usar la salida del prefiltro

**El sampler no lee embeddings ni scores. Nunca.**

Si los párrafos se eligieran por su `max_semantic_score`, el conjunto de
evaluación quedaría definido por lo mismo que se quiere evaluar. El sesgo es
específico y fatal: **nunca aparecería un párrafo que es divulgación de IA y que
los anchors actuales rankean bajo**. Ese es precisamente el error que el golden
set tiene que poder detectar — y el que ya se observó al descubrir que un anchor
mal escrito ponía *"We are dependent on third-party suppliers"* en el puesto 12
del corpus.

De ahí que todo lo que estratifica sea **anterior e independiente** al prefiltro:
atributos del documento (país, forma, sección, año, tipo de contenido), atributos
del emisor (sector SIC), y un diccionario de palabras clave **congelado** en
`golden_set.py` (`SAMPLING_KEYWORDS`), deliberadamente separado de
`configs/ai_prefilter.yaml`: retocar los términos del prefiltro no debe cambiar
qué párrafos componen el conjunto de evaluación.

Ese diccionario congelado deja un sesgo residual — la divulgación de IA que no
usa ninguna de esas palabras está sub-representada en los estratos por palabra
clave. Por eso existe la **etapa 3, una muestra aleatoria pura** (§5): es el
único estrato donde ese caso puede aparecer, y es el que permite estimar cuánto
se está perdiendo.

---

## 3. Identidad del párrafo

Cada fila se identifica por la **llave natural completa**:

```
(country_code, form, accession_number, item_key, paragraph_index)
```

Verificado sobre `paragraphs`: 3.281.038 filas, 3.281.038 llaves distintas. Es la
misma llave de `ai_embed.py` y `ai_prefilter.py`, así que una etiqueta se une a
su vector y a su score sin ambigüedad.

Los cinco componentes son necesarios, y los tres primeros son los que sostienen
la unicidad cuando entren más países y tipos de informe:

- `country_code` — `accession_number` es un identificador de la SEC. Cuando entre
  Chile (`cl`), sus documentos se identifican por `rut`/`nemo` y nada garantiza
  que no colisionen.
- `form` — un mismo emisor presenta 10-K y 10-Q; `item_key` `1A` existe en ambos,
  con numeración de párrafo independiente.
- `item_key` — la numeración de párrafos reinicia por sección.

Además se guarda `text_hash` (BLAKE2b de 8 bytes). La llave identifica *la
posición*; el hash identifica *el contenido*. Si una corrida posterior de
extracción cambia el texto sin cambiar la posición, la etiqueta quedó obsoleta y
el hash lo detecta — sin él la corrupción sería silenciosa.

---

## 4. El muestreo es exploratorio, no representativo

El diseño sobre-representa a propósito los sectores donde se espera más
divulgación de IA. Sobre una muestra aleatoria simple, con ~0,4% de prevalencia,
10.000 párrafos darían unos 40 positivos: no alcanza para medir precisión por
categoría ni para decidir un umbral.

Consecuencias que hay que respetar:

1. **No estima prevalencia.** La proporción de positivos en el golden set no dice
   nada sobre la proporción en el corpus.
2. **Toda cifra a nivel de corpus requiere reponderar.** Cada fila guarda
   `stratum`, `stage` e `inclusion_weight` = (población del estrato) /
   (muestreados del estrato). Sin esos pesos, precisión y prevalencia quedan
   sesgadas hacia arriba.
3. **Las métricas por estrato son directas.** "Precisión del prefiltro dentro de
   las empresas tech" no necesita reponderación.

---

## 5. Las tres etapas

### Etapa 1 — sobremuestreo de empresas tecnológicas (4.000)

Los sectores donde se espera más presencia de IA, según la taxonomía de
`configs/us/config.yaml` (`classification.sector_groups`) resuelta por
`scripts/us/sector_map.py`:

| sector | códigos SIC (grupo mayor) |
|---|---|
| `software_it` | 73 — servicios informáticos y software |
| `computing_hardware` | 35, 36 — equipos de cómputo, electrónica, semiconductores |
| `instruments_devices` | 38 — instrumentos de medición y dispositivos médicos |

La razón es un prior sustantivo y **verificable sin el prefiltro**: se espera más
presencia de IA en esos sectores. El sampler no define su propia lista de
sectores — llama a `sector_map.sic_rule_map()`, para que retocar la taxonomía en
un solo lugar valga para todo el pipeline.

Esa capa de reglas mapea **códigos SIC**, no nombres de grupo. `industry_group`
está vacío en las 517 empresas del universo (`edgar_fetch` lo llenaba con
`company.sic_description`, atributo que edgartools renombró a `industry` en
5.55, y el `getattr` con default se lo tragaba en silencio), mientras que `sic`
está poblado 516/517. Las empresas sin SIC caen en `sector_unknown`, que es un
estrato propio y no se descarta.

Dentro de la etapa, las cuotas se reparten por sección y por presencia de palabra
clave congelada, para que el sobremuestreo no se concentre todo en Risk Factors.

### Etapa 2 — máxima variación (4.500)

Asignación balanceada por round-robin sobre el producto cartesiano de siete ejes:

| eje | fuente | valores |
|---|---|---|
| país | `paragraphs.country_code` | `us`, `cl` (cl aún sin párrafos extraídos) |
| forma | `paragraphs.form` | `10-K`, `10-Q` |
| sección | `paragraphs.item_key` | `1` Business, `1A` Risk Factors, `2` MD&A 10-Q, `7` MD&A 10-K |
| tipo de contenido | `paragraphs.content_type` | `prose`, `list`, `table` |
| año | `filing_manifest[_10q].filing_date` | 2021–2026 |
| sector | `sector_map.py` sobre `firm_universe.sic` | 12 sectores + `sector_unknown` |
| palabra clave | `SAMPLING_KEYWORDS` congelado | `strong`, `weak`, `none` |

Se recorre de la celda **más rara a la más común**, tomando hasta
`ceil(4500 / celdas_no_vacías)` de cada una y redistribuyendo el sobrante. Una
asignación proporcional dejaría `cl`, `10-Q`/`1A` y `table` en cero, y son justo
los casos donde el pipeline tiene más probabilidad de romperse.

### Etapa 3 — aleatoria pura (1.500)

Muestra aleatoria simple sobre todo el corpus, sin estratificar por nada.

Es el estrato con menos positivos esperados (~6 párrafos con IA) y el más
importante metodológicamente: es el único **libre de cualquier supuesto**, y por
lo tanto el único que puede revelar divulgación de IA que ni las palabras clave
ni los sectores tech anticipan. También es el que da un estimador insesgado de
prevalencia, con el que se reponderan las otras dos etapas.

Las cuotas redondean, así que el total efectivo queda cerca del pedido pero no
exacto: con los valores por defecto la muestra sale de 9.900, no de 10.000.

### Deduplicación previa

El boilerplate se repite entre filings — la misma frase apareció cinco veces en
el top semántico. Antes de asignar cuotas se deduplica por `text_hash`,
conservando una ocurrencia y anotando `duplicate_count`. Sin esto se paga varias
veces por la misma etiqueta y se infla el acuerdo.

---

## 6. Etiquetado

- **Modelo**: `gemini-3.8-flash` vía `pydantic-ai` con salida estructurada
  (Pydantic). Configurable con `--judge-model`; queda guardado en cada fila. El
  modelo se construye explícito (`GoogleModel(...)`) y no por string
  `"google-gla:<nombre>"`: la lista de nombres conocidos de pydantic-ai va por
  detrás de la API y rechaza modelos que la cuenta sí tiene, `gemini-3.8-flash`
  entre ellos.
- **Prompt**: versionado (`PROMPT_VERSION`). El par
  `(judge_model, prompt_version)` define una **población de etiquetas**: filas de
  otro modelo u otro prompt no se mezclan, se reportan aparte.
- **Salida estructurada**, diseñada para evaluar el prefiltro:
  - `is_ai_disclosure` (bool) — la etiqueta binaria contra la que se mide.
  - `relevance` — `none` / `incidental` / `substantive`. La distinción entre
    mención de pasada y disclosure sustantivo define si el umbral debe ser
    agresivo o conservador.
  - `categories` — las siete de `configs/ai_prefilter.yaml`, para medir
    `best_semantic_anchor` contra la categoría real.
  - `evidence_quote` + `evidence_verbatim` — cita literal, verificada contra el
    texto del párrafo. Si el juez la inventa, la fila queda marcada.
  - `mentions_ai_explicitly` — permite separar "IA nombrada" de "IA descrita sin
    nombrarla", que es la pregunta de fondo sobre el aporte del prefiltro
    semántico.
  - `confidence` y `reasoning` — para filtrar por confianza y revisar a mano.

### Costo y resiliencia

10.000 párrafos es caro en tiempo y llamadas, así que nada se pierde:

- Parte parquet cada `--part-rows` filas (2.000 por defecto), escrita
  atómicamente (`.partial` y luego rename).
- Un fallo de API en un ítem se guarda en la fila (`error`) y la corrida sigue.
- SIGINT/SIGTERM hacen flush de lo acumulado antes de salir.
- La corrida siguiente es aditiva: verifica lo que hay y etiqueta sólo lo que
  falta.

---

## 7. Aditividad entre sesiones

> **Un solo juez por población, y hacerlo cumplir (2026-09-06).** La regla de
> abajo —"(judge_model, prompt_version) define una población"— se violó en la
> práctica: el prefiltro se entrenó con la unión de gemini-3.8-flash y
> qwen3.7-flash, repartidos por estrato (gemini: stage1 + parte de stage2;
> qwen: resto de stage2 + todo stage3_random). Corregido re-etiquetando todo
> con qwen (`label --coverage-judge current`, que trata como pendiente lo
> etiquetado por otro juez) y filtrando por `judge_model` en cada consumidor.
> Las etiquetas viejas NO se borran: son la única forma de medir acuerdo entre
> jueces, y dieron κ=0,87 sobre 6.038 filas pareadas
> (`scripts/verif/judge_agreement.py`, `prefilter_evaluation.md` §8.15).


1. Al arrancar se validan las partes existentes (que abran, que traigan la llave,
   que coincidan `judge_model` y `prompt_version`). Lo que no califica se
   **reporta y se ignora, nunca se borra**.
2. Lo pendiente sale de un `NOT EXISTS` contra las llaves ya etiquetadas.
3. Cada fila guarda `session_id`, `labeled_at`, `judge_model`, `prompt_version`,
   `sampling_version` y `text_hash`.

Permite crecer en varias sesiones, retomar tras un corte, y sumar Chile más
adelante sin volver a pagar lo ya etiquetado. Un cambio de modelo no contamina:
crea una población nueva e identificable, que además sirve para medir acuerdo
entre jueces.

---

## 8. Cómo se usa para evaluar el prefiltro

Uniendo el golden set a `ai_prefilter_scores` por la llave:

- **Curva precisión-recall** de `max_semantic_score` contra `is_ai_disclosure`,
  ponderada por `inclusion_weight`, para elegir el umbral.
- **Comparación de scorings** — léxico solo, semántico solo, híbrido — sobre la
  misma etiqueta independiente. Es la comparación que la etiqueta léxica no
  permite hacer (§1).
- **Ranking por categoría**: precisión promedio (AP, área bajo precisión-recall)
  de cada `score_<categoría>` contra "esa categoría está en `categories`", junto
  con el **lift** sobre la tasa base y `P@k`. Dos advertencias, ambas verificadas
  sobre las primeras 4.000 etiquetas:

  1. **No usar la precisión de `best_semantic_anchor`.** El juez asigna 2,1
     categorías por párrafo y el argmax elige una sola, así que esa métrica mide
     tasas base: `ai_governance` acierta el argmax en 4% de los casos porque
     pierde siempre contra `ai_risk` — en filings "board oversight of AI risk" es
     honestamente ambas —, mientras `ai_capability` gana el 93% de sus argmax
     siendo de las peores rankeando.
  2. **No usar ROC-AUC solo.** Las categorías van de 21,9% a 0,9% de tasa base y
     el ROC-AUC se infla con desbalance: `ai_governance` da 0,924 de AUC pero
     0,246 de AP. El lift la rescata (16,4x, el más alto de las siete), que es la
     lectura correcta para una categoría rara.

  Mirando AP y lift juntos aparece la categoría realmente floja, invisible bajo
  las otras dos métricas: `ai_outcome`, con AP 0,135 sobre base 4,475% (lift
  3,0x) y 18% de precisión en el top-50 — sus anclas hablan de productividad,
  costos e ingresos, el lenguaje de medio MD&A tenga IA o no.

- **Diagnóstico de anchors**: los falsos positivos apuntan al anchor que los
  atrajo. Así se detectó que *"The company relies on third-party artificial
  intelligence providers or models."* capturaba boilerplate de proveedores.
- **Recall fuera del radar**: los positivos de la etapa 3 con
  `mentions_ai_explicitly = false` son la medida directa de lo que un filtro
  puramente léxico pierde.
