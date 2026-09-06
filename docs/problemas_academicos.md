# Registro de problemas académicos: qué está cerrado y qué no

Lista viva de los defectos que una revisión metodológica del proyecto encontró
(sesión 2026-09-05/06), con su estado actual y qué haría falta para cerrar cada
uno. **Sirve como checklist de defensa**: cada fila es algo que un jurado puede
preguntar, y la columna de estado dice si hay respuesta.

Regla de este documento: un problema sólo pasa a "cerrado" con una MEDICIÓN que
lo demuestre, no con un argumento. Los que se cerraron llevan el número y dónde
está el script que lo produce.

## Resumen

| # | Problema | Estado |
|---|---|---|
| 1 | Ninguna medida validada contra anotación humana | **ABIERTO** — el hueco más grande |
| 2 | El golden set mezclaba dos jueces LLM | **CERRADO** |
| 3 | El prefiltro no estaba validado en DEF 14A / 8-K | **CERRADO (medido) + mejorado** |
| 4 | El score de washing ignoraba mezcla documental y dependencia | **CERRADO** |
| 5 | El único caso externo (Welltower) no lo detecta el score | **ABIERTO** — falta el corpus de earnings calls |
| 6 | Marco muestral y potencia (sólo large caps) | **DESCARTADO por decisión del autor** |
| 7 | `gold_ai_frames` acumulaba la unión histórica de despliegues | **CERRADO** |
| 8 | El lado contable/mercado no tenía código que lo generara | **CERRADO** |
| 9 | El "DiD de tendencia" SEC/DeepSeek no es un DiD | **ABIERTO** |
| 10 | ROIC−WACC mezcla valor libro y de mercado | **PARCIAL** (ERP corregido; libro/mercado no) |
| 11 | K-means con silhouette 0,15 sostiene 4 categorías | **ABIERTO** |
| 12 | El panel empresa-año tiene entrada endógena | **ABIERTO** |

---

## 1. Ninguna medida está validada contra anotación humana — ABIERTO

Todo el constructo —`rhetoric_promotional`, `specificity_*`, `temporal`,
`concepts`— sale de un LLM (`qwen3.7-flash`) sin un solo caso anotado a mano.

Lo único que se midió es acuerdo **entre LLMs**, y sólo en la etapa del
prefiltro: κ=0,866 entre gemini-3.8-flash y qwen3.7-flash sobre 6.038 párrafos
con doble lectura (`scripts/verif/judge_agreement.py`). **La extracción de
frames no tiene ni eso.** Acuerdo entre dos LLMs tampoco es validez: pueden
compartir el mismo sesgo.

**Qué lo cerraría**: anotar a mano 300-500 frames estratificados por formulario
y por etiqueta, reportar κ humano-LLM por dimensión, y declarar en la tesis la
confiabilidad de cada medida que se usa en un resultado. Sin esto, todo número
que dependa de `rhetoric_promotional` —incluido el score de AI-washing entero—
descansa en una medición no validada.

## 2. El golden set mezclaba dos jueces — CERRADO

`gemini-3.8-flash` etiquetó `stage1` completo y parte de `stage2`;
`qwen3.7-flash` el resto de `stage2` y **todo `stage3_random`**, el estrato que
ancla la reponderación. El prefiltro se ajustaba con la unión, sin ninguna fila
en común entre jueces, así que el acuerdo ni siquiera era medible.

**Cerrado así**: re-etiquetado completo con un solo juez (9.900 etiquetas), las
de gemini conservadas en disco, filtro por `judge_model` en todos los
consumidores, y el manifiesto del despliegue ahora declara con qué juez se
entrenó. Ver `prefilter_evaluation.md` §8.15.

**Con un matiz honesto**: al medirlo, los jueces coincidían (κ=0,87), así que el
defecto era estructural pero chico. El cambio de población fue del 2%
(solapamiento 0,978 Jaccard). Lo que compró fue provenance y la medición de κ,
no datos distintos.

## 3. El prefiltro no estaba validado fuera de 10-K/10-Q — CERRADO como medición, mejorado como desempeño

El golden set es 100% 10-K y 10-Q, y el mismo modelo se aplicaba a DEF 14A y
8-K, que aportan el 24% de los frames y sostienen el hallazgo #8 de
`analytics/01_...md` (16,4% de frames promocionales en proxy contra 7,2% en
10-K).

**Medido** con 1.500 etiquetas nuevas de esos formularios que nunca entran a
ningún ajuste (`prefilter_form_validation.py`), y **mejorado** con el modelo v2
(`ai_prefilter_deploy.py`, §8.16):

| | v1 desplegado | v2 (recall, β=2) |
|---|---:|---:|
| Recall DEF 14A | 0,844 | **0,971** |
| Recall 8-K | 0,812 | **0,915** |
| Recall 10-K/10-Q | 0,981 | 0,987 |
| Precisión holdout | 0,678 | 0,702 |

**Por qué el recall es la métrica que importa acá**: río abajo el juez LLM
vuelve a filtrar, así que un falso positivo del prefiltro cuesta una llamada y
se descarta —las tablas de análisis filtran por `has_frame`—, mientras que un
falso negativo no reaparece nunca. La población analizada depende del recall,
no de la precisión.

**Qué queda abierto**: con el recall casi igualado entre formularios (0,97-0,99)
la comparación #8 es mucho más defendible que antes, pero **la tasa de error del
propio juez de frames por formulario sigue sin medirse** (problema 1). Hasta
entonces, el contraste proxy vs. 10-K se reporta con el caveat.

## 4. El score de AI-washing ignoraba la mezcla documental — CERRADO

La DEF 14A tiene 16,4% de frames promocionales contra 7,2% del 10-K, así que una
empresa con proxy extenso parecía promocional por composición documental. La
cola de washing tenía 41% de frames de proxy contra 23% del resto del corpus, y
9 de las 22 empresas detectadas salían de ahí.

**Cerrado así** (`washing_score.py`, `09_washing_score.md`): unidad = frame
único, efectos fijos de formulario en el logit de nivel frame, conteo nulo
Poisson-binomial exacto, y corrección por dependencia intra-documento (ICC
medido por ANOVA sobre los residuos). Más una batería de validación
(`validate_washing_score.py`): placebo (0 falsos positivos en 5 permutaciones),
split-half (Spearman 0,52), persistencia entre épocas (0,41), grilla de
especificaciones (sólo CDNS, CRWD y PANW sobreviven a todas) y casos externos.

## 5. El constructo del score no es el del regulador — ABIERTO

Welltower, la única empresa del corpus con una carta de comentario de la SEC
sobre sus afirmaciones de IA, **no la marca el score** (1 frame promocional de
43, percentil 64). No es un bug: la SEC no le objetó el 10-K por promocional, le
objetó decir "industry-leading" en el earnings call y el press release **sin
respaldo proporcional en el 10-K**. El score mide exceso promocional DENTRO del
filing; el regulador persigue una BRECHA ENTRE CANALES.

**Qué lo cerraría**: construir el corpus de transcripciones de earnings calls
(`document_expansion_plan.md`, fase 4) y medir el mismo score por canal, con la
diferencia por empresa como variable. Es el paso que convierte el instrumento en
uno que mide lo que la tesis dice medir.

## 6. Marco muestral y potencia — DESCARTADO por decisión del autor

El registro del hecho, sin la recomendación: el marco es S&P 500 congelado al
2021-12-31 más 122 empresas elegidas a mano, todas large caps; el AI-washing
que la SEC efectivamente persiguió ocurrió en micro-caps (Presto, Kubient) que
el marco excluye. Y el score sólo tiene potencia en empresas con 100+ frames:
de 210 empresas con ≤25 frames, el test no rechaza en ninguna.

El autor descartó tratarlo como problema. Queda anotado para que la limitación
esté escrita, no para reabrirlo.

## 7. `gold_ai_frames` acumulaba despliegues viejos — CERRADO

La vista unía frames a párrafos por `text_hash` sin filtrar por la población
vigente, y `ai_classify.py` nunca borra: **20.899 textos tenían frames y 1.332
(6,4%) ya no pertenecían a la población marcada** por el modelo desplegado. Cada
cifra de `docs/analytics/` dependía del orden histórico de los despliegues.
Corregido con un JOIN contra la población vigente en `build_duckdb.py`.

## 8. El lado contable/mercado no tenía código — CERRADO

Cinco parquets (`firm_year_financials`, `..._ratios`, `..._market_factors`,
`..._filing_returns`, `..._roic_wacc`) existían como datos sin generador: el
script original nunca se versionó y se perdió. Escritos y verificados contra las
copias archivadas (retornos idénticos a 1e-6, ratios con correlación 0,98-1,00);
ver `10_builders_y_recalculo.md`. Todo corre con `make analytics`.

## 9. El "DiD de tendencia" SEC/DeepSeek no es un DiD — ABIERTO

`01_...md` compara empresas "vagas" contra "específicas" antes y después de
marzo 2024 con una regresión segmentada sobre medias trimestrales por grupo.
Problemas, en orden de gravedad:

1. **Reversión a la media por construcción**: los grupos se definen por valores
   extremos de las MISMAS métricas que después se miden.
2. **Composición cambiante**: la media trimestral de cada grupo se calcula sobre
   las empresas que divulgaron ese trimestre, y la difusión de la IA 2023-2025
   cambió quiénes son.
3. Sin efectos fijos de empresa ni errores estándar clusterizados (ya anotado en
   el doc).
4. Sin test ni gráfico de tendencias paralelas.
5. Cuatro métricas × dos cortes sin corrección por comparaciones múltiples, en
   un proyecto que aplica FDR en otras secciones.

**Qué lo cerraría**: panel empresa-trimestre con efectos fijos de empresa y de
tiempo, SE clusterizados por empresa, event-study con leads y lags en vez de dos
tramos, y el mismo FDR que se usa en `05_...md`.

## 10. ROIC−WACC mezcla libro y mercado — PARCIAL

El ERP ya está corregido (geométrico 6,48% en vez del aritmético 8,20%, que
inflaba las diferencias de WACC en proporción al beta). Sigue en pie que el ROIC
usa capital invertido a valor LIBRO mientras los ponderadores del WACC usan
market cap: el spread queda sesgado con el market-to-book, que es una dimensión
donde los segmentos difieren.

## 11. K-means con silhouette 0,15 — ABIERTO

`06_...md` mide silhouette entre 0,150 y 0,162 para todo k probado y aun así los
clusters 0 y 1 se reportan como grupos distintos en `07_...md` y `08_...md`. Con
esa separación lo defendible es un score continuo (como el 09) o k=2.

## 12. Entrada endógena al panel empresa-año — ABIERTO

Una empresa-año entra al panel sólo si tiene ≥3 frames ese año, así que las
series temporales y la matriz de transición mezclan cambio de discurso, cambio
de mezcla documental y ruido de denominador chico. La fuga C→D de 35,3% es la
afirmación más expuesta.

---

## Qué NO arregla el trabajo del prefiltro

Vale decirlo explícito, porque es fácil confundir "mejoré la medición" con
"resolví el problema de la tesis": el prefiltro decide **qué párrafos se miran**.
No dice nada sobre si las etiquetas que salen de mirarlos son válidas (1), si el
constructo de washing es el correcto (5), ni sobre la econometría de los
resultados (9-12). El único de los tres problemas de primer orden que el
prefiltro toca es el 3.
