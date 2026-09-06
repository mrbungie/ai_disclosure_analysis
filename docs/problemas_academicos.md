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
| 9 | El "DiD de tendencia" SEC/DeepSeek estaba mal construido | **CERRADO** — rehecho; el estimando original era mecánico, el nuevo es un cambio diferencial válido (no causal) |
| 10 | ROIC−WACC mezclaba valor libro y de mercado | **CERRADO** — y medido: no cambiaba el ordenamiento |
| 11 | K-means con silhouette 0,15 sostiene 4 categorías | **CERRADO** — reemplazado por 3 segmentos estables (`11_segmentacion.md`) |
| 12 | El panel empresa-año tiene entrada endógena | **CERRADO** en el modo extensivo (02-05, 08, 11-14); queda en los arquetipos de 07/08 y el test binomial de 09 |

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

## 9. El "DiD de tendencia" SEC/DeepSeek estaba mal construido — CERRADO

**Encuadre primero, porque la primera versión de esta ficha lo tenía mal.** La
propuesta de tesis plantea las técnicas cuasi-causales como posibilidad para las
preguntas extendidas ("quasi-causal techniques like DiD **may** be used"), no
como el diseño central; el núcleo es medición → arquetipos → evolución. Evaluar
este análisis como si la tesis dependiera de identificar el ATT del enforcement
fue un error de alcance mío. Hay tres niveles distintos y hay que no
confundirlos:

| afirmación | exigencia |
|---|---|
| "las empresas se ven así y se agrupan así" | ninguna |
| "esto evolucionó de esta forma" | ninguna |
| "el grupo A cambió distinto que el B después del evento" | tratamiento exógeno al outcome, FE, SE clusterizados, pre-tendencias |
| "el enforcement lo causó" | además, variación que separe el evento de todo lo demás que pasó esa fecha |

Lo que sigue aplica al tercer nivel, que es el que la propuesta pide.

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

**Hecho** (`scripts/analytics/sec_event_study.py`): panel empresa-trimestre,
efectos fijos de empresa y de trimestre, SE clusterizados por empresa, y un
coeficiente por trimestre relativo al evento (los previos SON el test de
tendencias paralelas).

**Resultado: los cuatro outcomes fallan el test.** La diferencia entre grupos
está presente y es significativa seis trimestres ANTES de marzo 2024 y no
cambia después — es una diferencia permanente entre empresas, que es lo
esperable porque los grupos se definieron por esas mismas métricas. El
"quiebre de tendencia" que reportaba `01_...md` era del diseño, no de los
datos, y esa sección quedó retirada.

**Segunda vuelta: el diseño SÍ se puede identificar, cambiando el tratamiento.**
El defecto de fondo era definir el grupo por el nivel pre-evento de la misma
variable que después se mide. Con **exposición** (volumen de frames de IA
pre-2023, que no es el outcome), panel de los cuatro formularios con controles
de composición documental, panel balanceado y el test conjunto de los pre:

| outcome | tendencias paralelas | efecto post |
|---|---|---|
| `promotional_rate` | **pasa** (p=0,32) | **+8,4 p.p.** (p<0,001) |
| `risk_share` | **pasa** (p=0,39) | −0,7 p.p. (nulo, p=0,71) |
| `specificity_index` | falla (p<0,001) | — |
| `hypothetical_share` | falla (p<0,001) | — |

Las empresas más expuestas a IA se volvieron MÁS promocionales después de
2024-Q1, no menos. **Pero el evento es común en el tiempo**: ese coeficiente
recoge todo lo que le pasó a las empresas expuestas a IA en esa fecha, y el boom
de IA generativa es la alternativa obvia. Atribuirlo a la SEC requiere variación
que separe ambas cosas — por ejemplo, empresas efectivamente contactadas por el
regulador (hay 1) o una comparación entre jurisdicciones.

Panel: 419 observaciones, 51 empresas.

## 10. ROIC−WACC mezclaba libro y mercado — CERRADO

El ROIC se calculaba sobre capital invertido CONTABLE y el WACC se ponderaba con
MARKET CAP: dividir con una regla y ponderar con otra sesga el spread con el
market-to-book, que es una dimensión donde los segmentos difieren.

`build_roic_wacc.py` ahora pondera el WACC con valores de libro (coherente con
el denominador del ROIC) y guarda la versión de mercado aparte
(`wacc_market`, `roic_minus_wacc_market`).

**Y midiendo el efecto: la corrección mueve el NIVEL, no el ordenamiento.**
Mediana del spread 3,98% → 4,91%, pero la correlación entre las dos versiones es
**0,999** sobre 1.907 empresas-año. Cualquier comparación ENTRE segmentos era en
la práctica insensible a esto. Se corrigió igual porque la definición ahora es
coherente y la cobertura sube (spread disponible 66% → 73%), pero el problema
era menor de lo que parecía. El ERP ya estaba corregido antes (geométrico 6,48%
en vez del aritmético 8,20%).

## 11. K-means con silhouette 0,15 — CERRADO, con reemplazo

`scripts/analytics/cluster_diagnostics.py` mide lo que faltaba:

| prueba | resultado |
|---|---|
| Confiabilidad de `specificity_index` a nivel empresa | **0,000** (varianza observada < varianza de muestreo) |
| `promotional_rate` / `quantified_rate` | 0,47 / 0,51 |
| Estabilidad bootstrap k=4 (Jaccard) | **0,53** — no reproducible |
| Estabilidad bootstrap k=2 | **0,81** — sólido |
| PCA: 2 factores | 49% de la varianza (5 componentes para 83%) |

Las 9 features **no son booleanos crudos**: son medias por empresa de banderas
booleanas, o sea tasas en [0,1] estandarizadas. El problema no es el tipo de
dato, es que k-means trata una tasa estimada con 5 frames como igual de
confiable que una estimada con 500 — el mismo defecto que ya había hundido la
definición de washing por clusters (`09_...md`).

**Reemplazo, ya en el pipeline** (`firm_voice_scores.parquet`, producido por
`build_firm_clusters.py`): tasas con encogimiento empírico-Bayes hacia la media
global, dos factores continuos, y una partición binaria estable —
`risk_hypothetical` (229 empresas) vs. `deployment_asserted` (265). Los cuatro
arquetipos se siguen calculando por compatibilidad, marcados como no
reproducibles.

**Y el cruce que faltaba** (`voice_behavior_factors.py`): dos factores de voz
solos no dicen nada sobre la pregunta de la tesis, así que se cruzan con los
factores de COMPORTAMIENTO por correlación canónica. **Primera correlación
canónica 0,930** (bootstrap 0,90-0,92): los dos bloques comparten ~86% de la
varianza y el comportamiento explica 55,7% del eje de voz. O sea: **voz y
conducta declarada no son dos ejes, son casi el mismo**, y lo que queda para
"washing" es el residuo — que ahora es una variable continua por empresa
(`firm_voice_behavior_factors.parquet`) en vez de un cruce de etiquetas.

## 12. Entrada endógena al panel empresa-año — MITIGADO

Una empresa-año entra al panel sólo si tiene ≥3 frames ese año, así que las
series temporales y la matriz de transición mezclan cambio de discurso, cambio
de mezcla documental y ruido de denominador chico. La fuga C→D de 35,3% es la
afirmación más expuesta.

---

**Estado.** El modo de análisis final es el margen extensivo
(`10_builders_y_recalculo.md`): `scripts/analytics/ai_intensity.py` arma la
tabla de todos los documentos con sus párrafos y conteos de frames, cero
incluido, y sobre ella corren el cruce financiero (`firm_year_master_v2`,
2.964 empresas-año), los shocks, la brecha entre canales y el score de
washing en intensidad; segmentos (`11`) y grilla (`12`) se construyen sobre
las 510 empresas con la intensidad como dimensión. Queda por construcción en
los arquetipos de voz de `07`/`08` y en el test binomial de `09`, que
necesitan frames para existir.

## Qué NO arregla el trabajo del prefiltro

Vale decirlo explícito, porque es fácil confundir "mejoré la medición" con
"resolví el problema de la tesis": el prefiltro decide **qué párrafos se miran**.
No dice nada sobre si las etiquetas que salen de mirarlos son válidas (1), si el
constructo de washing es el correcto (5), ni sobre la econometría de los
resultados (9-12). El único de los tres problemas de primer orden que el
prefiltro toca es el 3.
