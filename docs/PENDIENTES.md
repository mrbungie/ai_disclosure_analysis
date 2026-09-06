# Qué falta, y cómo seguir en otra máquina

Estado al 2026-09-06. Todo lo que está acá se puede retomar en cualquier equipo:
el código está en git y los datos en B2.

## Cómo levantar el proyecto en el laptop

```bash
git pull
uv sync                                    # o: uv pip install -e .
cp .env.example .env                       # y completar las claves (ver abajo)
scripts/common/sync_data_b2.sh pull        # baja data/ entero (~decenas de GB)
```

Bajar sólo lo necesario para seguir, en vez del bucket completo:

```bash
scripts/common/sync_data_b2.sh pull --path interim/ai_classify
scripts/common/sync_data_b2.sh pull --path interim/prefilter_predictions_unique
scripts/common/sync_data_b2.sh pull --path interim/manifests
scripts/common/sync_data_b2.sh pull --path processed/clusters
```

Claves que hacen falta en `.env`: `OPENROUTER_API_KEY` (clasificación),
`B2_KEY_ID` / `B2_APPLICATION_KEY` / `B2_BUCKET` / `RCLONE_REMOTE_NAME` (datos).

**Qué necesita GPU y qué no.** Sólo dos pasos la usan: embeddings
(`ai_embed.py`) y scoring de anchors (`ai_prefilter.py`). **Los dos ya están
corridos para todo el corpus**, así que de acá en adelante todo corre en CPU —
la clasificación es I/O contra OpenRouter y el analytics es pandas y sklearn. No
hace falta una máquina con GPU para seguir.

## 1. Lo único que quedó a medias

**Clasificar los frames de earnings calls.** Aditivo e idempotente: se corta y se
retoma sin perder nada.

```bash
uv run python scripts/common/ai_classify.py --concurrency 20
```

| | |
|---|---|
| pendientes | **6.843 textos** de 8.968 (el resto de los formularios está completo) |
| costo | ~US$0,70 |
| tiempo | ~1,5 h — el límite es el rate-limit de OpenRouter sobre `qwen3.7-flash`, no la máquina |

Devuelve ~40% de errores 429 por corrida; hay que correrlo varias veces hasta
que diga `Pendientes en total: 0`. Un `for i in $(seq 1 10)` alrededor alcanza.

Cuando termine:

```bash
uv run python scripts/common/build_duckdb.py --with-text-tables   # refresca vistas
uv run python scripts/analytics/earnings_calls_analysis.py        # el canal
make analytics                                                     # todo lo demás
```

## 2. Análisis que quedan pendientes

### 2.1 Comparación formal entre canales (lo que más valor agrega)

Es el diseño que puede separar el efecto del regulador del boom de IA, y el que
corresponde al mecanismo del caso Welltower (promocional en la call, nada
proporcional en el 10-K):

```
promocional[i, canal, t] = a[i,t] + b · (post[t] × es_filing[canal]) + e
```

Con efectos fijos **empresa × trimestre**, el boom se absorbe entero: se compara
a la misma empresa, el mismo trimestre, en dos canales con distinta exposición
legal. Requiere los frames de calls del punto 1. Razonamiento completo en
`docs/pregunta_identificacion_sec.md`; el supuesto (brechas previas paralelas) es
testeable.

### 2.2 Reescribir las tablas de los docs 02, 03, 04, 07 y 08

Siguen mostrando cifras de corridas anteriores. Los deltas están en
`docs/analytics/10_builders_y_recalculo.md`; los números nuevos salen de
`scripts/analytics/report_crosscheck_stats.py`.

### 2.3 Decidir `.mean()` vs `.median()` en `build_firm_panels.py`

`07_...md` documenta mediana y el script hace media. Hay que elegir una.

## 3. Pendientes metodológicos

El registro completo está en `docs/problemas_academicos.md` (12 ítems con su
estado). Los que siguen abiertos:

| # | qué falta | costo |
|---|---|---|
| **1** | **Validar las etiquetas contra anotación humana.** Anotar a mano 300-500 frames y reportar κ humano-LLM por dimensión. Hoy sólo hay acuerdo entre dos LLMs (κ=0,87) y **sólo en el prefiltro**; la extracción de frames no tiene ninguna validación. | tiempo humano |
| 5 | El score de washing no detecta el caso Welltower porque mide exceso *dentro* del filing y el regulador persigue la brecha *entre canales*. Se cierra con 2.1. | incluido en 2.1 |
| 12 | Entrada endógena al panel empresa-año (una empresa entra sólo si tuvo ≥3 frames ese año). | diseño |

**El ítem 1 es el bloqueante real de la tesis.** Todo resultado descansa en
`rhetoric_promotional` y `specificity_*`, que nunca se compararon contra un
humano.

## 4. Lo que ya está terminado (para no rehacerlo)

- **Prefiltro v2** desplegado: árboles sobre 39 señales (párrafo + forma del
  texto + oración), umbral 0,30 elegido sobre **tres** muestras de validación
  (10-K/10-Q, DEF 14A/8-K, earnings calls). Recall 0,98 / 0,94 / 0,98.
- **Golden set de un solo juez** (qwen3.7-flash, 9.900 etiquetas) + κ=0,866
  medido contra gemini sobre 6.038 filas pareadas.
- **Validación por canal**: 1.500 etiquetas de proxy/8-K + 800 de calls.
- **Segmentación** (`11_segmentacion.md`): 3 segmentos elegidos por estabilidad
  bootstrap, con persistencia año a año de 71,8%.
- **Grilla voz × conducta** (`12_grilla_voz_conducta.md`): 9 celdas, con
  `confianza_celda` por empresa.
- **Score de AI-washing** (`09_washing_score.md`): 8 empresas con FDR 5%, con
  placebo, split-half y grilla de especificaciones.
- **Shocks** (`13_shocks.md`): nulo bien medido para el evento SEC; DeepSeek
  fuera de alcance con ese diseño.
- **Builders financieros** versionados (`10_builders_y_recalculo.md`) — antes
  eran datos sin código.

## 5. Costos, para dimensionar

| | |
|---|---|
| OpenRouter gastado | US$3,2 de 20 acreditados |
| Falta para terminar la clasificación | ~US$0,70 |
| Instancia vast.ai (RTX 5090) | **US$0,45/h = US$10,9 por día** |

La GPU ya no se necesita: cuesta 15 veces más por día que todo lo que falta
gastar en LLM. Si el laptop puede correr Python, conviene apagarla.
