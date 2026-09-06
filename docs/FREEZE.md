# Corpus e instrumento congelados

A partir de esta versión no se cambia el instrumento salvo que la validación
humana (`ui-validator/`) revele un problema grave. Todo análisis de
`docs/analytics/` se produce sobre esto.

| componente | versión congelada |
|---|---|
| corpus EE.UU. | 10-K, 10-Q, DEF 14A, 8-K (2020-11 → 2026-09) y earnings calls (ejercicios 2021-2025, última transcripción a mediados de 2025); 55.981 documentos, 8.354.547 párrafos puntuables, 4.868.804 textos únicos — `docs/analytics/00_funnel_del_corpus.md` |
| embeddings | `BAAI/bge-m3`, fp16, 1024 dims, por texto único (`ai_embed.py`) |
| prefiltro | v2: gradient boosting sobre 39 señales (párrafo + forma del texto + oración), **umbral 0,17**, `run=20260906T160624Z`, 30.280 textos únicos marcados (`ai_prefilter_deploy.py --threshold 0.17`) |
| esquema de frames | `docs/classification_model.md`; `ParagraphExtraction` / `AIFrame` en `ai_classify.py` |
| juez | `qwen/qwen3.7-flash` vía OpenRouter, `prompt_version = v1`, pydantic-ai con `retries=2` |
| extracción | 30.579 textos únicos clasificados, 26.469 con ≥1 frame, 43.366 frames únicos → 46.215 en documentos; partes en `data/interim/ai_classify/` (78 sesiones, todas en B2) |
| `gold_ai_frames` | vista de `build_duckdb.py --with-text-tables`: frames del texto único unidos a TODAS sus instancias, población del último despliegue del prefiltro |
| segmentación | `build_segments.py`, k=3 por estabilidad + "sin IA" por regla, semilla 42; no se reclusteriza |
| grilla | `build_voice_behavior_grid.py`, terciles sobre tasas encogidas, semilla 42 |
| financieros | XBRL (`data/raw/xbrl_facts/us/`), precios y factores FF3; ERP geométrico 6,48% |
| código | commit de git que introduce este archivo y posteriores etiquetados como `analytics-final` |

Qué puede cambiar sin romper el congelamiento: figuras, redacción de los
docs, tablas regeneradas con los mismos scripts. Qué no: umbral del
prefiltro, prompt o modelo del juez, esquema, k de la segmentación, cortes de
la grilla, definición de intensidad.

Validación humana pendiente (`docs/PENDIENTES.md` §1): 400-500 párrafos
estratificados por formulario, año, score del prefiltro, promocional y
realizado; se reporta relevancia, presencia de cada dimensión, `temporal`,
`rhetoric_promotional`, `specificity` y los cinco bloques de conceptos. Si un
campo sale mal, se colapsa (p. ej. realizado / no realizado) y se re-corren
los análisis con el campo colapsado; eso es lo único que reabre el
congelamiento.
