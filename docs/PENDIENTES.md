# Estado del proyecto y qué falta

Todo se retoma en cualquier equipo: el código está en git y los datos en B2.

## Cómo levantar el proyecto

```bash
git pull
cp .env.example .env                       # y completar las claves (ver abajo)
scripts/common/sync_data_b2.sh pull        # baja data/ entero (~decenas de GB)
```

Bajar sólo lo necesario para analytics, en vez del bucket completo:

```bash
scripts/common/sync_data_b2.sh pull --path interim/ai_classify
scripts/common/sync_data_b2.sh pull --path interim/prefilter_predictions_unique
scripts/common/sync_data_b2.sh pull --path interim/manifests
scripts/common/sync_data_b2.sh pull --path processed/clusters
scripts/common/sync_data_b2.sh pull --path raw/xbrl_facts
```

Claves en `.env`: `OPENROUTER_API_KEY` (clasificación), `B2_KEY_ID` /
`B2_APPLICATION_KEY` / `B2_BUCKET` / `RCLONE_REMOTE_NAME` (datos).

**Entorno.** `uv sync` falla por un conflicto de resolución entre
`sentence-transformers==6.0.1` y el extra `pdf-vlm-mineru`; todo corre con
`uv run --frozen --no-sync python ...` sobre el `.venv`. `statsmodels` y
`matplotlib` no están declarados en `pyproject.toml` pero `shock_*.py` y
`channel_gap_analysis.py` los importan
(`uv pip install --python .venv/bin/python statsmodels matplotlib`).

**GPU.** Sólo embeddings (`ai_embed.py`) y scoring de anchors
(`ai_prefilter.py`) la usan, y los dos están corridos para todo el corpus.
Todo lo demás es CPU: clasificación (I/O contra OpenRouter) y analytics
(pandas, sklearn, statsmodels).

## Corpus final

| | |
|---|---|
| prefiltro vigente | v2, árboles, umbral 0,17 (`run=20260906T160624Z`), 30.280 textos únicos marcados |
| frames en formularios SEC | 29.945 (10-K 17.758, DEF 14A 7.284, 10-Q 4.590, 8-K 313) |
| frames en earnings calls | 16.270, 403 empresas, 2.709 transcripciones |
| pendientes de clasificar | 0 |
| panel empresa-año (02-08) | 1.426 filas, 460 empresas, 25.355 frames |

Regenerar todo desde cero, en orden:

```bash
uv run --frozen --no-sync python scripts/common/ai_prefilter_deploy.py --threshold 0.17
uv run --frozen --no-sync python scripts/common/ai_classify.py --concurrency 20   # repetir hasta "Pendientes en total: 0"
uv run --frozen --no-sync python scripts/common/build_duckdb.py --with-text-tables
uv run --frozen --no-sync python scripts/analytics/earnings_calls_analysis.py
make analytics
for s in report_crosscheck_stats.py validate_washing_score.py voice_behavior_factors.py \
         behavior_block_eval.py cluster_diagnostics.py washing_hierarchical.py channel_gap_analysis.py; do
  .venv/bin/python scripts/analytics/$s; done
.venv/bin/python scripts/analytics/report_crosscheck_stats.py --json data/processed/clusters/crosscheck_stats.json
scripts/common/sync_data_b2.sh push
```

`gold_ai_frames` toma la población del ÚLTIMO despliegue del prefiltro:
cambiar el umbral cambia qué frames entran a todos los análisis. Con 0,30 el
panel pierde ~700 frames de DEF 14A y 8-K.

## Qué falta

### 1. Validación humana de las etiquetas — el bloqueante real

Todo resultado descansa en `rhetoric_promotional`, `temporal` y
`specificity_*`, etiquetas de `qwen3.7-flash` que nunca se compararon con
un humano. Lo único medido es acuerdo entre dos LLMs en el prefiltro
(κ=0,87). Existe la herramienta: `ui-validator/` (ver su README) muestrea
300 párrafos con frames y 300 del prefiltro, guarda las anotaciones en el
navegador y `summarize.py` calcula κ humano-juez por dimensión y
precisión/recall del prefiltro reponderados por estrato. Falta anotar.

### 2. Decisiones de tesis abiertas

- **`09_washing_score.md`**: con efectos fijos de sector la cola queda en 3
  empresas; sin ellos, en 8. Cuál es la pregunta ("¿habla más que el
  corpus?" vs. "¿más que su industria?") es una decisión, no técnica.
- **`14_brecha_entre_canales.md`**: la brecha call-filing es un segundo score
  de washing, ortogonal al de `09` (Spearman 0,10). Hay que decidir cuál va
  al centro de la tesis; el de `14` es el que corresponde al mecanismo que
  persigue la SEC.
- Entrada endógena al panel empresa-año (una empresa entra sólo si tuvo ≥3
  frames ese año; `docs/problemas_academicos.md` #12).

### 3. Deuda técnica

- Declarar `statsmodels` y `matplotlib` y destrabar el conflicto que impide
  `uv sync`.
- Las 10 preguntas de `01_...md` son SQL en el documento; convertirlas en
  script como `report_crosscheck_stats.py`.
- `13_shocks.md`: el barrido de ventanas se hace copiando el script con otro
  `WINDOW`; merece un flag.
- `12_grilla_voz_conducta.md`: el manifiesto no guarda la estabilidad 2×2 ni
  la de esquinas; se calculan a mano.

## Costos

| | |
|---|---|
| OpenRouter gastado | ~US$4 de 20 acreditados |
| Instancia GPU | no hace falta |
