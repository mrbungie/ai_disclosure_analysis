# Estado del proyecto y qué falta

Todo se retoma en cualquier equipo: el código está en git y los datos en B2.
El instrumento está congelado (`docs/FREEZE.md`); analytics está cerrado en
diseño (`docs/analytics/README.md`) y le faltan dos cosas, una de ellas humana.

## Cómo levantar el proyecto

```bash
git pull
cp .env.example .env                       # y completar las claves (ver abajo)
scripts/common/sync_data_b2.sh pull        # baja data/ entero (~decenas de GB)
```

Bajar sólo lo necesario para analytics:

```bash
for p in interim/ai_classify interim/prefilter_predictions_unique interim/manifests processed/clusters raw/xbrl_facts; do
  scripts/common/sync_data_b2.sh pull --path $p; done
```

Claves en `.env`: `OPENROUTER_API_KEY` (clasificación), `B2_KEY_ID` /
`B2_APPLICATION_KEY` / `B2_BUCKET` / `RCLONE_REMOTE_NAME` (datos).

**Entorno.** `uv sync` falla por un conflicto de resolución entre
`sentence-transformers==6.0.1` y el extra `pdf-vlm-mineru`; todo corre con
`uv run --frozen --no-sync python ...` sobre el `.venv`. `statsmodels` y
`matplotlib` no están declarados en `pyproject.toml`
(`uv pip install --python .venv/bin/python statsmodels matplotlib`). No hace
falta GPU: embeddings y scoring están corridos para todo el corpus.

## Plan de analytics, cerrado

| bloque | doc | estado |
|---|---|---|
| 0 | freeze + validación humana (`docs/FREEZE.md`, `ui-validator/`) | **falta la anotación** |
| 1 | evolución 2021-2025 (`01`) | hecho |
| 2 | segmentación k=3 + sin IA (`02`) | hecho, no se reclusteriza |
| 3 | voz × conducta como mapa continuo (`03`) | hecho |
| 4 | perfiles económicos por segmento, crudos y dentro de sector × año (`04`) | hecho |
| 5 | **señal incremental: fundamentals → + volumen → + contenido** (`05`) | hecho: 2-4 puntos de R² parcial, cargados por especificidad |
| 6 | brecha promocional entre canales (`06`) | hecho, headline secundario |
| 7 | SEC 2024 (`07`) | cerrado como no-identificación; no se rescata |
| 8 | DeepSeek (`07`) | sección corta, nulo |
| 9 | robustez mínima (en `04` y `05`: sector × año, sólo 10-K, sin IT, winsor, FE de empresa, bootstrap) | hecho salvo la sensibilidad a la validación humana |
| 10 | clusters viejos / ROIC−WACC por grupos de washing | eliminado del cuerpo; en git |

## Qué falta

### 1. Validación humana de las etiquetas — el único bloqueante

Todo resultado descansa en `rhetoric_promotional`, `temporal`,
`specificity_*` y los bloques de conceptos, etiquetas de `qwen3.7-flash` que
nunca se compararon con un humano. La herramienta existe: `ui-validator/`
muestrea 300 párrafos con frames y 300 del prefiltro, guarda las anotaciones
en el navegador y `summarize.py` calcula κ humano-juez por dimensión y
precisión/recall del prefiltro reponderados por estrato. Objetivo: 400-500
párrafos (subir `--frames` en `build_sample.py`). Después: si un campo tiene
acuerdo mediocre, colapsarlo y re-correr `04`-`06` con el campo colapsado
(robustez 7 del plan).

### 2. Decisión de escritura

Los tres hallazgos que sostienen la tesis, con independencia de lo que salga
de la validación:

1. **La divulgación de IA no es unidimensional**: tres tipos estables de
   divulgación más el silencio, con perfiles económicos coherentes dentro de
   sector y año (`02`, `04`).
2. **El contenido aporta señal más allá del volumen y de los fundamentals**:
   2-4 puntos de R² parcial en riesgo, valuación, I+D y crecimiento,
   concentrados en la especificidad (`05`).
3. **La misma empresa cuenta otra historia de IA en la call que en el filing**:
   20 veces más promoción por párrafo, la brecha se abre con el boom de 2023
   y no hay evidencia identificable de que el escrutinio de la SEC la haya
   reducido (`06`, `07`).

## Deuda técnica

- Declarar `statsmodels` y `matplotlib` y destrabar el conflicto que impide
  `uv sync`.
- Las transcripciones de calls terminan a mediados de 2025 (~3,2 de 4 por
  empresa-ejercicio); si la fuente se actualiza,
  `scripts/us/earnings_calls/01_fetch_transcripts.py --from-year 2025 --to-year 2026`
  y después prefiltro + clasificación (aditivas). Eso reabre el congelamiento
  sólo para el corpus de calls.
- Los descriptivos SQL del apéndice no tienen script.

## Costos

| | |
|---|---|
| OpenRouter gastado | ~US$4 de 20 acreditados |
| Instancia GPU | no hace falta |
