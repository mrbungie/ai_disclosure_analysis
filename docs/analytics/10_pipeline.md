# Pipeline de analytics: qué produce cada script y cómo se regenera

## Modo de análisis final: margen extensivo

Todo agregado sobre empresas o períodos se calcula sobre **todos los
documentos**, con cero cuando el documento no habla de IA. La unidad es la
intensidad por 1.000 párrafos del canal (`scripts/analytics/ai_intensity.py`:
tabla de documentos 10-K, 10-Q, DEF 14A, 8-K y calls con sus párrafos y sus
conteos de frames, unidos a todas las instancias de cada texto). Condicionar a
hablar de IA —"la empresa entra si tiene ≥N frames"— selecciona sobre el
fenómeno y no se usa como diseño (`docs/problemas_academicos.md` #12). Lo
único que necesita frames para existir es el test binomial de `08` y los ejes
de `03`, que para la empresa sin frames valen "sin IA".

## Scripts

| script | produce | lo lee |
|---|---|---|
| `ai_intensity.py` | tabla de documentos con párrafos y conteos de frames; `firm_intensity()` por empresa o empresa-año | todo lo de abajo |
| `build_firm_clusters.py` | `firm_year_archetype_behaviors`, `voice_x_behavior` (tasas de texto por empresa-año, sólo con frames) | `build_firm_panels.py` |
| `build_firm_financials.py`, `build_market_factors.py`, `build_roic_wacc.py` | `firm_year_financials*`, `firm_year_market_factors`, `firm_year_filing_returns`, `firm_year_roic_wacc` | `build_firm_panels.py` |
| `build_firm_panels.py` | **`firm_year_master_v2`**: todas las empresas-año con filings, intensidades con ceros, financieros, etiquetas de texto donde existen | `04`, `05`, `01` |
| `build_segments.py` | `firm_segments`, `firm_year_segments` (k=3 + sin IA, 510 empresas) | `02`, `04`, `05`, `07` |
| `build_voice_behavior_grid.py` | `firm_voice_behavior_grid`, `firm_year_voice_behavior_grid` | `03` |
| `economic_profiles.py` | `economic_profiles.json` (paneles A y B) | `04` |
| `incremental_signal.py` | `incremental_signal.json` (M0-M3, ΔR², bootstrap, robustez) | `05` |
| `channel_gap_analysis.py` | `channel_gap_cells`, `channel_gap_firm`, `channel_gap_analysis.json` | `06` |
| `shock_analysis.py`, `shock_did_simple.py` | `shock_analysis.json`, `shock_did_simple.json`, `shock_did_*.png` | `07` |
| `washing_score.py`, `validate_washing_score.py` | `firm_washing_score`, `firm_washing_score_all`, validación | `08` |
| `evolution_figures.py` | `fig_evolucion_*.png`, `fig_sec_event_study.png` | `01`, `07` |
| `report_crosscheck_stats.py` | `crosscheck_stats.json` (correlaciones, FDR, perfiles por nivel de IA) | `apendice/` |

Ninguno llama a un LLM. Todo corre en CPU con `uv run --frozen --no-sync
python scripts/analytics/<script>.py` o, para el bloque financiero,
`make analytics`.

## Decisiones de construcción que afectan cifras

1. **Cadenas de fallback de conceptos XBRL**: cobertura de capex 87%, SG&A
   80%, `net_margin` 98%, revenue 98%.
2. **`next_*_yoy` se anula cuando el gap fiscal sale de [340, 380] días.**
3. **Denominadores ≤ 0 producen NULL**, no un ratio absurdo (ROE con equity
   negativo, P/E con EPS negativo).
4. **ERP geométrico (6,48%)** en el WACC, no aritmético (8,20%). Como
   `coe = rf + β·ERP`, un ERP inflado infla las diferencias de WACC en
   proporción a las de beta.
5. **`build_firm_panels.py` agrega por empresa con `.median()`.**
6. **Año = año de presentación** en el panel empresa-año (igual que los
   financieros); la brecha entre canales alinea por ejercicio fiscal cubierto
   (`06`).
7. **Winsorización 1/99** de todo lo financiero en `04` y `05`.

## Regenerar todo desde cero

```bash
uv run --frozen --no-sync python scripts/common/ai_prefilter_deploy.py --threshold 0.17
uv run --frozen --no-sync python scripts/common/ai_classify.py --concurrency 20   # repetir hasta "Pendientes en total: 0"
uv run --frozen --no-sync python scripts/common/build_duckdb.py --with-text-tables
make analytics
for s in build_segments build_voice_behavior_grid economic_profiles incremental_signal \
         channel_gap_analysis shock_analysis shock_did_simple washing_score validate_washing_score \
         evolution_figures report_crosscheck_stats; do
  uv run --frozen --no-sync python scripts/analytics/$s.py; done
scripts/common/sync_data_b2.sh push
```

`gold_ai_frames` toma la población del ÚLTIMO despliegue del prefiltro; el
umbral vigente y las versiones congeladas están en `docs/FREEZE.md`.

## Pendiente

- `statsmodels` y `matplotlib` no están declarados en `pyproject.toml`, y
  `uv sync` falla por el conflicto `sentence-transformers` / extra
  `pdf-vlm-mineru`; se corre con `--frozen --no-sync`.
- Los descriptivos SQL del apéndice no tienen script.
