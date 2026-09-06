# Los builders del panel financiero

El lado contable/mercado del panel lo producen scripts versionados que
corren con `make analytics`. Este documento registra qué hace cada uno, las
decisiones de construcción que afectan las cifras de `02`-`08`, y qué queda
pendiente.

## Los scripts

| Script | Produce | Lee |
|---|---|---|
| `scripts/analytics/build_firm_clusters.py` | arquetipos de voz, clusters de comportamiento, panel empresa-año | `gold_ai_frames` |
| **`scripts/analytics/build_firm_financials.py`** | `firm_year_financials`, `firm_year_financials_ratios` | `data/raw/xbrl_facts/us/`, `filing_manifest` |
| **`scripts/analytics/build_market_factors.py`** | `firm_year_filing_returns`, `firm_year_market_factors` | precios, factores FF3, ratios |
| **`scripts/analytics/build_roic_wacc.py`** | `firm_year_roic_wacc` | ratios, factores, mercado |
| `scripts/analytics/build_firm_panels.py` | los cuatro parquets con etiquetas que leen 02-08 | los tres anteriores |
| **`scripts/analytics/report_crosscheck_stats.py`** | las tablas numéricas de 02/04/05/08 | los parquets |
| `scripts/analytics/washing_score.py` | `firm_washing_score` | `gold_ai_frames` |

Todo corre con **`make analytics`**, en el orden
de dependencia correcto. Ninguno llama a un LLM ni cuesta un peso de API:
leen `gold_ai_frames` (que sí costó llamadas) sin tocarlo.

El insumo `data/raw/xbrl_facts/us/` (515 tickers, 113 MB) no estaba en el
disco local: se baja con
`scripts/common/sync_data_b2.sh pull --path raw/xbrl_facts`.

## Decisiones de construcción

1. **Cadenas de fallback de conceptos XBRL.** Un filer que reporta
   `RevenueFromContractWithCustomerExcludingAssessedTax` en vez de `Revenues`
   ya no queda sin revenue. Cobertura: capex **69% → 87%**, SG&A **56% → 80%**,
   `net_margin` **87% → 98%**, EV/EBITDA **48% → 68%**, revenue 96% → 98%.
2. **`next_*_yoy` se anula cuando el gap fiscal sale de [340, 380] días.** Son
   huecos de XBRL que ponen el "FY siguiente" a 2-4 años (11 filas acá), lo
   que `leakage-checking.md` pedía filtrar y nadie filtraba.
3. **Denominadores <= 0 producen NULL, no un ratio absurdo.** Equity negativo
   ya no genera ROE positivo espurio; EPS negativo ya no genera P/E negativo
   (por eso la cobertura de P/E baja de 98% a 89%: esas filas nunca debieron
   contar).
4. **ERP geométrico (6,48%) en vez de aritmético (8,20%)** en el WACC. El 8,2%
   es la media aritmética del `mktrf` diario 2000-2026 anualizada ×252; la
   geométrica del mismo período es 6,48% y es la que corresponde para
   descontar. **No es neutral entre segmentos**: como `coe = rf + β·ERP`, un
   ERP inflado infla las diferencias de WACC en proporción a las diferencias
   de beta, y los segmentos de `08_...md` difieren justamente en beta (0,46
   vs 1,00). `--erp 0.082` reproduce la versión vieja.

## Qué queda pendiente

- El panel empresa-año sigue condicionado a ≥3 frames en el año, así que
  entrar y salir del panel es endógeno al propio fenómeno que se mide.
- Las 10 preguntas de `01_...md` son consultas SQL en el propio documento;
  convertirlas en script como `report_crosscheck_stats.py` evitaría
  re-correrlas a mano.
- Declarar `statsmodels` y `matplotlib` en `pyproject.toml` y destrabar el
  conflicto `sentence-transformers` / extra `pdf-vlm-mineru` que hoy impide
  `uv sync` (se corre con `uv run --frozen --no-sync`).
- `build_firm_panels.py` agrega por empresa con `.median()`, que es lo que
  `07_...md` documenta.
