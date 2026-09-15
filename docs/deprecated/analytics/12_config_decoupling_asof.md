# Configuración + decoupling, todo as-of, y los bugs de integridad que
# tapaba `call_beta_main_panel_10k10q_asof.parquet`

Sesión de auditoría del panel de beta/NCSKEW: se encontraron y corrigieron
cuatro bugs de integridad de datos en `call_beta_main_panel_10k10q_asof.parquet`
y en `call_crash_risk_panel.parquet`, y se construyó una especificación
alternativa a la oficial (`call-beta-regressions.md`) que reemplaza los
constructos D/S por configuración (pesos de arquetipo continuos, as-of) +
decoupling (HistW/SurpriseW), para ver si el hallazgo de Governance-Led
sobre crash risk sobrevive a una prueba genuinamente predictiva (sin
leakage), no solo contemporánea.

## Los bugs encontrados en `build_call_beta_panel.py` (ya corregidos)

Antes de esta sesión, `call_beta_main_panel_10k10q_asof.parquet` tenía 10,751
filas con 288 duplicados reales, con tres causas distintas:

1. **Dual-class shares.** Discovery (DISCA/DISCK) reporta la misma call bajo
   dos símbolos; `firm_universe` alias-ea DISCK → DISCA, así que la misma
   call sobrevivía dos veces bajo el ticker DISCA. Fix: en
   `build_call_spine()`, filtrar filas donde el símbolo de origen en
   `call_accession_number` (`{SYMBOL}_{YEAR}Q{N}`) no coincide con el ticker
   del panel.
2. **Retailers con año fiscal no-calendario** (LOW, TGT, HD, DG, BBWI, DLTR,
   PVH, ULTA, KR, ROST...): la misma call real llega dos veces con etiquetas
   de año fiscal distintas para el MISMO número de trimestre (ej.
   `LOW_2022Q2` y `LOW_2023Q2`, ambas el 2022-08-17, con `n_words`
   ligeramente distintos — dos transcripciones de fuentes distintas del
   mismo evento). Fix: dedupe por (ticker, fecha, número de trimestre
   extraído de `call_accession_number`), quedándose con la transcripción de
   más palabras.
3. **`attach_market()` mergeaba solo por `(ticker, fecha)`**, sin
   `call_accession_number`. Para el puñado de tickers (ADI, DG, TDG) donde
   dos calls trimestrales GENUINAMENTE DISTINTAS comparten una `fecha`
   corrupta río arriba (bug de `document_table()`, sin diagnosticar del
   todo — no hay evidencia de que sea el mismo mecanismo que 1-2), el merge
   cruzaba en producto cartesiano: 2 filas del panel × 2 filas de mercado
   calculadas = 4 filas, duplicando beta/precio. Fix: agregar
   `call_accession_number` a la clave del merge en `attach_market()`.
4. **`hist_disclosure`/`hist_substance`** (expanding().shift(1)) podían
   quedar en un orden no determinista para esos mismos pares de fecha
   corrupta. Fix: ordenar por `(ticker, fecha, call_accession_number)` en
   vez de solo `(ticker, fecha)` antes del `expanding()`.

Resultado: 10,751 → 10,507 filas. Solo quedan 6 filas (3 pares: ADI, DG,
TDG) con `(ticker, fecha)` compartida entre dos trimestres genuinamente
distintos — correctamente preservadas y ordenadas, no un bug.

`call_crash_risk_panel.parquet` es un artefacto **huérfano**: no hay ningún
script en el repo que lo construya (probablemente el script que lo generó
fue borrado o nunca se commiteó). Tenía el mismo patrón de duplicados
(DISCA×84, retailers×16-24). Como NCSKEW/DUVOL dependen solo de
`(ticker, fecha)` — no del contenido fiscal de la call —, se verificó que
los grupos duplicados tuvieran valores idénticos (0 de 72 grupos difería) y
se aplicó `drop_duplicates(subset=["ticker","fecha"])` directo sobre el
archivo. 10,751 → 10,535 filas.

**Impacto en el número headline oficial (`call-beta-regressions.md`):** con
el panel limpio, $\beta$ sobre `hist_disclosure` para `beta_post_126` se
mueve de 0.079 (p=0.006, N=8,555) a ~0.078-0.098 según la ventana usada
(ver más abajo) — el hallazgo no cambia de signo ni de significancia.

## Otros fixes de cobertura, en la misma sesión (ver también `call-beta-regressions.md`)

- **Factores Fama-French (`data/raw/market/factors/ff3_daily.parquet`)**
  estaban stale (corte en junio 2026 pese a que precios y calls llegan a
  septiembre 2026) — refrescados con
  `scripts/raw_ingestion/market/01_collect_market_data.py` (función
  `collect_factors(refresh=True)`). El límite real de publicación de Ken
  French's data library es ~1 mes de rezago, no arreglable más allá de eso.
- **56 tickers sin ningún archivo de precio** en `data/raw/market/prices/`
  (empresas absorbidas/deslistadas — yfinance/stooq/tiingo solo sirven
  tickers activos). De esos, 18 resultaron ser fallas de descarga
  recuperables (siguen cotizando: CAG, KMX, NOV, UNM, VFC, VNO, WU, XRAY,
  XRX, SLG, VNT, DXC, EMN, FLS, FTI, PRGO, BBWI, PARA) — recuperados con el
  collector oficial. Los 38 restantes son delisting real (ATVI, TWTR, CERN,
  XLNX, MXIM, DISCA/DISCK como entidad separada de precio, PBCT, SIVB, FRC,
  WBA, WLTW, PXD, KSU, EA —delistó a mediados de 2026—, etc.) y necesitarían
  CRSP; se probó `HaiwenWang/multimodal_stock_data` en HuggingFace (tiene
  CRSP daily/delistings) pero el repo está *gated*, pendiente de aprobación
  del autor.
- **`operating_margin`** usaba un solo concept-tag XBRL
  (`us-gaap:OperatingIncomeLoss`), sin fallback — a diferencia de todas las
  demás métricas en `DURATION_METRICS`. 74.7% de cobertura, concentrado en
  financieras/bancos/seguros (SIC 60-63, 67), que estructuralmente no
  reportan bajo ese tag. `shares_out` tiene el mismo patrón (un solo tag
  `dei:EntityCommonStockSharesOutstanding`), peor en 2021-2022 por
  inmadurez de tagging DEI temprana, mejora sola con los años.
  **`roa` (`net_income`/`total_assets`, ambos casi universales) tiene
  99.7% de cobertura** — se adoptó como único control financiero en vez de
  `operating_margin`+`asset_turnover` en `build_call_beta_panel.py`,
  `call_beta_regressions.py`, `call_beta_generalized_targets.py`, y
  `build_call_crash_and_archetypes.py` (`ACCOUNTING_VARS`/`BASE`/
  `FIXED_CONTROLS`/`BASE_CTRLS`).
- **Ventana de beta**: 126 días (semestral) censura casi todo 2026 sin
  necesidad — a 63 días (trimestral) la cobertura de 2026 sube de 18.9% a
  55.4%, sin cambiar nada en 2021-2025 (87-91% en ambas ventanas). Se
  agregó `beta_post_63`/`BETA_MIN_OBS_63=60` a `build_call_beta_panel.py` y
  se usa como headline en `call_beta_regressions.py`; NCSKEW se mantiene a
  126 días como especificación principal (necesita más observaciones para
  estimar asimetría de forma estable), con una versión a 63 días construida
  aparte (`build_ncskew_63.py`) para comparación.

## La especificación de configuración + decoupling

Reemplaza los cuatro constructos D/S (`HistD`, `HistS`, `SurpriseD`,
`SurpriseS`) de `call-beta-regressions.md` por dos bloques conceptualmente
separados:

$$
Y_{i,t} = \gamma_1 w^{voc}_{i,t^-} + \gamma_2 w^{gov}_{i,t^-}
+ \gamma_3 HistD_{i,t^-} + \theta_1 HistW_{i,t^-} + \theta_2 SurpriseW_{i,t}
+ \delta X^{pre}_{i,t} + \rho Y^{pre}_{i,t} + FE_{SIC2 \times CallYear} + \epsilon_{i,t}
$$

- **Configuración** ($w^{voc}$, $w^{gov}$): pesos continuos de mezcla del
  Archetypal Analysis expansivo (`build_archetype_weights_quarterly_asof.py`),
  matcheados *as of* el trimestre fiscal propio de la call — extraído de
  `call_accession_number` (`{TICKER}_{YEAR}Q{N}`), **no de `fecha`** (que
  para un puñado de tickers no distingue trimestres, bug #3 de arriba).
  Backward, trimestre estrictamente anterior (`allow_exact_matches=False`).
  Firmas sin clasificación de arquetipo (sin suficiente historial de IA)
  reciben $w^{voc}=w^{gov}=0$, no se descartan.
- **$HistD_{i,t^-}$**: la intensidad de disclosure expansiva propia de la
  tesis (`hist_disclosure`, ya en el panel principal), 0 si no hay historial
  previo — reemplaza a una dummy binaria de "No AI" por una medida continua
  del mismo estado.
- **Decoupling** ($HistW$, $SurpriseW$): $w^{raw}_{i,t} = z(\text{disclosure}_{i,t}) - z(\text{substance}_{i,t})$,
  con $z(\cdot)$ estandarizado con **media/std EXPANSIVA transversal por
  fecha de calendario** (no la constante de toda la muestra completa, que
  filtraría los años de disclosure mucho más alto hacia atrás en el tiempo
  — la tesis documenta que la intensidad narrativa creció ~7x 2021-2025).
  $HistW_{i,t^-} = $ media expansiva de $w^{raw}$ de calls estrictamente
  anteriores de la misma firma; $SurpriseW_{i,t} = w^{raw}_{i,t} - HistW_{i,t^-}$
  (contemporáneo por diseño, mide si ESTA call se desvía del patrón propio).
- **Controles** ($X^{pre}$): `log_market_cap`, `return60`, `roa` — las
  únicas con cobertura alta, ver arriba. `beta_pre`/`ncskew_pre`/
  `ncskew_pre_63` como $Y^{pre}$ (variable dependiente rezagada,
  especificación ANCOVA, igual que la oficial).

### Resultados (N sobre el panel ya limpio, `main()` de cada script)

| Outcome | $w^{voc}$ | $w^{gov}$ | $HistD$ | $HistW$ | $SurpriseW$ | N |
|---|---|---|---|---|---|---|
| `beta_post_63` (headline) | +0.005, p=0.71 | **+0.029, p=0.003** | **+0.084, p<0.0001** | +0.006, p=0.68 | +0.005, p=0.61 | 8,637 |
| `ncskew_post` (126d) | -0.011, p=0.67 | -0.002, p=0.90 | -0.014, p=0.62 | -0.033, p=0.11 | -0.013, p=0.24 | 7,939 |
| `ncskew_post_63` (63d) | -0.012, p=0.53 | -0.010, p=0.42 | -0.016, p=0.47 | **-0.042, p=0.010** | -0.024, p=0.057 | 8,641 |

Todos con $Y^{pre}$ y los 3 controles incluidos (no tabulados arriba por
espacio, ver el output completo del script).

**Lectura:** $w^{gov}$ (Governance-Led continuo, as-of, sin leakage) es
significativo en beta pero con signo **positivo** — al revés del hallazgo
oficial contemporáneo ($\beta=-0.150$, p=0.0016 sobre NCSKEW, con dummy no
rezagada). En NCSKEW (ninguna de las dos ventanas) $w^{gov}$ no es
significativo bajo esta especificación. `HistD` (disclosure expansiva
propia) domina la predicción de beta. `HistW`/`SurpriseW` (decoupling) son
los únicos regresores de este bloque que salen significativos en NCSKEW a
63 días.

**Esto no reemplaza el hallazgo oficial de `call-beta-regressions.md` /
`build_call_crash_and_archetypes.py`** (dummy de arquetipo contemporánea,
survives Benjamini-Hochberg FDR en la familia de 18 tests) — es una prueba
de robustez adicional con una especificación genuinamente predictiva
(as-of, sin leakage), que da un resultado distinto y debe leerse como tal:
el efecto de Governance-Led sobre crash risk es sensible a si la
clasificación de arquetipo es contemporánea o estrictamente rezagada.

## Scripts

- `scripts/gold/posture/build_archetype_weights_quarterly_asof.py` — pesos
  continuos de AA expansivo a cortes trimestrales (no anuales), guarda
  `data/processed/clusters/panel_expanding_archetype_weights_quarterly.parquet`.
- `scripts/gold/call_beta/build_ncskew_63.py` — NCSKEW a 63 días desde cero
  (Chen-Hong-Stein 2001, residuos del mismo modelo de mercado que beta),
  guarda `data/processed/clusters/ncskew_63.parquet`.
- `scripts/analytics/call_beta/call_beta_config_decoupling_asof.py` — corre la
  especificación completa sobre los tres outcomes, imprime la tabla de
  coeficientes.

```bash
source .venv/bin/activate
.venv/bin/python scripts/gold/posture/build_archetype_weights_quarterly_asof.py
.venv/bin/python scripts/gold/call_beta/build_ncskew_63.py
.venv/bin/python scripts/analytics/call_beta/call_beta_config_decoupling_asof.py
```

## Pendiente

- Diagnosticar la causa exacta de la `fecha` corrupta compartida entre dos
  calls trimestrales distintas para ADI/DG/TDG (bug #3) — no se llegó a la
  causa raíz en `document_table()`/`filing_manifest`, solo se mitigó su
  efecto en `attach_market()`.
- Si se aprueba el acceso a `HaiwenWang/multimodal_stock_data` (CRSP) en
  HuggingFace, reintentar recuperar precios para los 38 tickers delisted
  restantes.
