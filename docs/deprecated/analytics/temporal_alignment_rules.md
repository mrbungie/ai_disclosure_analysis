# Reglas de alineamiento temporal

Este documento fija cómo joinear, agregar y comparar en el tiempo datos que
vienen de fuentes con granularidad y calendario distintos: filings SEC
(10-K, 10-Q, 8-K, DEF 14A, 20-F, 6-K), earnings calls, precios de mercado,
factores Fama-French y variables de IA derivadas del texto. Las reglas
existen porque cada una de ellas, ignorada, produjo un número que parecía
un hueco de datos y no lo era — la sección "Por qué existe cada regla"
documenta el caso real detrás de cada una.

## 1. Nunca agregues por año calendario sin verificar el año fiscal

**Regla:** un conteo o promedio "por año" que agrupa por
`extract(year from filing_date)` mezcla, dentro del mismo balde, calendarios
fiscales distintos. Alrededor de 40-50 firmas del panel (Apple, Costco,
Cisco, Qualcomm, Microsoft, Nike, Disney, entre otras) cierran su año fiscal
en un mes distinto a diciembre, así que su 10-K cae en octubre-noviembre,
no en febrero-marzo. Comparar un año calendario completo (12 meses) contra
un año-en-curso truncado a una fecha de corte (YTD) exagera la caída: el
YTD todavía no alcanzó el mes en que esas ~45 firmas fiscal-no-calendario
filean.

**Cómo hacerlo bien**, según lo que se está midiendo:

- **Comparar cobertura año contra año (¿está completo 2026 YTD?):**
  usa una **ventana móvil de 12 meses terminando en la misma fecha
  calendario cada año** (ej. `[fin - 365 días + 1, fin]` para
  `fin = 2021-09-03, 2022-09-03, ..., 2026-09-03`), nunca
  `[enero 1, fecha de corte]`. La ventana de 12 meses le da a cada firma,
  sin importar su cierre fiscal, un ciclo completo de filings dentro del
  balde — así el conteo es comparable entre años aunque el año en curso
  esté a medias.
- **Panel de eventos para regresión (beta post-call, CAR, etc.):** no
  agregues por año en absoluto — cada observación ya es un evento con su
  propia fecha (`fecha` = fecha efectiva del call/filing); el año calendario
  solo debería aparecer como *fixed effect* (`fe`), nunca como ventana de
  agregación.
- **Serie fiscal por firma (ratios, ROA, revenue growth):** ancla al
  `period_end` (cierre fiscal informado en el propio filing), nunca al
  `filing_date` — ver regla 3.

## 2. La granularidad de "fecha" no es la misma en todas las fuentes

| Fuente | Columna de fecha primaria | Qué significa |
|---|---|---|
| Filing SEC (10-K/10-Q/8-K/DEF14A/20-F/6-K) | `filing_date` | Cuándo se subió a EDGAR, no el período que cubre |
| Filing SEC (financials) | `period_end` (en `filing_xbrl_facts`/`firm_financials`) | Cierre fiscal que el dato realmente describe |
| Earnings call | `filing_date` (formato string `YYYY-MM-DD`, no `DATE`) | Fecha del call en sí — para calls, `filing_date` *sí* es la fecha efectiva del evento, a diferencia de un 10-K |
| Precios / factores Fama-French | `date` | Fecha de mercado (trading day) |

**Regla:** antes de cualquier join o comparación entre fuentes, decide
explícitamente cuál de las dos fechas de un filing (`filing_date` vs.
`period_end`) es la relevante para la pregunta que estás respondiendo. Un
ratio contable (ROA, leverage) se ancla a `period_end`; un evento de mercado
(¿movió el precio el filing?) se ancla a `filing_date`. Mezclarlas fue
exactamente el bug documentado en `04_perfiles_economicos.md`: mergear
`shares_out` (dato de portada, fechado cerca del filing) por `period_end`
exacto perdía el balance completo porque buscaba la fila equivocada — ver
`attach_asof()` en `scripts/gold/financials/build_firm_financials.py`.

## 3. `merge_asof` — qué `direction` usar y cuándo

El repo usa tres patrones de `pd.merge_asof`, cada uno resuelve una pregunta
distinta. Nunca copies uno sin confirmar cuál pregunta estás haciendo:

### 3.1 `direction="backward", allow_exact_matches=False` — estrictamente anterior

Para cualquier variable que se usa como *predictor* de un evento (histórico,
sorpresa, control): el dato pegado debe ser el último conocido **antes**
del evento, nunca el del mismo día ni uno futuro — de lo contrario hay
look-ahead bias.

```python
return pd.merge_asof(left, right, left_on="fecha", right_on="filing_date",
                     by="ticker", direction="backward",
                     allow_exact_matches=False)
```

Ejemplos reales: `attach_sue()` (SUE conocido antes del call),
`attach_roa()` (ROA conocido antes del call) en
`scripts/analytics/call_beta/call_car_regressions.py`. La misma lógica es la que
sostiene `HistDisclosure`/`HistSubstance` (media expandida de calls
estrictamente anteriores) y `Surprise*` (actual menos ese histórico) —
verificado manualmente contra 481 tickers, cero desviaciones.

### 3.2 `direction="backward"` (con match exacto permitido) — "vigente a esa fecha"

Para series que representan una tasa/nivel vigente en el tiempo (risk-free
rate, factor de mercado) donde el valor del mismo día sí es válido — no hay
look-ahead porque el dato no es un *resultado* del evento, es una condición
de mercado observable en tiempo real ese mismo día.

```python
merged = pd.merge_asof(frame, daily, on="date", direction="backward")
```

Ejemplo real: `risk_free_at()` en `scripts/gold/financials/build_roic_wacc.py`.

### 3.3 `direction="nearest", tolerance=...` — más cercano dentro de una ventana

Para pegar dos anclas temporales *distintas mediciones del mismo hecho
contable* que legítimamente no caen el mismo día (cierre fiscal vs. fecha
de portada del filing) y donde ni "antes" ni "después" es conceptualmente
correcto — lo correcto es el más cercano, acotado por una tolerancia para
no pegar datos de un trimestre distinto por accidente.

```python
merged = pd.merge_asof(left, right.rename(...), left_on=left_on,
                       right_on=f"_{left_on}_matched", by="ticker",
                       direction="nearest",
                       tolerance=pd.Timedelta(days=tolerance_days))
```

Ejemplo real: `attach_asof()` en `scripts/gold/financials/build_firm_financials.py`.

**Checklist antes de cualquier `merge_asof` nuevo:**
1. ¿Ambos lados están ordenados por la columna de merge? (`merge_asof`
   requiere orden ascendente en ambos, `by=` no lo hace automático)
2. ¿`by="ticker"` está presente? Sin partición por ticker, el "más cercano"
   puede venir de otra empresa.
3. ¿El dato de la derecha, si se usa como predictor, puede ser
   *estrictamente posterior* al evento en algún caso? Si sí, falta
   `allow_exact_matches=False` o el `direction` está mal.
4. ¿Las dos columnas de fecha están en el mismo dtype (`datetime64[ns]`,
   no `DATE` de DuckDB mezclado con string)? Ver regla 4.

## 4. Normaliza dtypes de fecha ANTES de unir fuentes, no después

`filing_manifest_earnings_calls*` escribe `filing_date` como string
`'YYYY-MM-DD'`; los manifests de SEC lo escriben como `DATE`. Un
`UNION BY NAME` (o cualquier concat) entre ambos sin normalizar colapsa la
columna a `VARCHAR`, y cualquier `extract(year from filing_date)` posterior
falla en binding — ver el comentario en `build_duckdb.py` sobre por qué la
vista `filing_manifest` hace `TRY_CAST(filing_date AS DATE)` una sola vez,
en la vista, en vez de dejar que cada consumidor lo repita (y probablemente
se olvide).

**Regla:** cualquier script que junte manifests de distinto origen hace el
cast a `DATE`/`datetime64[ns]` en el punto de unión, no en cada consumidor
downstream.

## 5. El identificador de "reporte" no es el mismo en todas las fuentes

`accession_number` (SEC) identifica un filing. Los earnings calls no tienen
`accession_number` — usan `document_id` (formato `TICKER_YYYYQq`). Un
`count(distinct accession_number)` sobre una tabla que mezcla ambas fuentes
cuenta cero earnings calls sin avisar (columna NULL, no error). Cualquier
conteo de "reportes" cruzando tipos de formulario debe usar
`coalesce(accession_number, document_id)` como identificador, nunca
`accession_number` solo.

## 6. Nunca pooles paneles de país o de instrumento distintos

- **Países:** US / Chile / Italia son paneles separados
  (`country_code`), nunca se pooled — universos, calendarios de filing y
  regímenes regulatorios distintos.
- **Instrumento dentro de US:** el panel núcleo 10-K (anual) y la serie de
  shocks 10-Q (trimestral) son dos instrumentos distintos con su propia
  pregunta de investigación cada uno — nunca se pooled en una sola
  regresión aunque compartan ticker y CIK (ver memoria de sesión
  "Data scope: two instruments").

## 7. El universo de firmas también tiene una regla de alineamiento

Un conteo de cobertura "por año" necesita, además de la ventana temporal
(regla 1), un universo de firmas fijo y con ancla explícita — no basta con
"todas las firmas en `firm_universe`". Ver
`scripts/raw_ingestion/us/docs/sp500_2021_start_panel_correction.md`: el universo crudo
de fetch (`firm_universe`, 548 firmas) es deliberadamente más amplio que el
panel analítico (`sp500_2021_start_panel`, 502 firmas, ancladas a
2021-01-01 real). Cualquier tabla de cobertura o regresión que pretenda
representar "el S&P 500 al inicio de la muestra" filtra explícitamente por
ese grupo en `configs/us/universe_membership.csv`, nunca asume que
`firm_universe` completo ya es ese panel.

## Resumen — antes de publicar cualquier tabla o regresión temporal

1. ¿La ventana de agregación es una ventana móvil de 12 meses (o el año
   fiscal propio de la firma), no un año calendario crudo? (regla 1)
2. ¿Elegiste `filing_date` o `period_end` a propósito, no por default?
   (regla 2)
3. ¿El `merge_asof` usa el `direction` correcto para la pregunta
   (predictor estrictamente anterior vs. condición vigente vs. ancla más
   cercana)? (regla 3)
4. ¿Los dtypes de fecha están normalizados antes del join, no después?
   (regla 4)
5. ¿El identificador de reporte es `coalesce(accession_number, document_id)`
   si mezclas SEC y earnings calls? (regla 5)
6. ¿País e instrumento (10-K vs 10-Q) siguen separados? (regla 6)
7. ¿El universo de firmas es el panel analítico anclado, no el universo
   crudo de fetch? (regla 7)
