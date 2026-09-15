# Spines de análisis

Un *spine* es la tabla base de un análisis: una fila por unidad de observación,
con su fecha ancla, antes de pegarle nada más. Todo lo demás (controles,
outcomes, texto de IA, precios) se une al spine con `merge_asof`, nunca al
revés — el spine define la granularidad y la fecha de referencia; las fuentes
que se pegan encima se adaptan a esa fecha, no viceversa.

Este documento cataloga los spines que ya existen o que cualquier análisis
nuevo debería reusar, y para cada uno da la receta de join. Las reglas de
*por qué* cada `direction` de `merge_asof` es la correcta están en
[`temporal_alignment_rules.md`](temporal_alignment_rules.md) — este
documento no las repite, solo dice qué spine usar y qué joins le pegan
encima.

## Cómo elegir spine

```
¿La unidad de análisis es...

  un earnings call?              -> Spine A (call)
  un filing SEC individual?      -> Spine B (filing)
  la tendencia anual de una      -> Spine C (firm-fiscal-year)
    firma (ratios, financials)?
  la serie trimestral de shocks  -> Spine D (firm-fiscal-quarter)
    (10-Q)?
  "¿cuánta data tenemos?"        -> Spine E (cobertura, ventana calendario)
    (conteos descriptivos, no
    una regresión)
```

Si dudas entre B y C/D: **B es "esta fila es un documento"**, C/D es
**"esta fila es un período fiscal, sin importar cuántos documentos lo
describen"** (un 10-K puede tener una enmienda, un período fiscal solo
tiene un valor de revenue).

---

## Spine A — Call (evento = earnings call)

**Grano:** una fila por `(ticker, call)`. **Ancla:** `fecha` = fecha
efectiva del call (no la fecha de un filing relacionado).

**Materializado por `scripts/gold/call_beta/build_call_beta_panel.py`** (`make analytics-call-beta`):
`data/processed/clusters/call_beta_main_panel_10k10q_asof.parquet`. Este spine
vivió como artefacto congelado sin script durante un tiempo (última vez sin
reconstruir: terminaba en 2025-05-15, nunca vio 2026); ese script lo
reconstruye desde `ai_intensity.document_table()` + `firm_activities.parquet`
+ precios/factores/contables, con una validación fila-a-fila contra la
corrida anterior (`build_call_beta_panel.py::validate`) antes de
sobrescribirla. Si necesitas una columna que no tiene, agrégala con un
`attach_*` nuevo en ese script, no por fuera.

**Qué se le pega y cómo:**

| Qué | Función de referencia | `direction` | Por qué |
|---|---|---|---|
| SUE, EPS yoy | `attach_sue()` (`call_car_regressions.py`) | `backward`, `allow_exact_matches=False` | Predictor — no puede ver el propio call ni nada futuro (regla 3.1) |
| ROA (trimestral o anual, el más reciente) | `attach_roa()` (`call_car_regressions.py`) | `backward`, `allow_exact_matches=False` | Igual — predictor estrictamente anterior |
| Beta pre/post, CAR, vol idiosincrática | `attach_market_metrics()` / `market_metrics()` (`call_car_regressions.py`) | ventana explícita `[-252,-21]` o `[-1,+5]` alrededor de `fecha`, no `merge_asof` | El precio es una serie continua, no eventos — se recorta por ventana de trading days, no se busca "el más cercano" |
| Leverage (apalancamiento de portada) | `attach_leverage()` (`call_beta_regressions.py`) | `merge` exacto por `accession_number` | El panel base ya tiene el `accession_number` del 10-K/10-Q asociado — no hace falta asof si ya está resuelto en la construcción del panel |
| HistDisclosure / HistSubstance / Surprise* | (variables ya en el panel) | expanding mean sobre calls **estrictamente anteriores** del mismo ticker | Nunca incluye el call actual — ver regla 3.1 y memoria de verificación (481 tickers, 0 desviaciones) |

**Plantilla de un `attach_*` nuevo sobre el call spine:**

```python
def attach_algo(panel: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    left = panel.sort_values(["fecha", "ticker"]).copy()
    right = source.sort_values(["evento_date", "ticker"]).copy()
    left["fecha"] = pd.to_datetime(left["fecha"])
    right["evento_date"] = pd.to_datetime(right["evento_date"])
    return pd.merge_asof(
        left, right, left_on="fecha", right_on="evento_date", by="ticker",
        direction="backward", allow_exact_matches=False,  # predictor -> sin leakage
    )
```

---

## Spine B — Filing (evento = un filing SEC individual)

**Grano:** una fila por `(ticker, accession_number)`. **Ancla:**
`filing_date` (cuándo se subió, no qué período describe — ver regla 2 si
necesitas el período, no el filing).

**Uso típico:** análisis de contenido/texto de un filing puntual (frames de
IA por filing, longitud, tono), o como tabla derecha para pegarse a un
spine de evento distinto (ej. "el 10-K más reciente antes de este call" ya
es exactamente lo que hace Spine A vía `accession_number`).

**Fuente:** `filing_manifest` / `filing_manifest_10q` / `filing_manifest_8k`
/ `filing_manifest_proxy` / `filing_manifest_20f` / `filing_manifest_6k`
(unidas por `_manifest_source()` en `build_duckdb.py`, `country_code`
siempre presente — nunca pooles países, regla 6). El texto vive en
`paragraphs` (join por `accession_number` + `country_code`); los hechos
XBRL en `filing_xbrl_facts` (join por `accession_number`).

**Qué se le pega:** cualquier variable de nivel-filing (frames de IA
prefiltrados/juzgados, hechos XBRL de portada) se pega por
`accession_number` exacto — no hace falta asof, el filing ya es la unidad.
Si necesitas la variable **más reciente conocida antes de este filing**
(ej. "el precio de cierre el día antes de este 8-K"), ese es un caso de
Spine A con `filing_date` en el rol de `fecha`.

---

## Spine C — Firm-fiscal-year (tendencia anual de una firma)

**Grano:** una fila por `(ticker, fiscal_year)`, `fiscal_year` derivado del
`period_end` del 10-K correspondiente, **no** del año calendario del
`filing_date` (Apple's FY2023 se filea a fines de octubre de 2023 con
`period_end` en septiembre — agruparlo por `year(filing_date)` lo pone en
el balde correcto por coincidencia, pero agruparlo por `year(period_end)`
es lo que realmente quieres decir).

**Fuente base:** `data/processed/clusters/firm_year_financials_ratios.parquet`
(anclado a `filing_date` del 10-K, pero derivado de hechos con `period_end`
propio — ver `attach_asof()` en `build_firm_financials.py` para cómo se
reconcilian dos anclas de fecha del mismo filing).

**Qué se le pega:** cualquier variable anual (retorno anualizado, WACC,
ROIC) se pega por `(ticker, fiscal_year)` exacto si ya comparten esa
columna derivada; si la fuente derecha solo tiene fecha continua (factores
de mercado), usa `merge_asof(direction="nearest", tolerance=...)` contra
`period_end`, como en `attach_asof()`.

---

## Spine D — Firm-fiscal-quarter (serie de shocks 10-Q)

**Grano:** una fila por `(ticker, fiscal_quarter)`, mismo criterio que C
pero trimestral. **Nunca se poolea con Spine C** (regla 6 — son dos
instrumentos distintos, panel anual núcleo vs. serie de shocks trimestral).

**Fuente base:** `data/processed/us_10q_financials_panel.parquet` +
`filing_manifest_10q` para resolver `(ticker, accession_number) ->
filing_date`.

**Qué se le pega:** SUE se calcula *dentro* de este spine
(`(EPS_q - EPS_{q-4}) / sd_prior(...)`, expanding y shifted — ver docstring
de `call_car_regressions.py`), no se pega desde afuera. Cualquier otra
variable trimestral se une por `(ticker, fiscal_quarter)` exacto.

---

## Spine E — Cobertura / descriptivos por ventana calendario

**Grano:** una fila por `(ticker, ventana)`, donde `ventana` es una
**ventana móvil de 12 meses** terminando en la misma fecha calendario cada
año (regla 1 de `temporal_alignment_rules.md`) — **no** un año calendario
crudo, y **no** un evento individual. Este spine es para tablas
descriptivas ("¿cuántos 10-K hay por año?", "¿está completo 2026 YTD?"),
nunca para una regresión.

**Receta:**

```python
anchors = [datetime.date(y, 9, 3) for y in range(2021, 2027)]  # fecha de corte fija
rows = []
for end in anchors:
    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    window = df[(df["filing_date"] > start_ts - pd.Timedelta(days=1))
                & (df["filing_date"] <= end_ts)]
    # agrega dentro de la ventana, no por year(filing_date)
```

**Universo de firmas:** filtra siempre por el panel analítico anclado
(`sp500_2021_start_panel` en `configs/us/universe_membership.csv`, ver
regla 7), nunca por `firm_universe` completo — ese es el universo crudo de
fetch, deliberadamente más amplio.

**Identificador de reporte:** `coalesce(accession_number, document_id)`
(regla 5) — un `count(distinct accession_number)` sobre earnings calls da
cero silenciosamente.

---

## Regla general al construir un spine nuevo

1. Nombra la columna de fecha ancla explícitamente (`fecha`, `filing_date`,
   `period_end`, `ventana_fin` — nunca una genérica `date` que obligue a
   adivinar qué representa).
2. Documenta en el spine mismo (docstring o comentario) qué unidad es una
   fila — si alguien tiene que leer el código para saber si es
   "por filing" o "por período fiscal", el spine está mal nombrado.
3. Todo lo que se le pega usa la tabla de la sección 3 de
   `temporal_alignment_rules.md` para elegir `direction`. Si no calza con
   ninguna, es señal de que falta un spine nuevo en este catálogo, no una
   excepción puntual.
