"""El lado CONTABLE del panel empresa-año desde XBRL crudo (librería de
`scripts/gold/firm_year/build_market_financials.py`, que escribe
covariates/firm_year/financials y targets/firm_year/financials).

`annual_panel()` devuelve una fila por 10-K presentado: el FY que reporta
(revenue / R&D / capex / SG&A, resultado, balance, acciones de portada,
márgenes, ROA/ROE, liquidez, apalancamiento, EBITDA, sic2), el crecimiento de
revenue que ese 10-K reporta (`revenue_yoy`: FY reportado contra el FY anual
anterior del mismo ticker, conocido al filing) y el crecimiento YoY del FY
SIGUIENTE (`next_*_yoy`, un outcome).

Regla de alineamiento temporal (docs/analytics/leakage-checking.md — esta es
la parte que NO se puede simplificar):

  1. Sólo hechos XBRL de duración anual (340-380 días) cuentan como "el FY".
  2. Para cada 10-K de un ticker, `disclosed_period_end` = el `period_end`
     anual más reciente <= `filing_date` (+10 días de margen). Es el FY que
     el documento efectivamente reporta: dato YA CONOCIDO al momento del
     filing, válido como covariable contemporánea, nunca como outcome.
  3. `next_period_end` = el `period_end` anual inmediatamente siguiente de
     ESE MISMO ticker. Es el FY que todavía no había terminado cuando se
     presentó el documento: el único válido como outcome futuro.

  Nunca se usa aritmética de año calendario. El mes de cierre fiscal varía
  por empresa (NVDA cierra en enero, KO en diciembre) y un shift fijo de -1
  año mezcla horizontes de predicción distintos entre empresas — el bug que
  documenta leakage-checking.md.

Diferencias deliberadas respecto de la versión perdida (rige lo nuevo):

  a. `next_*_yoy` queda en NULL cuando el gap entre `disclosed_period_end` y
     `next_period_end` cae fuera de [340, 380] días (lo mismo para `revenue_yoy`
     con el FY anterior). Son huecos de XBRL que
     ponen el "FY siguiente" a 2-4 años de distancia (3 de 2.325 filas en la
     versión anterior, que leakage-checking.md pedía filtrar y nadie
     filtraba). `next_gap_days` queda en la tabla para poder auditarlo.
  b. Cadenas de fallback por métrica (ver `DURATION_METRICS`): un filer que
     reporta `RevenueFromContractWithCustomerExcludingAssessedTax` y no
     `Revenues` ya no queda sin revenue. Sube cobertura sin cambiar la
     definición.
  c. Denominadores <= 0 producen NULL en vez de un ratio absurdo (equity
     negativo -> ROE/PB/deuda-equity nulos). La versión anterior los dejaba
     entrar y después los winsorizaba al 2-98%.

  d. Fuente de los hechos XBRL: `bronze.xbrl_facts` (construida por
     `scripts/bronze/xbrl_facts.py` desde `data/raw/xbrl_facts/us_by_filing/`),
     el inline-XBRL parseado directamente de cada 10-K/10-Q cacheado
     (`scripts/raw_processing/us/04_extract_inline_xbrl_facts.py`), no el
     bulk pull de SEC Company Facts (`data/raw/xbrl_facts/us/`,
     `scripts/raw_ingestion/us/03_fetch_accounting_data.py`).
     Company Facts colapsa reexpresiones: el valor que devuelve para un
     period_end puede ser el que la empresa reportó AÑOS después de ese
     10-K, no el que ese 10-K efectivamente reveló — filtra el propósito de
     `disclosed_period_end` (dato conocido al momento del filing). El
     inline-XBRL, al venir del HTML de cada filing individual, no tiene ese
     problema. `docs/sources/accounting_data.md` tiene la cobertura vigente
     y el detalle de los bugs de fondo encontrados y corregidos en esta
     fuente (hechos dimensionales corrompiendo el consolidado, fugas de
     reexpresión) — no se repite aquí para no tener el número en dos
     lugares que se puedan desincronizar.

Determinístico, sin LLM, sin costo de API: lee `bronze.xbrl_facts`,
`silver.filing_manifest` y `silver.firm_universe`; no escribe nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 340, 380
# Margen entre el cierre fiscal y el filing: el 10-K se presenta semanas
# después del cierre, así que el FY disclosed es el último que cerró ANTES
# del filing. Los 10 días cubren el caso raro de un filing casi simultáneo.
DISCLOSED_SLACK_DAYS = 10
# Balance: los hechos "instant" del cierre fiscal caen exactamente en
# `period_end`; 20 días cubren diferencias de convención (último día hábil).
INSTANT_TOLERANCE_DAYS = 20
# Portada del 10-K: `dei:EntityCommonStockSharesOutstanding` se fecha cerca
# del FILING, no del cierre fiscal — semanas después. Se busca por separado.
SHARES_TOLERANCE_DAYS = 75

# Cadenas de fallback: se toma el primer concepto con dato para ese
# (ticker, period_end). El orden es de definición más estrecha a más amplia.
DURATION_METRICS: dict[str, list[str]] = {
    "revenue": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueNet",
        "us-gaap:SalesRevenueGoodsNet",
    ],
    "cost_of_revenue": [
        "us-gaap:CostOfRevenue",
        "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:CostOfGoodsSold",
        "us-gaap:CostOfServices",
    ],
    "rd_expense": [
        "us-gaap:ResearchAndDevelopmentExpense",
        "us-gaap:ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        # Mismo gasto corriente de I+D, desagregado para software por algunos
        # emisores (p. ej. Adobe); no confundir con costos de desarrollo
        # capitalizados, de petróleo/inmobiliario o I+D en proceso adquirido.
        "us-gaap:ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
        # Equivalente IFRS para los emisores extranjeros del universo US.
        "ifrs-full:ResearchAndDevelopmentExpense",
    ],
    "sga_expense": [
        "us-gaap:SellingGeneralAndAdministrativeExpense",
        "us-gaap:GeneralAndAdministrativeExpense",
    ],
    "capex": [
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
        "us-gaap:PaymentsToAcquireProductiveAssets",
        "us-gaap:PaymentsToAcquireOtherPropertyPlantAndEquipment",
        # REITs' own name for capex: cash spent improving real estate
        # already owned, not acquiring new PP&E outright.
        "us-gaap:PaymentsForCapitalImprovements",
    ],
    "operating_income": ["us-gaap:OperatingIncomeLoss"],
    "net_income": [
        "us-gaap:NetIncomeLoss",
        "us-gaap:ProfitLoss",
    ],
    "da": [
        "us-gaap:DepreciationDepletionAndAmortization",
        "us-gaap:DepreciationAndAmortization",
    ],
    "interest_expense": [
        "us-gaap:InterestExpense",
        "us-gaap:InterestExpenseNonoperating",
        "us-gaap:InterestExpenseDebt",
    ],
    "pretax_income": [
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "tax_expense": ["us-gaap:IncomeTaxExpenseBenefit"],
    "eps_diluted": [
        "us-gaap:EarningsPerShareDiluted",
        # Filers with a simple capital structure (no dilutive securities)
        # sometimes tag one combined basic-and-diluted figure instead of
        # a separate diluted concept; the two are equal by definition
        # when this tag is used, so it's a safe fallback, not a proxy.
        "us-gaap:EarningsPerShareBasicAndDiluted",
    ],
}

INSTANT_METRICS: dict[str, list[str]] = {
    "total_assets": ["us-gaap:Assets"],
    "equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "current_assets": ["us-gaap:AssetsCurrent"],
    "current_liabilities": ["us-gaap:LiabilitiesCurrent"],
    "long_term_debt": [
        "us-gaap:LongTermDebtNoncurrent",
        "us-gaap:LongTermDebt",
        # Filers that fold finance-lease obligations into the debt line
        # instead of a separate LongTermDebt concept (e.g. CSX, Cardinal
        # Health, Cummins) -> 45 tickers (~9%) had zero debt tag without
        # this pair; adding both recovers 26 of them.
        "us-gaap:LongTermDebtAndCapitalLeaseObligations",
        # Some non-financial filers (homebuilders, distributors) tag their
        # long-term borrowings as NotesPayable rather than LongTermDebt.
        "us-gaap:NotesPayable",
    ],
    "cash": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
}
SHARES_METRIC = {"shares_out": ["dei:EntityCommonStockSharesOutstanding"]}

# Balance-sheet concepts that can never legitimately be negative — a
# negative value here is a filer sign-tagging error, not a fact. Verified
# case: DuPont's own 2021-02-12 10-K tags us-gaap:LongTermDebt as
# -$21.811B in the SAME filing where the sibling concept
# LongTermDebtAndCapitalLeaseObligations correctly shows +$21.806B for
# the identical period — a sign flip, not a real negative liability.
# Dropping the impossible value lets the normal fallback-priority chain
# recover the correct sibling concept instead of guessing at a fix.
NEVER_NEGATIVE_CONCEPTS = {
    concept
    for metric in ("total_assets", "current_assets", "current_liabilities", "long_term_debt", "cash")
    for concept in INSTANT_METRICS[metric]
}

GROWTH_METRICS = ["revenue", "rd_expense", "capex", "sga_expense"]
LEVEL_COLUMNS = ["revenue", "rd_expense", "capex", "sga_expense"]

# `da` sum-fallback: 58 of 509 tickers never tag a combined D&A concept but
# DO tag depreciation and intangible amortization as two separate line
# items (verified magnitudes plausible, e.g. ABT: $1.1B depreciation +
# $2.18B amortization). Only filled when BOTH components exist for that
# (ticker, period_end) — a company with only one tagged would understate
# D&A if filled from that alone, a different, worse-biased definition
# than simply leaving it NULL (CLAUDE.md: one construct, one formula).
DA_COMPONENTS = {
    "depreciation_only": ["us-gaap:Depreciation"],
    "amortization_only": ["us-gaap:AmortizationOfIntangibleAssets"],
}

# `pretax_income` sum-fallback: same pattern, smaller (6 tickers) — BR,
# CMA, LH, MCD, ORCL, PFG tag domestic and foreign pretax income
# separately instead of one combined concept.
PRETAX_COMPONENTS = {
    "pretax_domestic": ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"],
    "pretax_foreign": ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"],
}

# Bank `revenue` OVERRIDE (not fill — see fill_via_component_sum's
# override mode): verified accounting identity, InterestIncomeExpenseNet
# (net interest income, i.e. interest income already net of interest
# expense) + NoninterestIncome = Revenues at 0.0% difference against
# every year of BAC/COF/JPM/C/PNC, the banks that already tag a working
# combined Revenues concept. FITB/ZION/CMA/SIVB/HBAN/RF/PBCT never tag
# that combined concept — only one scoped to ASC 606 fee income, which
# structurally excludes net interest income, a bank's core revenue
# (FITB reads ~$580M against several billion actually earned). Gross
# interest income (InterestAndDividendIncomeOperating) does NOT reconcile
# to Revenues the same way — checked and rejected before landing on the
# net figure. NoninterestIncome is tagged by exactly 27 tickers in this
# universe, every one financial-sector (SIC 6021/6022/6035/6141/6199/6211
# — banks, thrifts, consumer credit, broker-dealers); safe to apply
# unconditionally rather than needing a SIC-code gate.
BANK_REVENUE_COMPONENTS = {
    "bank_net_interest_income": ["us-gaap:InterestIncomeExpenseNet"],
    "bank_noninterest_income": ["us-gaap:NoninterestIncome"],
}

# Per-ticker concept exclusions: a specific (ticker, concept) combination
# known to be wrong for that filer, dropped BEFORE pivot_metrics so its
# normal same-day priority tiebreak falls through to the next concept in
# the chain instead of reordering the chain globally (which would risk
# the Iron Mountain / General Mills / Mastercard / Philip Morris cases
# `pivot_metrics`'s docstring already worked out).
#
# BLK: starting with its FY2023-filed 10-K (accession 0000950170-24-019271,
# filed 2024-02-23), BlackRock began ALSO tagging a much smaller
# `us-gaap:Revenues` non-dimensionally alongside its established
# `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` (same
# filing, same period, both non-dimensional) — e.g. FY2024: Revenues
# $12.794B vs RevenueFromContractWithCustomer... $20.407B. Verified against
# BlackRock's actual reported FY2021-2025 revenue (~$17-24B every year):
# RevenueFromContractWithCustomer... matches every year including the five
# years BEFORE the second tag appeared; Revenues does not match any year
# it's present for. `us-gaap:Revenues` has priority 0 in DURATION_METRICS
# (checked first), so without this exclusion it silently wins the same-day
# tiebreak and understates BLK's revenue by ~35-40% for FY2023 onward.
TICKER_CONCEPT_EXCLUSIONS: dict[str, set[str]] = {
    "BLK": {"us-gaap:Revenues"},
}


def _concept_priority(metrics: dict[str, list[str]]) -> pd.DataFrame:
    rows = [(concept, metric, rank)
            for metric, concepts in metrics.items()
            for rank, concept in enumerate(concepts)]
    return pd.DataFrame(rows, columns=["concept", "metric", "priority"])


def scan_facts() -> pl.LazyFrame:
    """Todos los hechos XBRL de `bronze.xbrl_facts`."""
    return L.scan("bronze.xbrl_facts")


CELL_KEYS = ["ticker", "concept", "period_type", "period_start", "period_end"]


def _first_reported(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Una fila por celda (ticker, concepto, período): la de `filing_date` más
    temprana. Dentro del mismo `filing_date` gana la primera aparición en el
    orden de los archivos de origen (un mismo filing a veces repite el hecho
    con otra precisión, p. ej. 12.786 vs 12.8 miles de millones); ese orden
    es `(source_file, source_row)`, la posición física en
    `bronze.xbrl_facts` -- determinístico porque bronze escribe los hechos
    ordenados por archivo (nombre) y por fila dentro del archivo."""
    return (lf.sort(["filing_date", "source_file", "source_row"], nulls_last=True, maintain_order=True)
              .unique(subset=CELL_KEYS, keep="first", maintain_order=True))


def load_facts() -> pd.DataFrame:
    """Hechos XBRL crudos, ya deduplicados a un valor por
    (ticker, concepto, period_type, period_end).

    XBRL repite cada cifra anual en 2-3 filings distintos (los comparativos
    del año anterior). Se toma el valor de la PRIMERA vez que se reportó —
    no la mediana ni el más reciente — porque este panel alimenta joins
    as-of: el valor asignado a un período tiene que ser el que un lector de
    ESE filing podía conocer entonces, nunca uno corregido por una
    reexpresión posterior. Mediana era la regla anterior (robusta a una
    reexpresión aislada, documentada en 05_senal_incremental.md) pero deja
    entrar información del futuro — verificado con DISH FY2021 revenue,
    donde un 10-K de 2024 post-fusión con EchoStar retaggea el período a
    ~10x lo que dos filings previos y mutuamente consistentes ya habían
    reportado; la mediana la habría tomado si el voto hubiera sido 2 a 1 al
    revés. "Primera vez reportado" es correcta pase lo que pase con el
    conteo de votos, porque es la única regla que nunca mira un filing
    posterior al que se está alineando.

    `has_dimensions` se excluye cuando existe: un inline-XBRL fact
    dimensional (contexto con `explicitMember`/`typedMember`, p. ej. el
    desglose por segmento de negocio o geografía) puede compartir el MISMO
    concepto y período que el total consolidado — `us-gaap:Revenues` de
    Amazon FY2021 sólo existía como un contexto dimensional de ~$55M (un
    segmento), no como el total de ~$470M. Sin este filtro esos hechos
    entran junto con — o en vez de — el consolidado y lo corrompen.
    `bronze.xbrl_facts` trae la columna; fuentes que no la
    traen (p. ej. el bulk de Company Facts) no la necesitan porque ya
    devuelven un único valor no dimensional por concepto-período — esas
    mismas fuentes tampoco traen `filing_date` por hecho individual (es un
    bulk pull, no por filing), así que caen de vuelta a la mediana."""
    raw = scan_facts()
    columns = set(raw.collect_schema().names())
    has_dims_col = "has_dimensions" in columns
    base = raw.filter(pl.col("numeric_value").is_not_null() & pl.col("ticker").is_not_null())
    if has_dims_col:
        base = base.filter(pl.col("has_dimensions").fill_null(False).not_())
    if "filing_date" in columns:
        facts = (_first_reported(base.select(CELL_KEYS + ["numeric_value", "filing_date", "source_file", "source_row"]))
                 .select(CELL_KEYS + [pl.col("numeric_value").alias("value"),
                                      pl.col("filing_date").cast(pl.Datetime("us")),
                                      pl.lit(False).alias("is_dimensional")])
                 .collect().to_pandas())
        if has_dims_col:
            facts = pd.concat([facts, _load_dimensional_singletons()], ignore_index=True)
    else:
        facts = (base.group_by(CELL_KEYS)
                     .agg(pl.col("numeric_value").median().alias("value"))
                     .with_columns(pl.lit(None).alias("filing_date"), pl.lit(False).alias("is_dimensional"))
                     .collect().to_pandas())
    # us_by_filing (inline-XBRL) stores period_start/period_end as raw XBRL
    # context strings, not parsed dates; us (company facts) already returns
    # them as datetimes. Normalize so downstream day-count arithmetic works
    # for either source.
    facts["period_start"] = pd.to_datetime(facts["period_start"], errors="coerce")
    facts["period_end"] = pd.to_datetime(facts["period_end"], errors="coerce")
    return facts


def _load_dimensional_singletons() -> pd.DataFrame:
    """Last-resort fallback for (ticker, concept, period) cells where NO
    non-dimensional fact exists at all, but every dimensional context
    tagged for that cell agrees on one value AND that value is
    corroborated by at least 2 observations (the original filing plus at
    least one comparative repeat — the same repetition every other fact
    in this pipeline gets, see `load_facts`'s docstring).

    Some filers (GM, General Dynamics, Sherwin-Williams, ...) tag a metric
    with exactly one dimensional member every period — not a true segment
    breakdown needing summation, just the whole-company figure filed under
    a dimensional context (verified against GM's R&D: $9.8B FY2022,
    $9.9B FY2023, matching its actual reported figures, corroborated by 3
    separate filings). Requiring a SINGLE distinct value across all
    contexts for that cell is NOT enough on its own — verified case: APA's
    us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax for
    FY2022 has exactly ONE dimensional observation ($18M, a single
    product/geography line, not the ~$11B total) which trivially passes
    "single distinct value" simply because there's nothing to disagree
    with it. Requiring >=2 observations rejects one-off, uncorroborated
    tags like that one while still keeping GM's (3+ observations every
    year) — a real multi-segment breakdown (2+ DIFFERENT values, at any
    observation count) is still left NULL rather than guessed at by
    picking or summing arbitrarily. `pivot_metrics()` only reaches for
    this after every non-dimensional concept in the fallback chain has
    nothing — see `is_dimensional` there."""
    dimensional = (scan_facts()
                   .filter(pl.col("numeric_value").is_not_null() & pl.col("ticker").is_not_null()
                           & pl.col("has_dimensions"))
                   .select(CELL_KEYS + ["numeric_value", "filing_date", "source_file", "source_row"])
                   .with_columns(pl.col("numeric_value").n_unique().over(CELL_KEYS).alias("n_distinct"),
                                 pl.len().over(CELL_KEYS).alias("n_obs")))
    return (_first_reported(dimensional)
            .filter((pl.col("n_distinct") == 1) & (pl.col("n_obs") >= 2))
            .select(CELL_KEYS + [pl.col("numeric_value").alias("value"),
                                 pl.col("filing_date").cast(pl.Datetime("us")),
                                 pl.lit(True).alias("is_dimensional")])
            .collect().to_pandas())


def pivot_metrics(facts: pd.DataFrame, metrics: dict[str, list[str]],
                  period_type: str) -> pd.DataFrame:
    """(ticker, period_end) x métrica, resolviendo la cadena de fallback.

    Orden de desempate: PRIMERO `filing_date` (el concepto disponible más
    temprano gana), la cadena de prioridad (`DURATION_METRICS`/
    `INSTANT_METRICS`) sólo rompe empates entre conceptos disponibles el
    MISMO día. Nunca al revés — el caso que lo exige es Iron Mountain
    FY2022 Q1: el 10-Q original de 2022 sólo taggeaba
    `RevenueFromContractWithCustomerExcludingAssessedTax` ($497M);
    `us-gaap:Revenues` para ese mismo trimestre aparece por primera vez
    más de un año después, en 2023, como comparativo reexpresado a ~2.5x
    el valor original. Prioridad-primero habría preferido `Revenues` (más
    arriba en la cadena) y arrastrado esa reexpresión aunque sea posterior
    al trimestre por más de un año — exactamente la fuga que este panel
    existe para evitar. Fecha-primero mantiene el valor que efectivamente
    se conocía en 2022.

    `is_dimensional` va PRIMERO en el desempate, antes que fecha: un hecho
    dimensional de valor único (ver `_load_dimensional_singletons`) sólo
    se usa cuando NINGÚN concepto no-dimensional tiene dato para esa celda,
    sin importar qué tan reciente sea — nunca reemplaza un total real.

    Probado y DESCARTADO (2026-09-08): "el valor más grande gana" como
    desempate de mismo día para `revenue`. Arreglaba General Mills
    (`Revenues` taggeado como subtotal de ~$2.19B en el mismo filing
    donde `RevenueFromContractWithCustomerExcludingAssessedTax` es el
    total real de ~$18.1B) pero rompía Mastercard (su tag
    `RevenueFromContract...ExcludingAssessedTax` es internamente
    inconsistente — el acumulado a 9 meses SUPERA el total anual real —
    mientras `Revenues` telescopa exacto al FY: $5.17B+$5.50B+$5.76B+
    $5.82B=$22.24B) y Philip Morris (`IncludingAssessedTax` es ~2x más
    grande por impuestos al tabaco de traspaso, no por ser "más completo").
    El efecto neto sobre la reconciliación trimestral-vs-anual fue
    NEGATIVO (99.0%→97.7% dentro de 1%). GIS queda como brecha residual
    conocida y aceptada en vez de una heurística que rompe otros
    tickers para arreglarlo — ver `docs/sources/accounting_data.md`."""
    priority = _concept_priority(metrics)
    df = facts[facts["period_type"] == period_type].merge(priority, on="concept")
    if period_type == "duration":
        days = (df["period_end"] - df["period_start"]).dt.days
        df = df[(days >= ANNUAL_MIN_DAYS) & (days <= ANNUAL_MAX_DAYS)]
    df = (df.sort_values(["ticker", "period_end", "metric", "is_dimensional", "filing_date", "priority"])
            .drop_duplicates(["ticker", "period_end", "metric"], keep="first"))
    wide = df.pivot_table(index=["ticker", "period_end"], columns="metric",
                          values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    for metric in metrics:
        if metric not in wide.columns:
            wide[metric] = np.nan
    return wide.sort_values(["ticker", "period_end"]).reset_index(drop=True)


def fill_via_component_sum(facts: pd.DataFrame, duration: pd.DataFrame, target: str,
                            components: dict[str, list[str]], override: bool = False) -> pd.DataFrame:
    """Fill (or, with `override=True`, REPLACE) `duration[target]` by
    summing two complementary concepts, ONLY where BOTH components exist
    for that (ticker, period_end). A filer with just one tagged keeps
    whatever `target` already had (fill mode) or NULL (override mode)
    rather than being filled from that one component alone — a partial
    figure passed off as the full construct is a worse, silently biased
    definition than a missing value (CLAUDE.md: one construct, one
    formula). `components` keys become temporary pivot columns; values are
    each a 1-concept fallback chain reusing `pivot_metrics`'s machinery.

    `override=True` is for when the sum is a verified accounting IDENTITY
    for the construct, not just a same-magnitude proxy — e.g. bank
    revenue = InterestIncomeExpenseNet + NoninterestIncome, checked at
    0.0% difference against every bank that already tags a combined
    `Revenues` concept (BAC/COF/JPM/C/PNC, every year). For those banks,
    `target` is NOT null — it's already (wrongly) populated from a
    concept scoped to ASC 606 fee income only, which structurally
    excludes net interest income (FITB's shows ~$580M against several
    billion in actual revenue). Fill mode would never reach it since
    there's nothing to fill; override mode replaces it with the correct,
    verified total whenever both components are available."""
    parts = pivot_metrics(facts, components, "duration")
    keys = list(components)
    summed = (parts.dropna(subset=keys)
                    .assign(_summed=lambda d: d[keys].sum(axis=1))
                    [["ticker", "period_end", "_summed"]])
    duration = duration.merge(summed, on=["ticker", "period_end"], how="left")
    if override:
        duration[target] = duration["_summed"].where(duration["_summed"].notna(), duration[target])
    else:
        duration[target] = duration[target].fillna(duration["_summed"])
    return duration.drop(columns=["_summed"])


def load_dual_class_shares() -> pd.DataFrame:
    """Fallback `shares_out` for dual/multi-class filers (GOOGL, META,
    BRK.B, F, CMCSA, ...) that tag `dei:EntityCommonStockSharesOutstanding`
    ONLY per share class — one dimensional context per class, no
    non-dimensional total — so `load_facts()`'s has_dimensions filter
    (needed to keep segment breakdowns out of revenue/etc., see
    `load_facts` docstring) correctly excludes them, but then has nothing
    to fall back to. 40 of 510 tickers had zero shares_out without this.

    Verified against GOOGL: exactly 3 dimensional contexts per filing
    (its 3 share classes), summing to a plausible total (~665M pre-split
    2021, ~12.6B post 20:1 split from mid-2022 on — matches GOOGL's known
    share count both sides of the split). Summed per (ticker, period_end,
    accession_number) — never across filings, so a share count is never
    double-counted with a stale prior-filing class figure — then resolved
    to the EARLIEST filing_date per (ticker, period_end), same as-of rule
    as `load_facts`."""
    raw = scan_facts()
    if "has_dimensions" not in raw.collect_schema().names():
        return pd.DataFrame(columns=["ticker", "period_end", "shares_out"])
    # orden por accession_number: dos filings del mismo día para el mismo
    # period_end se resuelven siempre igual en el drop_duplicates de abajo
    per_filing = (raw.filter((pl.col("concept") == "dei:EntityCommonStockSharesOutstanding")
                             & (pl.col("period_type") == "instant") & pl.col("has_dimensions")
                             & pl.col("numeric_value").is_not_null())
                     .group_by(["ticker", "period_end", "accession_number", "filing_date"])
                     .agg(pl.col("numeric_value").sum().alias("value"))
                     .sort(["ticker", "period_end", "filing_date", "accession_number"], nulls_last=True)
                     .collect().to_pandas())
    per_filing["period_end"] = pd.to_datetime(per_filing["period_end"], errors="coerce")
    per_filing["filing_date"] = pd.to_datetime(per_filing["filing_date"], errors="coerce")
    resolved = (per_filing
                .sort_values(["ticker", "period_end", "filing_date"])
                .drop_duplicates(["ticker", "period_end"], keep="first"))
    return resolved.rename(columns={"value": "shares_out"})[["ticker", "period_end", "shares_out"]]


def align_to_filings(filings: pd.DataFrame, annual: pd.DataFrame,
                     metrics: list[str]) -> pd.DataFrame:
    """El corazón del script: para cada 10-K, el FY que reporta y el que
    todavía no había terminado. Ver el docstring del módulo."""
    out = []
    by_ticker = {t: g.reset_index(drop=True) for t, g in annual.groupby("ticker")}
    for row in filings.itertuples(index=False):
        group = by_ticker.get(row.ticker)
        if group is None or group.empty:
            continue
        ends = group["period_end"].values
        cutoff = np.datetime64(row.filing_date + pd.Timedelta(days=DISCLOSED_SLACK_DAYS))
        idx = int(np.searchsorted(ends, cutoff, side="right")) - 1
        if idx < 0:
            continue
        disclosed = group.iloc[idx]
        record = {
            "ticker": row.ticker,
            "year": row.filing_date.year,
            "filing_date": row.filing_date,
            "accession_number": row.accession_number,
            "disclosed_period_end": disclosed["period_end"],
        }
        record.update({m: disclosed.get(m, np.nan) for m in metrics})
        # Growth the 10-K itself discloses: its fiscal year against the previous one.
        record["revenue_yoy"] = np.nan
        if idx >= 1:
            prior = group.iloc[idx - 1]
            prior_gap = (disclosed["period_end"] - prior["period_end"]).days
            base, cur = prior.get("revenue", np.nan), disclosed.get("revenue", np.nan)
            if ANNUAL_MIN_DAYS <= prior_gap <= ANNUAL_MAX_DAYS and pd.notna(base) and pd.notna(cur) and base > 0:
                record["revenue_yoy"] = cur / base - 1
        if idx + 1 < len(group):
            following = group.iloc[idx + 1]
            gap = (following["period_end"] - disclosed["period_end"]).days
            record["next_period_end"] = following["period_end"]
            record["next_gap_days"] = gap
            usable = ANNUAL_MIN_DAYS <= gap <= ANNUAL_MAX_DAYS
            for metric in GROWTH_METRICS:
                base, nxt = disclosed.get(metric, np.nan), following.get(metric, np.nan)
                record[f"next_{metric}_yoy"] = (
                    nxt / base - 1 if usable and pd.notna(base) and pd.notna(nxt)
                    and base > 0 else np.nan)
        else:
            record["next_period_end"] = pd.NaT
            record["next_gap_days"] = np.nan
            for metric in GROWTH_METRICS:
                record[f"next_{metric}_yoy"] = np.nan
        out.append(record)
    return pd.DataFrame(out)


def attach_asof(panel: pd.DataFrame, wide: pd.DataFrame, metrics: list[str],
                left_on: str, tolerance_days: int) -> pd.DataFrame:
    """Pega hechos 'instant' al panel por CERCANÍA de fecha, por ticker.

    Dos anclas distintas y ese es el punto: el balance se ancla al cierre
    fiscal (`disclosed_period_end`), pero `shares_out` es la cifra de portada
    del 10-K y se fecha cerca del FILING. Mergear todo por `period_end`
    exacto hacía que la búsqueda agarrara la fila de portada en vez de la del
    cierre fiscal y perdiera el balance entero (bug documentado en 04_perfiles_economicos.md)."""
    left = panel.sort_values(left_on).copy()
    right = (wide[["ticker", "period_end"] + metrics]
             .dropna(subset=["period_end"]).sort_values("period_end"))
    merged = pd.merge_asof(
        left, right.rename(columns={"period_end": f"_{left_on}_matched"}),
        left_on=left_on, right_on=f"_{left_on}_matched", by="ticker",
        direction="nearest", tolerance=pd.Timedelta(days=tolerance_days))
    return merged.drop(columns=[f"_{left_on}_matched"])


def safe_div(numerator: pd.Series, denominator: pd.Series,
             positive_denominator: bool = True) -> pd.Series:
    """División que devuelve NULL en vez de un ratio sin sentido.

    `positive_denominator` importa para equity y revenue: una empresa con
    patrimonio contable negativo produce un ROE positivo espurio (pérdida
    dividida por equity negativo), y ese número no significa nada."""
    den = denominator.astype(float).copy()
    den = den.where(den > 0) if positive_denominator else den.where(den != 0)
    return numerator.astype(float) / den


def annual_panel() -> pd.DataFrame:
    """One row per US 10-K of the analysis universe: the fiscal year it
    discloses (duration metrics, balance sheet, cover-page shares, ratios,
    sic2), its revenue growth over the prior fiscal year (`revenue_yoy`) and
    the next fiscal year's growth (`next_*_yoy`). Keyed by
    (ticker, year = calendar year of `filing_date`)."""
    filings = (L.scan("silver.filing_manifest")
                .filter((pl.col("country_code") == "us") & (pl.col("form_type") == "10-K")
                        & pl.col("ticker").is_not_null())
                .select("ticker", "cik", pl.col("filing_date").cast(pl.Datetime("us")), "accession_number")
                .sort(["ticker", "filing_date", "accession_number"], nulls_last=True)
                .collect().to_pandas())
    universe = (L.scan("silver.firm_universe")
                 .filter((pl.col("country_code") == "us") & pl.col("ticker").is_not_null())
                 .select("ticker", "sic")
                 .collect().to_pandas())
    facts = load_facts()
    dual_class_shares = load_dual_class_shares()
    facts = facts[~(facts["concept"].isin(NEVER_NEGATIVE_CONCEPTS) & (facts["value"] < 0))]
    for ticker, excluded_concepts in TICKER_CONCEPT_EXCLUSIONS.items():
        facts = facts[~((facts["ticker"] == ticker) & (facts["concept"].isin(excluded_concepts)))]
    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    print(f"{len(filings):,} filings 10-K | {len(facts):,} hechos XBRL "
          f"({facts['ticker'].nunique():,} tickers)")

    duration = pivot_metrics(facts, DURATION_METRICS, "duration")
    duration = fill_via_component_sum(facts, duration, "da", DA_COMPONENTS)
    duration = fill_via_component_sum(facts, duration, "pretax_income", PRETAX_COMPONENTS)
    duration = fill_via_component_sum(facts, duration, "revenue", BANK_REVENUE_COMPONENTS, override=True)
    instants = pivot_metrics(facts, INSTANT_METRICS, "instant")
    shares = pivot_metrics(facts, SHARES_METRIC, "instant")
    # A public filer never genuinely has zero shares outstanding; a 0 here
    # is a filer tagging error, not a fact — verified against Ball Corp's
    # own 2022-02-16 10-K, which tagged its cover-page share count as
    # exactly 0.0 (every adjacent filing shows ~310-330M). Treated as
    # missing so it doesn't produce spurious zero market cap / infinite
    # EPS-type ratios downstream.
    shares.loc[shares["shares_out"] <= 0, "shares_out"] = np.nan
    shares = (shares.merge(dual_class_shares, on=["ticker", "period_end"],
                           how="outer", suffixes=("", "_dual_class"))
                     .assign(shares_out=lambda d: d["shares_out"].fillna(d["shares_out_dual_class"]))
                     .drop(columns=["shares_out_dual_class"]))
    print(f"anuales: {len(duration):,} (ticker, period_end) | "
          f"instantáneos: {len(instants):,} | shares: {len(shares):,}")

    panel = align_to_filings(filings, duration, list(DURATION_METRICS))
    print(f"alineados: {len(panel):,} filas ({panel['ticker'].nunique():,} tickers)")
    gap = panel["next_gap_days"].dropna()
    print(f"gap al FY siguiente dentro de [340,380]: "
          f"{(gap.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)).mean()*100:.1f}% "
          f"({int((~gap.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)).sum())} filas con "
          f"next_*_yoy anulado)")

    ratios = attach_asof(panel, instants, list(INSTANT_METRICS),
                         "disclosed_period_end", INSTANT_TOLERANCE_DAYS)
    ratios = attach_asof(ratios, shares, ["shares_out"],
                         "filing_date", SHARES_TOLERANCE_DAYS)

    ratios["gross_margin"] = safe_div(ratios["revenue"] - ratios["cost_of_revenue"],
                                      ratios["revenue"])
    ratios["operating_margin"] = safe_div(ratios["operating_income"], ratios["revenue"])
    ratios["net_margin"] = safe_div(ratios["net_income"], ratios["revenue"])
    ratios["roa"] = safe_div(ratios["net_income"], ratios["total_assets"])
    ratios["roe"] = safe_div(ratios["net_income"], ratios["equity"])
    ratios["current_ratio"] = safe_div(ratios["current_assets"], ratios["current_liabilities"])
    ratios["debt_to_equity"] = safe_div(ratios["long_term_debt"], ratios["equity"])
    ratios["asset_turnover"] = safe_div(ratios["revenue"], ratios["total_assets"])
    ratios["rd_intensity"] = safe_div(ratios["rd_expense"], ratios["revenue"])
    ratios["capex_intensity"] = safe_div(ratios["capex"], ratios["revenue"])
    # Sin D&A reportada el EBITDA queda nulo: gold no imputa D&A = 0.
    ratios["ebitda"] = ratios["operating_income"] + ratios["da"]
    universe["sic2"] = universe["sic"].astype("string").str.zfill(4).str[:2]
    ratios = ratios.merge(universe[["ticker", "sic2"]].drop_duplicates("ticker"),
                          on="ticker", how="left")

    coverage = {c: f"{ratios[c].notna().mean()*100:.0f}%"
                for c in ["revenue", "rd_expense", "capex", "sga_expense",
                          "operating_income", "total_assets", "equity",
                          "long_term_debt", "shares_out", "next_revenue_yoy"]}
    print("cobertura:", coverage)
    return ratios


FS_GROWTH_METRICS = ["revenue", "rd_expense", "capex", "sga_expense"]


def annual_panel_fs() -> pd.DataFrame:
    """fs (FactSet) replacement for `annual_panel()`: one row per fs annual
    STND period matched to a filing date (docs/plans/fs_gold_replacement.md).
    Same output columns/semantics as `annual_panel()` -- `year` is the
    CALENDAR year of `filing_date` (matches the firm_year spine's join key),
    not the fiscal year. Source: `silver.fs_financials` (period="annual"),
    already point-in-time dated (EDGAR filing_date primary, fs release-date
    fallback, `pit_source` column) and scale-corrected
    (scripts/sources/fs/build_fundamentals_wide.py)."""
    fin = (L.scan("silver.fs_financials")
           .filter((pl.col("period") == "annual") & pl.col("filing_date_pt").is_not_null())
           .collect().to_pandas())
    fin["filing_date_pt"] = pd.to_datetime(fin["filing_date_pt"])
    fin["period_end"] = pd.to_datetime(fin["period_end"])
    fin = fin.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    fin = fin.rename(columns={"filing_date_pt": "filing_date", "period_end": "disclosed_period_end"})
    fin["year"] = fin["filing_date"].dt.year

    # Two fs periods can resolve to the same (ticker, year) mostly in
    # pre-corpus history (before ~2020) where the EDGAR-fallback release
    # date is looser -- the spine only ever needs one fiscal year per
    # (ticker, calendar filing year), so keep the most RECENT disclosed
    # fiscal year for that year (the one an actual 10-K filed that calendar
    # year would report), same one-row-per-filing intent as the XBRL-era
    # annual_panel(), which never had this collision because it iterated
    # actual 10-K filings one at a time.
    fin = (fin.sort_values(["ticker", "year", "disclosed_period_end"])
           .drop_duplicates(["ticker", "year"], keep="last")
           .sort_values(["ticker", "disclosed_period_end"]).reset_index(drop=True))

    grp = fin.groupby("ticker", sort=False)
    prior_end = grp["disclosed_period_end"].shift(1)
    prior_gap = (fin["disclosed_period_end"] - prior_end).dt.days
    prior_revenue = grp["revenue"].shift(1)
    fin["revenue_yoy"] = np.where(
        prior_gap.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS) & prior_revenue.notna() & (prior_revenue > 0),
        fin["revenue"] / prior_revenue - 1, np.nan)

    next_end = grp["disclosed_period_end"].shift(-1)
    next_gap = (next_end - fin["disclosed_period_end"]).dt.days
    fin["next_period_end"] = next_end
    fin["next_gap_days"] = next_gap
    usable_next = next_gap.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)
    for metric in FS_GROWTH_METRICS:
        base, nxt = fin[metric], grp[metric].shift(-1)
        fin[f"next_{metric}_yoy"] = np.where(usable_next & base.notna() & nxt.notna() & (base > 0),
                                             nxt / base - 1, np.nan)

    fin["gross_profit"] = fin["gross_profit"] if "gross_profit" in fin.columns else np.nan
    fin["gross_margin"] = safe_div(fin["revenue"] - fin["cost_of_revenue"], fin["revenue"])
    fin["operating_margin"] = safe_div(fin["operating_income"], fin["revenue"])
    fin["net_margin"] = safe_div(fin["net_income"], fin["revenue"])
    fin["roa"] = safe_div(fin["net_income"], fin["total_assets"])
    fin["roe"] = safe_div(fin["net_income"], fin["equity"])
    fin["current_ratio"] = safe_div(fin["current_assets"], fin["current_liabilities"])
    fin["debt_to_equity"] = safe_div(fin["long_term_debt"], fin["equity"])
    fin["asset_turnover"] = safe_div(fin["revenue"], fin["total_assets"])
    fin["rd_intensity"] = safe_div(fin["rd_expense"], fin["revenue"])
    fin["capex_intensity"] = safe_div(fin["capex"], fin["revenue"])
    # ebitda is fs's own STND field (direct read), not operating_income + da
    # (kept from build_firm_financials.annual_panel's XBRL-era formula) --
    # fs reports EBITDA natively per docs/sources/fs.md's field mapping.

    universe = (L.scan("silver.firm_universe")
                .filter((pl.col("country_code") == "us") & pl.col("ticker").is_not_null())
                .select("ticker", "sic").collect().to_pandas())
    universe["sic2"] = universe["sic"].astype("string").str.zfill(4).str[:2]
    fin = fin.merge(universe[["ticker", "sic2"]].drop_duplicates("ticker"), on="ticker", how="left")

    coverage = {c: f"{fin[c].notna().mean()*100:.0f}%"
                for c in ["revenue", "rd_expense", "capex", "sga_expense",
                          "operating_income", "total_assets", "equity",
                          "long_term_debt", "shares_out", "next_revenue_yoy"]}
    print("fs annual_panel cobertura:", coverage)
    print("pit_source:", fin["pit_source"].value_counts(dropna=False).to_dict())
    return fin
