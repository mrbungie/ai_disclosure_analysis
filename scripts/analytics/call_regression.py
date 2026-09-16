"""
scripts/analytics/call_regression.py — the call-level regression design shared
by every earnings-call regression of the Market Relevance chapter
(call_beta_regressions, call_beta_robustness, call_beta_generalized_targets,
call_crash_regressions, call_archetype_full_battery, call_beta_config_decoupling_asof).

Panel (`load_call_panel`): one row per call of the call spine
(spines/call/call: ticker, fecha, SIC2 x year cell `fe`) with the call's
disclosure (covariates/call/disclosure), pre-call market and accounting
controls (covariates/call/market: price_pre, return60, beta_pre,
log_market_cap; covariates/call/financials: last 10-K before the call), the
post-call betas (targets/call/market) and its point-in-time predictors:
  - `w`, `hist_w` (covariates/call/disclosure): decoupling of the call
    (percentile ranks against calls of earlier quarters) and the firm's mean
    decoupling over its earlier calls.
  - `intensity_expanding`: frames per 1,000 words over every 10-K, 8-K and
    DEF 14A published up to the last closed quarter
    (covariates/firm_quarter/disclosure_volume: posture_expanding_frames_per_1k).
  - `w_voc`, `w_gov`, `w_def`: posture archetype weights over the trailing 12
    months up to the last closed quarter
    (covariates/firm_quarter/posture_archetype: posture_ttm_w_*).
The panel reads datasets/call/call, where the firm-quarter covariates are
attached as of the call date (fq__ columns, pit.asof_join); calls before the
first closed quarter are dropped.

Analytics fills (gold leaves them null): a firm without posture frames has
intensity 0 and the three weights 0, and `has_posture` marks the firms that
do have one. Models carry `has_posture`, `w_voc` and `w_gov`, holding
Defensive Disclosers as the reference: `w_voc` is the difference from shifting
weight out of Defensive into Vocal at constant Governance, `w_gov` the same
into Governance, and a NEGATIVE coefficient means less tail risk than
Defensive. Omitting the Defensive weight discards nothing -- it is determined
by the other two and the dummy -- and the three postures are still
represented.

Weights stay on their own 0-1 scale (they are never standardized), so ten
points of weight moved out of Defensive is 0.1 x beta of the outcome.

Dropping to two weights plus the dummy does not change the fit or the
predictions; it changes the question each coefficient answers, from "does this
posture differ from having no posture at all" to "does it differ from Vocal".
Carrying all three weights alongside a constant asks neither: the weights sum
to one wherever a posture exists, so with the constant the design loses rank
and OLS returns the minimum-norm solution among infinitely many. The
individual coefficients then track the parameterization rather than the data
-- measured, `w_def` moves from +0.059 to -0.071 between two fits of identical
R2 -- while contrasts between weights stay invariant.

Fixed effects are ADDITIVE sector and year rather than their interaction. The
interaction spent 307 parameters on SIC2 x year cells, some holding two calls
of one firm, whose dummies then absorb that firm's outcome; sector and year
separately cost 58, leave no degenerate cell, and answer the same question.
Sector is the block that carries the variance (F ~ 100, p < 1e-200 on both
crash-risk targets); year is smaller but real (F 6.5 to 10.6).

Estimation (`fit`): outcome and continuous regressors standardized on the
estimation sample, weights left in [0, 1]; standard errors clustered by firm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers as L  # noqa: E402
from fills import VALUE_COMPONENTS, roic_wacc_debt_as_zero  # noqa: E402

AI_VARS = ["w_call", "w_expanding", "intensity_expanding"]
# gold names them `w` and `hist_w`; renamed here because the pair is only
# readable side by side: `w_call` is this call's decoupling, `w_expanding`
# the mean over the firm's earlier calls. Two thirds of the latter's variance
# is between firms (measured), so it describes a firm's habit and `w_call` a
# single event.
RENAMED = {"w": "w_call", "hist_w": "w_expanding"}
WEIGHTS = ["w_voc", "w_gov", "w_def"]
REFERENCE_WEIGHT = "w_def"  # Defensive Disclosers: the posture the others are read against
# what a model actually estimates for the composition: the extensive margin and
# the two shifts away from the reference posture
REPORTED_WEIGHTS = ["has_posture"] + [w for w in WEIGHTS if w != REFERENCE_WEIGHT]
BASE_CTRLS = ["log_market_cap", "return60", "roa"]
LABELS = {
    "w_call": "Call decoupling (W_call)",
    "w_expanding": "Historical decoupling (W_expanding)",
    "has_posture": "Discloses AI in filings",
    "intensity_expanding": "Cumulative disclosure intensity",
    "w_voc": "Vocal Substantives weight",
    "w_gov": "Governance-Led weight",
    "w_def": "Defensive weight",
}

WEIGHTS_VARIANT = "posture_ttm"
INTENSITY_COLUMN = "posture_expanding_frames_per_1k"
# Coverage, not a model output: AI frames in the firm's 10-K, 8-K and DEF 14A
# of the trailing year. `has_posture` reads this, so the extensive margin is a
# fact about the corpus rather than a by-product of the archetype fit.
POSTURE_FRAMES_COLUMN = "posture_ttm_n_frames"
FUNDAMENTAL_TARGETS = ["log_market_cap", "gross_margin", "ps_ratio", "next_revenue_yoy", "roic_minus_wacc"]
# The pre-call level that controls each `{target}_post` outcome. Revenue growth
# of the fiscal year after the last pre-call 10-K is not known at the call; its
# control is the growth that 10-K reports over its prior fiscal year.
PRE_CONTROL = {**{t: f"{t}_pre" for t in FUNDAMENTAL_TARGETS}, "next_revenue_yoy": "revenue_yoy_pre"}
# Outcomes reported as a change from the pre-call level, in percentage points.
CHANGE_TARGETS = ("gross_margin", "roic_minus_wacc")
WINSOR = 0.01  # each tail of the change outcomes
# Where each `{stem}_pre` / `{target}_post` column lives. The pre-call log market
# cap is covariates/call/market.log_market_cap.
FUNDAMENTAL_FAMILIES = {
    "_pre": {"gross_margin": ("covariates", "financials"),
             "ps_ratio": ("covariates", "market"), "revenue_yoy": ("covariates", "financials"),
             "roic_minus_wacc": ("covariates", "financials")},
    "_post": {"log_market_cap": ("targets", "market"), "gross_margin": ("targets", "financials"), "ps_ratio": ("targets", "market"),
              "next_revenue_yoy": ("targets", "financials"), "roic_minus_wacc": ("targets", "financials")},
}


# Firm-quarter snapshot attached to the call in datasets/call/call (fq__ columns).
SNAPSHOT_COLUMNS = {"fq__quarter": "quarter", "fq__as_of_date": "as_of_date",
                    **{f"fq__{WEIGHTS_VARIANT}_{w}": w for w in WEIGHTS},
                    f"fq__{INTENSITY_COLUMN}": "intensity_expanding",
                    f"fq__{POSTURE_FRAMES_COLUMN}": "posture_frames_ttm"}


def load_call_panel() -> pd.DataFrame:
    panel = L.read_dataset("call", ("covariates", "disclosure"),
                           ("covariates", "market", ["price_pre", "return60", "beta_pre", "log_market_cap"]),
                           ("covariates", "financials", ["accession_number", "filing_date_pt", "revenue",
                                                         "operating_income", "total_assets", "operating_margin",
                                                         "asset_turnover", "roa", "shares_out"]),
                           ("targets", "market", ["beta_post_63", "beta_post_126", "beta_post_252"]),
                           columns=list(SNAPSHOT_COLUMNS)).rename(columns=SNAPSHOT_COLUMNS)
    panel = panel[panel["quarter"].notna()].copy()
    panel[WEIGHTS + ["intensity_expanding"]] = panel[WEIGHTS + ["intensity_expanding"]].fillna(0.0)
    # 1 when the firm filed AI frames in the trailing year, 0 otherwise: the
    # extensive margin, read from the corpus so the weights carry composition
    # alone and the dummy does not depend on the archetype model
    panel = panel.rename(columns=RENAMED)
    panel["has_posture"] = (panel["posture_frames_ttm"].fillna(0.0) > 0).astype(float)
    # measured: 8 firm-quarters (NSC 2024, XOM 2021) carry weights with no frames
    # behind them, at roughly the centroid. A weight without a frame is not a
    # measurement, so the composition follows the coverage it was fitted on.
    panel.loc[panel["has_posture"] == 0, WEIGHTS] = 0.0
    panel["sic2"] = panel["fe"].astype(str).str.split("_").str[0]
    panel["anio"] = pd.to_datetime(panel["fecha"]).dt.year.astype(str)
    return panel.reset_index(drop=True)


def attach(panel: pd.DataFrame, kind: str, name: str, columns: list[str],
           rename: dict[str, str] | None = None) -> pd.DataFrame:
    """Left-join columns of the call family `kind`/`name` (one row per call) on
    call_accession_number, read from datasets/call/call."""
    extra = L.read_dataset("call", (kind, name, columns), spine_columns=["call_accession_number"])
    extra = extra.rename(columns=rename or {})
    return panel.merge(extra, on="call_accession_number", how="left", validate="one_to_one")


def attach_crash_risk(panel: pd.DataFrame) -> pd.DataFrame:
    panel = attach(panel, "covariates", "market", ["ncskew_pre", "duvol_pre"])
    return attach(panel, "targets", "market", ["ncskew_post_105d", "duvol_post_105d"],
                  {"ncskew_post_105d": "ncskew_post", "duvol_post_105d": "duvol_post"})


def attach_fundamentals(panel: pd.DataFrame) -> pd.DataFrame:
    """PRE_CONTROL / `{target}_post` for FUNDAMENTAL_TARGETS; ROIC - WACC reads
    unreported long-term debt as zero debt (fills.roic_wacc_debt_as_zero)."""
    panel = panel.assign(log_market_cap_pre=panel["log_market_cap"])
    for suffix, families in FUNDAMENTAL_FAMILIES.items():
        value_family = families["roic_minus_wacc"]
        for (kind, name) in dict.fromkeys(families.values()):
            cols = [f"{t}{suffix}" for t, fam in families.items() if fam == (kind, name)]
            cols += [f"{c}{suffix}" for c in VALUE_COMPONENTS] if (kind, name) == value_family else []
            panel = attach(panel, kind, name, cols)
        panel[f"roic_minus_wacc{suffix}"] = roic_wacc_debt_as_zero(panel, suffix)[f"roic_minus_wacc{suffix}"]
        panel = panel.drop(columns=[f"{c}{suffix}" for c in VALUE_COMPONENTS])
    # A margin and a spread are levels a firm carries, so what a call could
    # plausibly move is the CHANGE in them, in percentage points; the level is
    # kept as its own control. Winsorized at 1%: both changes are ratios whose
    # denominator can be small, which sends a handful of firm-quarters to
    # hundreds of points (gross margin reached +627pp, the spread -599pp) and
    # lets them set a coefficient on their own.
    for stem in CHANGE_TARGETS:
        change = 100.0 * (panel[f"{stem}_post"] - panel[f"{stem}_pre"])
        lo, hi = change.quantile([WINSOR, 1 - WINSOR])
        panel[f"{stem}_change_pp"] = change.clip(lo, hi)
    return panel


def controls_for(outcome: str, controls: list[str] = BASE_CTRLS) -> list[str]:
    """Controls minus the one that is the outcome itself (its pre level already enters)."""
    stem = outcome.removesuffix("_post_63").removesuffix("_post")
    return [c for c in controls if c != stem]


def _standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(lambda c: (c - c.mean()) / c.std(ddof=0))


def fit(panel: pd.DataFrame, outcome: str, regressors: list[str], raw: list[str] = WEIGHTS,
        fe_cols: tuple[str, ...] = ("sic2", "anio"), min_fe_size: int = 2, fe_col: str | None = None):
    """OLS of `outcome` on standardized `regressors` plus raw (unstandardized) `raw`
    and additive `fe_cols` fixed effects, errors clustered by firm.

    Carrying the weights means carrying `has_posture` with them and dropping
    the reference weight, so each remaining weight is a shift away from
    Defensive Disclosers. Passing `fe_col` estimates a single interacted cell instead
    (kept for the robustness table that varies the fixed-effect scheme).

    Returns (result, estimation sample)."""
    raw = [c for c in raw if c not in regressors]
    if set(WEIGHTS) <= set(raw):
        raw = ["has_posture"] + [c for c in raw if c != REFERENCE_WEIGHT]
    fes = (fe_col,) if fe_col else fe_cols
    d = panel.dropna(subset=[outcome, *fes, *regressors, *raw]).copy()
    for f in fes:
        d = d[d.groupby(f)["ticker"].transform("size") >= min_fe_size].copy()
    y = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=0)
    blocks = [_standardize(d, regressors), d[raw]]
    blocks += [pd.get_dummies(d[f], prefix=f, drop_first=True, dtype=float) for f in fes]
    design = sm.add_constant(pd.concat(blocks, axis=1))
    return sm.OLS(y, design).fit(cov_type="cluster", cov_kwds={"groups": d["ticker"]}), d


def r2(result, sample: pd.DataFrame) -> float:
    return float(result.rsquared)


def coef_table(result, variables: list[str], **meta) -> pd.DataFrame:
    rows = []
    for v in variables:
        b, se = float(result.params[v]), float(result.bse[v])
        rows.append({**meta, "variable": v, "label": LABELS.get(v, v), "beta_std": b, "se": se,
                     "ci95_low": b - 1.96 * se, "ci95_high": b + 1.96 * se, "p": float(result.pvalues[v])})
    return pd.DataFrame(rows)


def sample_row(result, sample: pd.DataFrame, **meta) -> dict:
    return {**meta, "n_calls": len(sample), "n_firms": sample["ticker"].nunique(),
            "n_fe_cells": sample["fe"].nunique(), "r2": r2(result, sample)}


def partial_r2(panel: pd.DataFrame, outcome: str, regressors: list[str], block: list[str],
               raw: list[str] = WEIGHTS) -> float:
    """Within-R2 gain of `block` (+ `raw`) over the remaining regressors, on the full model's sample."""
    full, d = fit(panel, outcome, regressors, raw)
    reduced, _ = fit(d, outcome, [r for r in regressors if r not in block], raw=[])
    return float(full.rsquared - reduced.rsquared)
