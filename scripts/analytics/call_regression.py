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
intensity 0 and the three weights 0. All three weights enter every model:
they sum to 1 for firms with posture and to 0 otherwise, so no category is
omitted and each weight's coefficient is measured against firms without
posture.

Estimation (`fit`): outcome and continuous regressors standardized on the
estimation sample, weights left in [0, 1]; fixed effects as dummy columns
(cells with fewer than two calls dropped); standard errors clustered by firm.
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

AI_VARS = ["w", "hist_w", "intensity_expanding"]
WEIGHTS = ["w_voc", "w_gov", "w_def"]
BASE_CTRLS = ["log_market_cap", "return60", "roa"]
LABELS = {
    "w": "Call decoupling (W)",
    "hist_w": "Historical decoupling (HistW)",
    "intensity_expanding": "Cumulative disclosure intensity",
    "w_voc": "Vocal Substantives weight",
    "w_gov": "Governance-Led weight",
    "w_def": "Defensive weight",
}

WEIGHTS_VARIANT = "posture_ttm"
INTENSITY_COLUMN = "posture_expanding_frames_per_1k"
FUNDAMENTAL_TARGETS = ["log_market_cap", "rd_intensity", "gross_margin", "ps_ratio", "next_revenue_yoy", "roic_minus_wacc"]
# The pre-call level that controls each `{target}_post` outcome. Revenue growth
# of the fiscal year after the last pre-call 10-K is not known at the call; its
# control is the growth that 10-K reports over its prior fiscal year.
PRE_CONTROL = {**{t: f"{t}_pre" for t in FUNDAMENTAL_TARGETS}, "next_revenue_yoy": "revenue_yoy_pre"}
# Where each `{stem}_pre` / `{target}_post` column lives. The pre-call log market
# cap is covariates/call/market.log_market_cap.
FUNDAMENTAL_FAMILIES = {
    "_pre": {"rd_intensity": ("covariates", "financials"), "gross_margin": ("covariates", "financials"),
             "ps_ratio": ("covariates", "market"), "revenue_yoy": ("covariates", "financials"),
             "roic_minus_wacc": ("covariates", "financials")},
    "_post": {"log_market_cap": ("targets", "market"), "rd_intensity": ("targets", "financials"),
              "gross_margin": ("targets", "financials"), "ps_ratio": ("targets", "market"),
              "next_revenue_yoy": ("targets", "financials"), "roic_minus_wacc": ("targets", "financials")},
}


# Firm-quarter snapshot attached to the call in datasets/call/call (fq__ columns).
SNAPSHOT_COLUMNS = {"fq__quarter": "quarter", "fq__as_of_date": "as_of_date",
                    **{f"fq__{WEIGHTS_VARIANT}_{w}": w for w in WEIGHTS},
                    f"fq__{INTENSITY_COLUMN}": "intensity_expanding"}


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
    return panel


def controls_for(outcome: str, controls: list[str] = BASE_CTRLS) -> list[str]:
    """Controls minus the one that is the outcome itself (its pre level already enters)."""
    stem = outcome.removesuffix("_post_63").removesuffix("_post")
    return [c for c in controls if c != stem]


def _standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(lambda c: (c - c.mean()) / c.std(ddof=0))


def fit(panel: pd.DataFrame, outcome: str, regressors: list[str], raw: list[str] = WEIGHTS,
        fe_col: str = "fe", min_fe_size: int = 2):
    """OLS of `outcome` on standardized `regressors` plus raw (unstandardized) `raw`
    and `fe_col` fixed effects, errors clustered by firm. Returns (result, estimation sample)."""
    raw = [c for c in raw if c not in regressors]
    d = panel.dropna(subset=[outcome, fe_col, *regressors, *raw]).copy()
    d = d[d.groupby(fe_col)["ticker"].transform("size") >= min_fe_size].copy()
    y = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=0)
    fe = pd.get_dummies(d[fe_col], prefix="fe", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([_standardize(d, regressors), d[raw], fe], axis=1))
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
