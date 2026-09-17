"""Posture archetype weights across the full outcome family -- now the
classic-literature specification of docs/plans/cap5_classic_specs.md (beta
SHIFT, weekly crash risk, CAR/drift, three firm fundamentals) -- Benjamini-
Hochberg over the archetype tests, and two robustness checks on the
crash-risk archetype weights (sector exclusion, posture switchers: calls at
which the firm's dominant archetype has already changed, known at the call).

Design, panel and estimation: scripts/analytics/call_regression.py. Controls:
the outcome's pre-call level (CAR/drift have none: an event-study abnormal
return has no natural pre-call level of itself), log market cap, 60-day
return and ROA (the control that is itself the outcome is dropped); the two
weekly crash-risk outcomes add the Kim-Li-Zhang (2011) controls DTURN, SIGMA,
RET, MB and LEV; SIC2 x year fixed effects; errors clustered by firm.

Outcomes (docs/plans/cap5_classic_specs.md has the exact windows/citations):
  beta_shift_delta     Brenner (1979) pooled beta shift (Delta-beta), pre-control beta_shift_pre
  ncskew_wk_post       Chen-Hong-Stein (2001) NCSKEW, weekly, 26 post-call weeks
  duvol_wk_post        Chen-Hong-Stein (2001) DUVOL, weekly, 26 post-call weeks
  car_m1_p1            Brown-Warner (1985) CAR[-1,+1]
  car_p2_p63           MacKinlay (1997) post-announcement drift, CAR[+2,+63]
  gross_margin_ttm_change_pp, roic_minus_wacc_ttm_change_pp, revenue_growth_ttm_post_pct
                       unchanged: already symmetric (four fiscal quarters after the
                       call's own quarter vs the four before)

Benjamini-Hochberg family: the full grid actually reported in the thesis --
all six disclosure coefficients (w_call, w_expanding, intensity_expanding,
has_posture, w_voc, w_gov) for each of the eight outcomes (48 tests), q =
0.05, step-up cutoff at the critical value of the LAST rank that passes.
This must match, term for term, the inline recomputation in
thesis_document/thesis.qmd (Market Relevance chapter, `_grid`), which
derives the same table directly from `call_archetype_full_battery_targets.csv`.

Outputs (data/results/crash_archetypes/):
  call_archetype_full_battery_targets.csv     coefficients per outcome
  call_archetype_full_battery_bh.csv          BH table over the full 48-coefficient disclosure family
  call_archetype_full_battery_robustness.csv  sector exclusion / switchers, NCSKEW and DUVOL (weekly)

Usage:
  .venv/bin/python scripts/analytics/crash_archetypes/call_archetype_full_battery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analytics"))
import call_regression as C  # noqa: E402

L = C.L
OUT_DIR = L.results_path("crash_archetypes", "call_archetype_full_battery_targets.csv").parent
# A level a firm carries (a margin, a spread) is reported as the CHANGE it
# records, in percentage points. Asset turnover was measured and left out: its
# post-call level is 97% explained by its pre-call level, so the design has
# nothing to find there and the outcome would only tighten the correction
# threshold for the rest. `None` pre-control: an event-study abnormal return
# (CAR) has no pre-call level of the same construct to lag.
TARGETS = {
    "beta_shift_delta": ("beta_shift_pre", "Beta shift (Brenner 1979 Delta-beta)"),
    "ncskew_wk_post": ("ncskew_wk_pre", "Negative return skewness (NCSKEW, weekly)"),
    "duvol_wk_post": ("duvol_wk_pre", "Down-to-up volatility (DUVOL, weekly)"),
    "car_m1_p1": (None, "Cumulative abnormal return [-1,+1]"),
    "car_p2_p63": (None, "Post-announcement drift, CAR[+2,+63]"),
    "gross_margin_ttm_change_pp": ("gross_margin_ttm_pre", "Gross margin expansion (pp, TTM)"),
    "roic_minus_wacc_ttm_change_pp": ("roic_minus_wacc_ttm_pre", "Change in ROIC-WACC spread (pp, TTM)"),
    "revenue_growth_ttm_post_pct": ("revenue_growth_ttm_pre_pct", "Revenue growth (%, TTM)"),
}
# Kim-Li-Zhang (2011) controls, added only to the two weekly crash-risk
# outcomes: DTURN (detrended weekly share turnover), SIGMA/RET (already read
# off the SAME pre-call weekly residuals NCSKEW/DUVOL use), MB, LEV. SIZE and
# ROA are already in C.BASE_CTRLS.
EXTRA_CONTROLS = {"ncskew_wk_post": ["dturn", "sigma_wk_pre", "ret_wk_pre", "mb", "lev"],
                  "duvol_wk_post": ["dturn", "sigma_wk_pre", "ret_wk_pre", "mb", "lev"]}
EXCLUDED_SECTORS = ["Financial Services", "Utilities", "Healthcare & Pharma"]


def attach_classic(panel: pd.DataFrame) -> pd.DataFrame:
    """The classic-literature covariates/targets of
    scripts/gold/call/build_market_financials.py, plus the KLZ controls
    derived from them (docs/plans/cap5_classic_specs.md)."""
    panel = C.attach(panel, "covariates", "market",
                     ["beta_shift_pre", "ncskew_wk_pre", "duvol_wk_pre", "sigma_wk_pre", "ret_wk_pre",
                      "avg_volume_wk_pre", "avg_volume_wk_prior26"])
    panel = C.attach(panel, "targets", "market",
                     ["beta_shift_delta", "car_m1_p1", "car_p2_p63", "ncskew_wk_post", "duvol_wk_post"])
    panel = C.attach(panel, "covariates", "financials", ["equity_pre", "long_term_debt_pre"])
    panel["mb"] = np.exp(panel["log_market_cap"]) / panel["equity_pre"].where(panel["equity_pre"] > 0)
    panel["lev"] = panel["long_term_debt_pre"] / panel["total_assets"].where(panel["total_assets"] > 0)
    panel["dturn"] = (panel["avg_volume_wk_pre"] - panel["avg_volume_wk_prior26"]) / panel["shares_out"].where(panel["shares_out"] > 0)
    return panel


def sector_map() -> dict[str, str]:
    fu = (L.scan("silver.firm_universe").filter(pl.col("ticker").is_not_null())
          .select("ticker", "sic").collect().to_pandas())
    fu["sic2"] = fu["sic"].astype(str).str.zfill(4).str[:2]

    def _map(sic2):
        try:
            s = int(sic2)
        except ValueError:
            return "Other / Diversified"
        if s in (28, 38, 80):
            return "Healthcare & Pharma"
        if 60 <= s <= 67:
            return "Financial Services"
        if s == 49:
            return "Utilities"
        return "Other / Diversified"

    fu["sector"] = fu["sic2"].apply(_map)
    return fu.set_index("ticker")["sector"].to_dict()


def switched_by_call(panel: pd.DataFrame) -> pd.Series:
    """True for a call once its firm has shown two different dominant
    archetypes (argmax of the as-of weights) at that call or earlier ones: a
    call with posture whose archetype differs from the archetype at some
    earlier call, and the firm's later calls, including calls without posture
    (the weights' comparison group)."""
    order = panel.sort_values(["ticker", "fecha"]).index
    has_posture = panel[C.WEIGHTS].sum(axis=1) > 0
    arch = pd.Series(panel[C.WEIGHTS].values.argmax(axis=1), index=panel.index).where(has_posture)
    seen = pd.concat([(arch == j).loc[order].groupby(panel.loc[order, "ticker"]).cummax()
                      for j in range(len(C.WEIGHTS))], axis=1)
    return (seen.sum(axis=1) >= 2).reindex(panel.index)


def regressors_for(y_col: str) -> list[str]:
    pre_col, _ = TARGETS[y_col]
    return C.AI_VARS + ([pre_col] if pre_col else []) + C.controls_for(y_col) + EXTRA_CONTROLS.get(y_col, [])


def fit_target(panel: pd.DataFrame, y_col: str):
    return C.fit(panel, y_col, regressors_for(y_col))


def posture_contrasts(result, target: str, label: str) -> pd.DataFrame:
    """The four comparisons the composition supports. Each is a REALLOCATION of
    weight between vertices holding everything else constant, not a comparison
    between firms labelled by their dominant archetype: a firm that is half
    Defensive and half Vocal enters both coefficients at half weight, and a
    coefficient describes moving a whole unit of weight from one vertex to
    another, which almost no firm does.

    Governance vs Vocal is a difference of two estimated coefficients, so it
    needs the covariance between them (a linear restriction, not arithmetic on
    the table): its significance does not follow from the two p-values."""
    rows = [("Defensive vs. no posture", "has_posture = 0"),
            ("Vocal vs. Defensive", "w_voc = 0"),
            ("Governance vs. Defensive", "w_gov = 0"),
            ("Governance vs. Vocal", "w_gov - w_voc = 0")]
    out = []
    for name, restriction in rows:
        t = result.t_test(restriction)
        out.append({"target": target, "label_outcome": label, "contrast": name,
                    "beta": float(np.squeeze(t.effect)), "se": float(np.squeeze(t.sd)),
                    "p": float(np.squeeze(t.pvalue))})
    joint = result.f_test("w_voc = 0, w_gov = 0")
    out.append({"target": target, "label_outcome": label, "contrast": "Composition (joint)",
                "beta": float("nan"), "se": float("nan"), "p": float(joint.pvalue),
                "F": float(np.squeeze(joint.fvalue)), "df_num": int(joint.df_num),
                "df_den": int(joint.df_denom)})
    return pd.DataFrame(out)


def regression_report(result, d, target: str, label: str, fes=("sic2", "anio")) -> str:
    """The statsmodels summary an appendix can carry: every estimated term
    except the fixed effects, which are collapsed into one line each.

    Printing the dummies themselves would add 58 rows per outcome and say
    nothing -- what a reader needs from an absorbed block is whether it is
    jointly different from zero, so each block reports its size and its own F
    test."""
    keep = [c for c in result.params.index if not any(c.startswith(f"{f}_") for f in fes)]
    body = pd.DataFrame({"coef": result.params[keep], "std err": result.bse[keep],
                         "t": result.tvalues[keep], "P>|t|": result.pvalues[keep],
                         "[0.025": result.conf_int().loc[keep, 0],
                         "0.975]": result.conf_int().loc[keep, 1]}).round(4)
    lines = [f"{label}  ({target})",
             f"  observations {len(d):,} | firms {d['ticker'].nunique()} | "
             f"R2 {result.rsquared:.4f} | adj. R2 {result.rsquared_adj:.4f} | "
             f"SE clustered by firm", "", body.to_string(), ""]
    for f in fes:
        cols = [c for c in result.params.index if c.startswith(f"{f}_")]
        if not cols:
            continue
        test = result.f_test(" = 0, ".join(cols) + " = 0")
        lines.append(f"  {f} fixed effects: {len(cols)} dummies absorbed | "
                     f"F = {float(test.fvalue):.2f}, p = {float(test.pvalue):.2e}")
    return "\n".join(lines) + "\n" + "=" * 78 + "\n"


def main() -> None:
    panel = attach_classic(C.attach_fundamentals(C.attach_crash_risk(C.load_call_panel())))
    panel["sector"] = panel["ticker"].map(sector_map()).fillna("Other / Diversified")

    tables, contrasts, reports = [], [], []
    for y_col, (pre_col, label) in TARGETS.items():
        res, d = fit_target(panel, y_col)
        regressors = regressors_for(y_col)
        pr2 = C.partial_r2(panel, y_col, regressors, C.AI_VARS)
        # every estimated term the table reports: the disclosure block, the
        # outcome's own lag (when it has one) and the controls. Fixed effects
        # are absorbed and reported as present rather than enumerated.
        tables.append(C.coef_table(res, C.AI_VARS + C.REPORTED_WEIGHTS + regressors[len(C.AI_VARS):],
                                   target=y_col, label_outcome=label, n_calls=len(d),
                                   n_firms=d.ticker.nunique(), r2=res.rsquared, partial_r2_ai_arch=pr2,
                                   pre_control=pre_col or ""))
        contrasts.append(posture_contrasts(res, y_col, label))
        reports.append(regression_report(res, d, y_col, label))
        print(f"{label:22s} n={len(d):5d} firms={d.ticker.nunique():4d} r2={res.rsquared:.3f} pR2={pr2:.4f} | "
              + " | ".join(f"{v} {res.params[v]:+.3f} (p={res.pvalues[v]:.3f})" for v in C.REPORTED_WEIGHTS))
    targets = pd.concat(tables, ignore_index=True)
    targets.to_csv(OUT_DIR / "call_archetype_full_battery_targets.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(OUT_DIR / "call_archetype_posture_contrasts.csv", index=False)
    (OUT_DIR / "call_archetype_regression_report.txt").write_text("\n".join(reports))

    # The family actually reported in the thesis: all six disclosure
    # coefficients (the three AI-disclosure regressors plus the three
    # reported posture terms) across all eight outcomes -- 48 tests. This
    # must match, term for term, the inline recomputation in
    # thesis_document/thesis.qmd (Market Relevance chapter, `_grid`): same
    # variable list, same q = 0.05, same ranking, same step-up cutoff (the
    # critical value at the LAST rank that passes, not the first).
    bh_vars = C.AI_VARS + C.REPORTED_WEIGHTS
    fam = targets[targets["variable"].isin(bh_vars)][["target", "variable", "beta_std", "p"]]
    fam = fam.sort_values("p").reset_index(drop=True)
    m = len(fam)
    fam["rank"] = np.arange(1, m + 1)
    fam["bh_crit"] = fam["rank"] / m * 0.05
    # step-up: every test ranked at or below the largest rank with p <= its critical value survives
    passing = np.where(fam["p"].values <= fam["bh_crit"].values)[0]
    fam["survives"] = fam["rank"] <= (passing.max() + 1 if len(passing) else 0)
    fam.to_csv(OUT_DIR / "call_archetype_full_battery_bh.csv", index=False)
    print(f"\n=== Benjamini-Hochberg across the {m}-test disclosure family ===\n{fam.head(10).round(4).to_string(index=False)}")

    samples = {
        "sector_exclusion": panel[~panel["sector"].isin(EXCLUDED_SECTORS)],
        "posture_switchers": panel[switched_by_call(panel)],
    }
    robust = []
    for check, sample in samples.items():
        for y_col in ("ncskew_wk_post", "duvol_wk_post"):
            res, d = fit_target(sample, y_col)
            robust.append(C.coef_table(res, C.AI_VARS + C.REPORTED_WEIGHTS, check=check, target=y_col,
                                       n_calls=len(d), n_firms=d.ticker.nunique(), r2=res.rsquared))
            print(f"{check:18s} {y_col:12s} n={len(d)} firms={d.ticker.nunique()} | "
                  + " | ".join(f"{v} {res.params[v]:+.3f} (p={res.pvalues[v]:.3f})" for v in C.REPORTED_WEIGHTS))
    pd.concat(robust, ignore_index=True).to_csv(OUT_DIR / "call_archetype_full_battery_robustness.csv", index=False)


if __name__ == "__main__":
    main()
