"""Posture archetype weights across the full outcome family (post-call beta,
six firm fundamentals, NCSKEW/DUVOL), Benjamini-Hochberg over the archetype
tests, and two robustness checks on the crash-risk archetype weights (sector
exclusion, posture switchers: calls at which the firm's dominant archetype has
already changed, known at the call).

Design, panel and estimation: scripts/analytics/call_regression.py. Controls:
the outcome's pre-call level, log market cap, 60-day return and ROA (the
control that is itself the outcome is dropped); SIC2 x year fixed effects;
errors clustered by firm.

Archetype test family: the three weight coefficients (each against firms
without posture) for each of the nine outcomes (27 tests).

Outputs (data/results/crash_archetypes/):
  call_archetype_full_battery_targets.csv     coefficients per outcome
  call_archetype_full_battery_bh.csv          BH table over the 27 weight coefficients
  call_archetype_full_battery_robustness.csv  sector exclusion / switchers, NCSKEW and DUVOL

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
# A level a firm carries (beta, size, a multiple, a skewness) is reported as
# its post-call level with the pre-call level as its own control; a margin and
# a spread, which move little and revert, are reported as the CHANGE they
# record, in percentage points. Asset turnover was measured and left out: its
# post-call level is 97% explained by its pre-call level, so the design has
# nothing to find there and the outcome would only tighten the correction
# threshold for the rest.
TARGETS = {
    "beta_post_63": ("beta_pre", "Market beta"),
    "log_market_cap_post": ("log_market_cap_pre", "Log market capitalization"),
    "gross_margin_change_pp": ("gross_margin_pre", "Gross margin expansion (pp)"),
    "roic_minus_wacc_change_pp": ("roic_minus_wacc_pre", "Change in ROIC-WACC spread (pp)"),
    "next_revenue_yoy_post": ("revenue_yoy_pre", "Revenue growth (%)"),
    "ps_ratio_post": ("ps_ratio_pre", "Price-to-sales ratio"),
    "ncskew_post": ("ncskew_pre", "Negative return skewness (NCSKEW)"),
    "duvol_post": ("duvol_pre", "Down-to-up volatility (DUVOL)"),
}
EXCLUDED_SECTORS = ["Financial Services", "Utilities", "Healthcare & Pharma"]


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


def fit_target(panel: pd.DataFrame, y_col: str):
    return C.fit(panel, y_col, C.AI_VARS + [TARGETS[y_col][0]] + C.controls_for(y_col))


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
                "beta": float("nan"), "se": float("nan"), "p": float(joint.pvalue)})
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
    panel = C.attach_fundamentals(C.attach_crash_risk(C.load_call_panel()))
    panel["sector"] = panel["ticker"].map(sector_map()).fillna("Other / Diversified")

    tables, contrasts, reports = [], [], []
    for y_col, (pre_col, label) in TARGETS.items():
        res, d = fit_target(panel, y_col)
        regressors = C.AI_VARS + [pre_col] + C.controls_for(y_col)
        pr2 = C.partial_r2(panel, y_col, regressors, C.AI_VARS)
        # every estimated term the table reports: the disclosure block, the
        # outcome's own lag and the controls. Fixed effects are absorbed and
        # reported as present rather than enumerated.
        tables.append(C.coef_table(res, C.AI_VARS + C.REPORTED_WEIGHTS + [pre_col] + C.controls_for(y_col),
                                   target=y_col, label_outcome=label, n_calls=len(d),
                                   n_firms=d.ticker.nunique(), r2=res.rsquared, partial_r2_ai_arch=pr2,
                                   pre_control=pre_col))
        contrasts.append(posture_contrasts(res, y_col, label))
        reports.append(regression_report(res, d, y_col, label))
        print(f"{label:22s} n={len(d):5d} firms={d.ticker.nunique():4d} r2={res.rsquared:.3f} pR2={pr2:.4f} | "
              + " | ".join(f"{v} {res.params[v]:+.3f} (p={res.pvalues[v]:.3f})" for v in C.REPORTED_WEIGHTS))
    targets = pd.concat(tables, ignore_index=True)
    targets.to_csv(OUT_DIR / "call_archetype_full_battery_targets.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(OUT_DIR / "call_archetype_posture_contrasts.csv", index=False)
    (OUT_DIR / "call_archetype_regression_report.txt").write_text("\n".join(reports))

    fam = targets[targets["variable"].isin(C.REPORTED_WEIGHTS[1:])][["target", "variable", "beta_std", "p"]]
    fam = fam.sort_values("p").reset_index(drop=True)
    m = len(fam)
    fam["rank"] = np.arange(1, m + 1)
    fam["bh_crit"] = fam["rank"] / m * 0.05
    # step-up: every test ranked at or below the largest rank with p <= its critical value survives
    passing = np.where(fam["p"].values <= fam["bh_crit"].values)[0]
    fam["survives"] = fam["rank"] <= (passing.max() + 1 if len(passing) else 0)
    fam.to_csv(OUT_DIR / "call_archetype_full_battery_bh.csv", index=False)
    print(f"\n=== Benjamini-Hochberg across the {m}-test archetype family ===\n{fam.head(6).round(4).to_string(index=False)}")

    samples = {
        "sector_exclusion": panel[~panel["sector"].isin(EXCLUDED_SECTORS)],
        "posture_switchers": panel[switched_by_call(panel)],
    }
    robust = []
    for check, sample in samples.items():
        for y_col in ("ncskew_post", "duvol_post"):
            res, d = fit_target(sample, y_col)
            robust.append(C.coef_table(res, C.AI_VARS + C.REPORTED_WEIGHTS, check=check, target=y_col,
                                       n_calls=len(d), n_firms=d.ticker.nunique(), r2=res.rsquared))
            print(f"{check:18s} {y_col:12s} n={len(d)} firms={d.ticker.nunique()} | "
                  + " | ".join(f"{v} {res.params[v]:+.3f} (p={res.pvalues[v]:.3f})" for v in C.REPORTED_WEIGHTS))
    pd.concat(robust, ignore_index=True).to_csv(OUT_DIR / "call_archetype_full_battery_robustness.csv", index=False)


if __name__ == "__main__":
    main()
