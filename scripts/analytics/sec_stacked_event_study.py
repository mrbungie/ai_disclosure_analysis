"""Stacked cohort event study around direct SEC comment letters on AI disclosure.

Reproduces the regulatory-scrutiny analysis of the thesis (Chapter 5 and
Appendix F) on the analysis universe (S&P 500 as of 2021-01-01, i.e. the
`firm_universe` view): every treated issuer and every matched control must be
a member of that panel, so a letter addressed to a firm that joined the index
later is dropped from the treated set.

Design
------
* Unit: annual report (Form 10-K). Event time k = 0 is the first 10-K filed
  after the letter date; k = -1 is the last 10-K before it (reference cycle).
* Controls: for each treated issuer, the three firms in the same two-digit SIC
  industry closest in Mahalanobis distance on pre-letter (k = -1) log market
  capitalisation, AI claim density, promotional density and the k-2 -> k-1
  change in claim density. Controls' 10-Ks are aligned to the treated firm's
  event calendar by filing year.
* Estimation: stacked OLS with cohort x firm and cohort x filing-year fixed
  effects and treated x event-time interactions (k = -1 omitted); standard
  errors clustered by firm. The static difference-in-differences coefficient
  (treated x post) is also reported with a randomisation-inference p-value
  obtained by re-drawing, within every cohort, which of its four members is
  labelled treated.
* Outcomes: AI claim density (frames per 1,000 words), the disclosure
  dimension targeted by each letter (specificity, quantified metrics or
  governance per 1,000 words, stacked as one outcome), the firm specificity
  index, Decoupling (W, firm-year) and promotional density.

Deterministic (fixed seed), no LLM. Reads duckdb/thesis.duckdb and
data/processed/clusters/{document_panel,firm_year_washing_score}.parquet;
writes data/processed/sec_*.json|parquet consumed by thesis.qmd.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
CLUSTERS = REPO_ROOT / "data" / "processed" / "clusters"
OUT = REPO_ROOT / "data" / "processed"
SEED = 20260911
N_CONTROLS = 3
N_PERMS = 2000
WINDOW_MAIN = [-2, -1, 0, 1]
WINDOW_EXT = [-2, -1, 0, 1, 2]

# Letters identified by the EDGAR correspondence search (UPLOAD / CORRESP),
# 2021 through 2026-05-27. `dimension` names the disclosure dimension the
# staff comment targeted and the per-1k-words column that measures it.
LETTERS = [
    {"ticker": "META", "company": "Meta Platforms, Inc.", "date": "2022-11-21", "form": "10-K",
     "topic": "Operational architecture & mechanisms of AI recommendation engine",
     "dimension": "System Specificity (spec_per_1k)"},
    {"ticker": "HPE", "company": "Hewlett Packard Enterprise", "date": "2023-04-20", "form": "10-K",
     "topic": "Quantify revenue drivers (price vs. volume) in HPC & AI segment",
     "dimension": "Quantified Metrics (quant_per_1k)"},
    {"ticker": "NVDA", "company": "NVIDIA Corporation", "date": "2023-06-29", "form": "10-K",
     "topic": "ASC 606 revenue recognition for AI cloud software arrangements",
     "dimension": "Quantified Metrics (quant_per_1k)"},
    {"ticker": "AMD", "company": "Advanced Micro Devices", "date": "2023-10-04", "form": "10-K",
     "topic": "Quantified data center AI revenue targets & accelerator deliveries",
     "dimension": "Quantified Metrics (quant_per_1k)"},
    {"ticker": "DIS", "company": "The Walt Disney Company", "date": "2024-02-27", "form": "10-K",
     "topic": "Factual basis & quantitative support for AI impact on stock price",
     "dimension": "Quantified Metrics (quant_per_1k)"},
    {"ticker": "MTCH", "company": "Match Group, Inc.", "date": "2024-04-15", "form": "10-K",
     "topic": "Board & management oversight, user safety & AI governance",
     "dimension": "Governance Oversight (gov_per_1k)"},
    {"ticker": "ANET", "company": "Arista Networks", "date": "2024-06-25", "form": "10-K",
     "topic": "AI networking revenue quantification and customer concentration",
     "dimension": "Quantified Metrics (quant_per_1k)"},
    {"ticker": "IRM", "company": "Iron Mountain", "date": "2024-09-12", "form": "10-K",
     "topic": "AI-related data center demand and quantified backlog",
     "dimension": "Quantified Metrics (quant_per_1k)"},
    {"ticker": "WELL", "company": "Welltower Inc.", "date": "2025-04-28", "form": "10-K",
     "topic": "Status of AI integration and data science platform claims across venues",
     "dimension": "System Specificity (spec_per_1k)"},
]
DIM_COL = {"spec_per_1k": "spec_density", "quant_per_1k": "quant_density", "gov_per_1k": "gov_density",
           "realized_per_1k": "realized_density"}
OUTCOMES = {
    "claim_density": "AI Claim Density (per 1k words)",
    "targeted_detail": "SEC-Targeted Disclosure Detail (per 1k words)",
    "specificity_index": "Firm Specificity Index [0, 1]",
    "w": "Decoupling Index (W) [-1, 1]",
    "promo_density": "Promotional Density (per 1k words)",
}


def load_inventory_topics() -> dict[str, dict]:
    """Prefer the topics recorded in the existing inventory file (they were
    transcribed from the letters); fall back to the constants above."""
    path = OUT / "sec_comment_letters_inventory.json"
    known = {d["ticker"]: d for d in LETTERS}
    if path.exists():
        for d in json.loads(path.read_text()):
            if d["ticker"] in known:
                known[d["ticker"]].update({k: d[k] for k in ("company", "date", "form", "topic", "dimension") if k in d})
    return known


def load_10k_panel() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    universe = con.execute("SELECT ticker, sic FROM firm_universe WHERE country_code = 'us'").df()
    con.close()
    universe["sic2"] = universe["sic"].astype(str).str.zfill(4).str[:2]
    docs = pd.read_parquet(CLUSTERS / "document_panel.parquet")
    docs = docs[(docs["form"] == "10-K") & docs["ticker"].isin(universe["ticker"])].copy()
    docs["filing_date"] = pd.to_datetime(docs["fecha"])
    docs["year"] = docs["filing_date"].dt.year
    docs = docs.sort_values(["ticker", "filing_date"]).drop_duplicates(["ticker", "year"], keep="last")
    per_1k = 1000.0 / docs["n_words"].replace(0, np.nan)
    docs["claim_density"] = docs["n_frames"] * per_1k
    docs["promo_density"] = docs["n_promo"] * per_1k
    docs["quant_density"] = docs["n_quant"] * per_1k
    docs["spec_density"] = docs["n_spec"] * per_1k
    docs["gov_density"] = docs["n_gov"] * per_1k
    docs["realized_density"] = docs["n_realized"] * per_1k
    docs["specificity_index"] = np.where(docs["n_frames"] > 0, docs["n_spec"] / docs["n_frames"].replace(0, np.nan), np.nan)
    w = pd.read_parquet(CLUSTERS / "firm_year_washing_score.parquet")[["ticker", "year", "w"]]
    docs = docs.merge(w, on=["ticker", "year"], how="left")
    mcap_path = OUT / "sec_filings_panel_for_matching.parquet"
    if mcap_path.exists():
        mc = pd.read_parquet(mcap_path)[["accession_number", "market_cap"]].drop_duplicates("accession_number")
        docs = docs.merge(mc, on="accession_number", how="left")
    if "market_cap" not in docs or docs["market_cap"].isna().all():
        master = pd.read_parquet(CLUSTERS / "firm_year_master_v2.parquet")[["ticker", "year", "market_cap"]]
        docs = docs.drop(columns=["market_cap"], errors="ignore").merge(master, on=["ticker", "year"], how="left")
    else:
        master = pd.read_parquet(CLUSTERS / "firm_year_master_v2.parquet")[["ticker", "year", "market_cap"]].rename(columns={"market_cap": "_mc2"})
        docs = docs.merge(master, on=["ticker", "year"], how="left")
        docs["market_cap"] = docs["market_cap"].fillna(docs["_mc2"])
        docs = docs.drop(columns=["_mc2"])
    docs["log_mcap"] = np.log(docs["market_cap"].where(docs["market_cap"] > 0))
    docs = docs.merge(universe[["ticker", "sic2"]], on="ticker", how="left")
    keep = ["accession_number", "ticker", "filing_date", "year", "n_words", "n_frames", "claim_density", "promo_density",
            "quant_density", "spec_density", "gov_density", "realized_density", "specificity_index", "w", "sic2", "market_cap", "log_mcap"]
    return docs[keep].reset_index(drop=True)


def event_calendar(panel: pd.DataFrame, ticker: str, letter_date: pd.Timestamp) -> dict[int, int] | None:
    """Map event time k -> filing year for a treated firm; None if no post-letter 10-K."""
    f = panel[panel["ticker"] == ticker].sort_values("filing_date")
    post = f[f["filing_date"] > letter_date]
    pre = f[f["filing_date"] <= letter_date]
    if post.empty or pre.empty:
        return None
    years_pre = pre["year"].tolist()[::-1]      # k = -1, -2, ...
    years_post = post["year"].tolist()          # k = 0, 1, ...
    cal = {}
    for i, y in enumerate(years_post):
        cal[i] = int(y)
    for i, y in enumerate(years_pre):
        cal[-(i + 1)] = int(y)
    return cal


def mahalanobis_controls(panel: pd.DataFrame, treated: str, cal: dict[int, int], treated_set: set[str]) -> list[str]:
    y_m1, y_m2 = cal[-1], cal.get(-2)
    sic2 = panel.loc[panel["ticker"] == treated, "sic2"].iloc[0]

    def features(sub: pd.DataFrame) -> pd.DataFrame:
        a = sub[sub["year"] == y_m1].set_index("ticker")
        b = sub[sub["year"] == y_m2].set_index("ticker") if y_m2 is not None else None
        out = pd.DataFrame({"log_mcap": a["log_mcap"], "claim": a["claim_density"], "promo": a["promo_density"]})
        out["dclaim"] = (a["claim_density"] - b["claim_density"].reindex(a.index)) if b is not None else 0.0
        return out.dropna()

    pool = panel[(panel["sic2"] == sic2) & (~panel["ticker"].isin(treated_set))]
    feats = features(pool)
    if len(feats) < N_CONTROLS + 1:
        feats = features(panel[~panel["ticker"].isin(treated_set)])
    tfeat = features(panel[panel["ticker"] == treated])
    if tfeat.empty:
        return []
    X = feats.values
    cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    inv = np.linalg.pinv(cov)
    diff = X - tfeat.values[0]
    d = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
    order = np.argsort(d)
    return [feats.index[i] for i in order[:N_CONTROLS]]


def build_stack(panel: pd.DataFrame, letters: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    treated_set = set(letters)
    rows, inventory = [], []
    for t, meta in letters.items():
        if t not in set(panel["ticker"]):
            print(f"  {t}: not in the analysis universe -> excluded")
            continue
        cal = event_calendar(panel, t, pd.Timestamp(meta["date"]))
        if cal is None:
            print(f"  {t}: no post-letter 10-K yet -> excluded")
            continue
        controls = mahalanobis_controls(panel, t, cal, treated_set)
        dim_col = DIM_COL[meta["dimension"].split("(")[1].rstrip(")")]
        for k, y in cal.items():
            if k not in WINDOW_EXT:
                continue
            for tk in [t] + controls:
                r = panel[(panel["ticker"] == tk) & (panel["year"] == y)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append({"cohort": t, "ticker": tk, "is_treated": int(tk == t), "event_k": k,
                             "filing_date": r["filing_date"], "filing_year": int(y),
                             "claim_density": r["claim_density"], "promo_density": r["promo_density"],
                             "targeted_detail": r[dim_col], "specificity_index": r["specificity_index"], "w": r["w"]})
        inv = {k: meta[k] for k in ("ticker", "company", "date", "form", "topic", "dimension")}
        inv["controls"] = ", ".join(controls)
        inventory.append(inv)
        print(f"  {t}: k=0 in {cal[0]}, controls {controls}")
    return pd.DataFrame(rows), inventory


def fit_event_study(stack: pd.DataFrame, outcome: str, window: list[int]) -> dict:
    d = stack[stack["event_k"].isin(window)].dropna(subset=[outcome]).copy()
    d["cf"] = d["cohort"] + ":" + d["ticker"]
    d["cy"] = d["cohort"] + ":" + d["filing_year"].astype(str)
    X = pd.get_dummies(d[["cf", "cy"]], drop_first=True).astype(float)
    ks = [k for k in window if k != -1]
    for k in ks:
        X[f"k{k}"] = (d["is_treated"] * (d["event_k"] == k)).astype(float)
    X = sm.add_constant(X)
    res = sm.OLS(d[outcome].values, X.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["ticker"])[0]})
    names = list(X.columns)
    coef = {k: float(res.params[names.index(f"k{k}")]) for k in ks}
    se = {k: float(res.bse[names.index(f"k{k}")]) for k in ks}
    p = {k: float(res.pvalues[names.index(f"k{k}")]) for k in ks}
    n_treated = {k: int(((d["is_treated"] == 1) & (d["event_k"] == k)).sum()) for k in window}
    return {"coef": coef, "se": se, "p": p, "n_treated": n_treated, "N": int(len(d)), "n_clusters": int(d["ticker"].nunique())}


def fit_static(stack: pd.DataFrame, outcome: str, window: list[int], rng: np.random.Generator) -> dict:
    d = stack[stack["event_k"].isin(window)].dropna(subset=[outcome]).copy()
    d["cf"] = d["cohort"] + ":" + d["ticker"]
    d["cy"] = d["cohort"] + ":" + d["filing_year"].astype(str)
    X0 = pd.get_dummies(d[["cf", "cy"]], drop_first=True).astype(float)
    post = (d["event_k"] >= 0).astype(float).values
    y = d[outcome].values
    groups = pd.factorize(d["ticker"])[0]

    def est(treated_flag: np.ndarray, clustered: bool = False):
        X = X0.copy()
        X["did"] = treated_flag * post
        X = sm.add_constant(X)
        if clustered:
            r = sm.OLS(y, X.values).fit(cov_type="cluster", cov_kwds={"groups": groups})
            j = list(X.columns).index("did")
            return float(r.params[j]), float(r.bse[j]), float(r.pvalues[j])
        beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
        return float(beta[list(X.columns).index("did")])

    b, s, p_cl = est(d["is_treated"].values.astype(float), clustered=True)
    # randomisation inference: within each cohort, relabel one member as treated
    members = d.groupby("cohort")["ticker"].unique().to_dict()
    perm_b = []
    for _ in range(N_PERMS):
        fake = {c: rng.choice(m) for c, m in members.items()}
        flag = np.array([1.0 if row.ticker == fake[row.cohort] else 0.0 for row in d.itertuples()])
        perm_b.append(est(flag))
    perm_b = np.array(perm_b)
    p_ri = float((np.abs(perm_b) >= abs(b)).mean())
    return {"post_b": b, "post_se": s, "p_cluster": p_cl, "p_ri": p_ri}


def main() -> None:
    rng = np.random.default_rng(SEED)
    letters = load_inventory_topics()
    panel = load_10k_panel()
    print(f"10-K panel: {len(panel):,} filings, {panel['ticker'].nunique()} firms")
    stack, inventory = build_stack(panel, letters)
    stack.to_parquet(OUT / "sec_stacked_cohort_event_study.parquet", index=False)
    (OUT / "sec_comment_letters_inventory.json").write_text(json.dumps(inventory, indent=2))
    panel.to_parquet(OUT / "sec_filings_panel_for_matching.parquet", index=False)

    def panel_json(keys: list[str], window: list[int]) -> dict:
        out = {}
        for key in keys:
            es = fit_event_study(stack, key, window)
            out[key] = {"label": OUTCOMES[key], "k": window,
                        "coef": [0.0 if k == -1 else es["coef"][k] for k in window],
                        "se": [0.0 if k == -1 else es["se"][k] for k in window],
                        "n_treated": [es["n_treated"][k] for k in window]}
        return out

    (OUT / "sec_stacked_event_study_targeted_2panel.json").write_text(json.dumps(panel_json(["claim_density", "targeted_detail"], WINDOW_MAIN), indent=2))
    (OUT / "sec_stacked_event_study_main_4period.json").write_text(json.dumps(panel_json(["claim_density", "specificity_index"], WINDOW_MAIN), indent=2))
    (OUT / "sec_stacked_event_study_2panel.json").write_text(json.dumps(panel_json(["claim_density", "specificity_index"], WINDOW_EXT), indent=2))

    multi = {}
    for key in OUTCOMES:
        es = fit_event_study(stack, key, WINDOW_MAIN)
        st = fit_static(stack, key, WINDOW_MAIN, rng)
        multi[key] = {"label": OUTCOMES[key], "N": es["N"], "n_clusters": es["n_clusters"],
                      "pre_b": es["coef"][-2], "pre_se": es["se"][-2], "pre_p": es["p"][-2],
                      "k0_b": es["coef"][0], "k0_se": es["se"][0], "k0_p": es["p"][0],
                      "k1_b": es["coef"][1], "k1_se": es["se"][1], "k1_p": es["p"][1],
                      **st, "n_treated": es["n_treated"]}
        print(f"  {key}: post_b={st['post_b']:+.4f} (se {st['post_se']:.4f}) p_cluster={st['p_cluster']:.3f} p_ri={st['p_ri']:.3f} N={es['N']}")
    (OUT / "sec_stacked_multivariate_results.json").write_text(json.dumps(multi, indent=2))
    print(f"-> {OUT}/sec_*.json, sec_stacked_cohort_event_study.parquet ({len(inventory)} treated issuers)")


if __name__ == "__main__":
    main()
