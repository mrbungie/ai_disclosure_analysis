"""SEC comment letters that touch corporate AI disclosure: verified inventory and case comparisons.

Population. A full-text search of every SEC staff comment letter (form UPLOAD)
and company response (CORRESP) filed 2022-01-01 .. 2026-09-11 for the terms
"artificial intelligence", "AI", "machine learning", "generative", "large
language" and "AI-driven/powered/enabled", restricted to the analysis universe
(S&P 500 as of 2021-01-01), returns correspondence for eight issuers. Reading
the staff letters themselves (not the company responses) gives the
classification recorded in LETTERS below:

* direct      - the staff comment questions the firm's own AI claims;
* adjacent    - the comment requires AI-related revenue disaggregation but
                does not question AI claims;
* incidental  - AI appears only as evidence cited inside a comment on another
                subject;
* none        - the correspondence matched an AI term for reasons unrelated to
                AI disclosure (segment names, page codes, climate letters,
                third-party proxy materials).

Only the direct case (Welltower) supports a before/after comparison of the
firm's own AI disclosure; the adjacent case (Arista) is reported alongside it.
With one and two treated firms there is no regression to estimate: the script
reports the treated firm's annual-report measures around the letter next to
the mean of three industry peers matched on pre-letter size, AI claim density,
promotional density and the change in claim density (Mahalanobis nearest
neighbours within the two-digit SIC industry).

Deterministic, no LLM. Reads duckdb/thesis.duckdb and
data/processed/clusters/{document_panel,firm_year_washing_score,firm_year_master_v2}.parquet;
writes data/processed/sec_comment_letter_cases.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
CLUSTERS = REPO_ROOT / "data" / "processed" / "clusters"
OUT = REPO_ROOT / "data" / "processed" / "sec_comment_letter_cases.json"
N_CONTROLS = 3
KS = [-2, -1, 0, 1]

# Verified against the staff letters on EDGAR (accession numbers of the UPLOAD filings).
LETTERS = [
    {"ticker": "META", "company": "Meta Platforms", "date": "2022-11-21", "accession": "0000000000-22-012584",
     "filing": "10-Q Q3 2022", "subject": "Drivers of the decline in advertising demand and steps to replace lost ad revenue; the AI discovery engine discussed on the Q3 call is cited as one such step",
     "relevance": "incidental", "targeted": None},
    {"ticker": "HPE", "company": "Hewlett Packard Enterprise", "date": "2023-04-20", "accession": "0000000000-23-003991",
     "filing": "10-K FY2022", "subject": "Quantification of price, volume and other drivers of segment revenue (including the HPC & AI segment); non-GAAP transformation costs",
     "relevance": "none", "targeted": None},
    {"ticker": "NVDA", "company": "NVIDIA", "date": "2023-06-15", "accession": "0000000000-23-006470",
     "filing": "10-K FY2023", "subject": "Quantification of MD&A drivers, inflation, segment operating income, revenue recognition for multi-year cloud service agreements",
     "relevance": "none", "targeted": None},
    {"ticker": "AMD", "company": "Advanced Micro Devices", "date": "2023-09-08", "accession": "0000000000-23-009979",
     "filing": "10-K FY2022", "subject": "Climate-related risk disclosure (regulation, water, reputational risk, carbon credits)",
     "relevance": "none", "targeted": None},
    {"ticker": "DIS", "company": "Walt Disney", "date": "2024-02-27", "accession": "0000000000-24-002194",
     "filing": "DEFA14A (Blackwells Capital)", "subject": "Support for stock-price targets in an activist's soliciting materials premised on greater use of AI; addressed to the activist, not to Disney's own disclosure",
     "relevance": "none", "targeted": None},
    {"ticker": "MTCH", "company": "Match Group", "date": "2024-04-15", "accession": None,
     "filing": "10-K FY2023", "subject": "Board and management oversight of AI and user safety (outside the analysis universe; not re-verified)",
     "relevance": "outside universe", "targeted": None},
    {"ticker": "ANET", "company": "Arista Networks", "date": "2024-07-25", "accession": "0000000000-24-008424",
     "filing": "10-K FY2023", "subject": "Key business metrics, gross-margin quantification, receivables; segment reporting, resolved (2024-10-16) by requiring revenue disclosure by customer sector including 'Cloud and AI Titans'",
     "relevance": "adjacent", "targeted": "quant_density"},
    {"ticker": "IRM", "company": "Iron Mountain", "date": "2025-03-25", "accession": "0000000000-25-003212",
     "filing": "10-K FY2024", "subject": "Segment expense disclosure under ASC 280 (Adjusted EBITDA, other reportable segment expenses)",
     "relevance": "none", "targeted": None},
    {"ticker": "WELL", "company": "Welltower", "date": "2025-04-28", "accession": "0000000000-25-004473",
     "filing": "10-K FY2024", "subject": "Requests discussion of the data science platform and the status of AI-integration efforts, contrasting the 10-K statement with the 'industry-leading' platform claimed on the Q4 2024 call and in a press release",
     "relevance": "direct", "targeted": "realized_density"},
]
MEASURES = ["claim_density", "promo_density", "quant_density", "spec_density", "realized_density", "specificity_index", "w"]


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
    docs["realized_density"] = docs["n_realized"] * per_1k
    docs["specificity_index"] = np.where(docs["n_frames"] > 0, docs["n_spec"] / docs["n_frames"].replace(0, np.nan), np.nan)
    w = pd.read_parquet(CLUSTERS / "firm_year_washing_score.parquet")[["ticker", "year", "w"]]
    docs = docs.merge(w, on=["ticker", "year"], how="left")
    master = pd.read_parquet(CLUSTERS / "firm_year_master_v2.parquet")[["ticker", "year", "market_cap"]]
    docs = docs.merge(master, on=["ticker", "year"], how="left")
    docs["log_mcap"] = np.log(docs["market_cap"].where(docs["market_cap"] > 0))
    return docs.merge(universe[["ticker", "sic2"]], on="ticker", how="left").reset_index(drop=True)


def event_calendar(panel: pd.DataFrame, ticker: str, letter_date: pd.Timestamp) -> dict[int, int]:
    f = panel[panel["ticker"] == ticker].sort_values("filing_date")
    post, pre = f[f["filing_date"] > letter_date], f[f["filing_date"] <= letter_date]
    cal = {i: int(y) for i, y in enumerate(post["year"].tolist())}
    cal.update({-(i + 1): int(y) for i, y in enumerate(pre["year"].tolist()[::-1])})
    return cal


def matched_controls(panel: pd.DataFrame, treated: str, cal: dict[int, int], exclude: set[str]) -> list[str]:
    y1, y2 = cal[-1], cal.get(-2)
    sic2 = panel.loc[panel["ticker"] == treated, "sic2"].iloc[0]

    def feats(sub):
        a = sub[sub["year"] == y1].set_index("ticker")
        b = sub[sub["year"] == y2].set_index("ticker") if y2 is not None else None
        f = pd.DataFrame({"log_mcap": a["log_mcap"], "claim": a["claim_density"], "promo": a["promo_density"]})
        f["dclaim"] = (a["claim_density"] - b["claim_density"].reindex(a.index)) if b is not None else 0.0
        return f.dropna()

    pool = feats(panel[(panel["sic2"] == sic2) & (~panel["ticker"].isin(exclude))])
    if len(pool) < N_CONTROLS:
        pool = feats(panel[~panel["ticker"].isin(exclude)])
    t = feats(panel[panel["ticker"] == treated])
    inv = np.linalg.pinv(np.cov(pool.values, rowvar=False) + np.eye(4) * 1e-6)
    diff = pool.values - t.values[0]
    d = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
    return [pool.index[i] for i in np.argsort(d)[:N_CONTROLS]]


def case_table(panel: pd.DataFrame, treated: str, cal: dict[int, int], controls: list[str]) -> dict:
    out = {"k": [], "year": [], "treated": {m: [] for m in MEASURES}, "controls_mean": {m: [] for m in MEASURES},
           "controls": controls}
    for k in KS:
        if k not in cal:
            continue
        y = cal[k]
        t = panel[(panel["ticker"] == treated) & (panel["year"] == y)]
        c = panel[(panel["ticker"].isin(controls)) & (panel["year"] == y)]
        if t.empty:
            continue
        out["k"].append(k); out["year"].append(y)
        for m in MEASURES:
            out["treated"][m].append(None if pd.isna(t[m].iloc[0]) else float(t[m].iloc[0]))
            out["controls_mean"][m].append(None if c.empty or c[m].isna().all() else float(c[m].mean()))
    return out


def main() -> None:
    panel = load_10k_panel()
    treated_set = {d["ticker"] for d in LETTERS}
    cases = {}
    for d in LETTERS:
        if d["relevance"] not in ("direct", "adjacent"):
            continue
        cal = event_calendar(panel, d["ticker"], pd.Timestamp(d["date"]))
        controls = matched_controls(panel, d["ticker"], cal, treated_set)
        tab = case_table(panel, d["ticker"], cal, controls)
        tab["targeted"] = d["targeted"]
        cases[d["ticker"]] = tab
        print(f"  {d['ticker']} ({d['relevance']}): k=0 in {cal.get(0)}, controls {controls}, k observed {tab['k']}")
    counts = pd.Series([d["relevance"] for d in LETTERS]).value_counts().to_dict()
    payload = {"search_window": "2022-01-01..2026-09-11", "letters": LETTERS, "relevance_counts": counts, "cases": cases}
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"-> {OUT}  ({counts})")


if __name__ == "__main__":
    main()
