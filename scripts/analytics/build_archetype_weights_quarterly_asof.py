"""Rebuilds expanding archetype weights at QUARTERLY cutoffs (not annual),
so each call can be matched to the most recent cutoff strictly before its
own call date -- true "as of" weights, no leakage, without discarding
close-to-a-year of legitimate recent information the way a full calendar-
year lag would.
"""
from pathlib import Path
import sys
import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/analytics"))
from archetypes import AA as _AA_exp

CLUSTERS = REPO_ROOT / "data/processed/clusters"
DB_PATH = REPO_ROOT / "duckdb/thesis.duckdb"
OUT = CLUSTERS / "panel_expanding_archetype_weights_quarterly.parquet"

conn = duckdb.connect(str(DB_PATH), read_only=True)
conn.execute(f'SET file_search_path = "{REPO_ROOT}";')

_POSTURE_EXP = ["promotional_posture", "hedging_posture", "risk_orientation", "governance_orientation",
                "temporal_posture", "ai_positioning", "specificity"]
_CROSS_FEATS_EXP = _POSTURE_EXP + ["disclosure_intensity"]
_exp_names = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers"]

_frames_exp = conn.execute("""
    WITH frame_domains AS (
        SELECT text_hash, frame_id, bool_or(domain = 'customer_facing') AS is_customer_facing
        FROM gold_ai_activities WHERE has_activity GROUP BY text_hash, frame_id
    )
    SELECT fm.ticker, fm.filing_date,
           f.concepts, f.temporal, f.specificity, f.rhetoric,
           COALESCE(fd.is_customer_facing, false) AS is_customer_facing
    FROM gold_ai_frames f
    LEFT JOIN frame_domains fd ON fd.text_hash = f.text_hash AND fd.frame_id = f.frame_id
    JOIN filing_manifest fm USING (country_code, accession_number)
    WHERE f.country_code = 'us' AND f.has_frame AND fm.ticker IS NOT NULL
""").fetchdf()
conn.close()

_frames_exp["filing_date"] = pd.to_datetime(_frames_exp["filing_date"])
_frames_exp["quarter"] = _frames_exp["filing_date"].dt.to_period("Q")

def _posture_frame_rates(df):
    df = df.copy()
    concepts = df["concepts"].apply(lambda c: set(c) if c is not None else set())
    df["promotional_posture"] = df["rhetoric"].apply(
        lambda r: np.mean([x in list(r) if r is not None else False for x in ("promotional", "strategic")]))
    df["hedging_posture"] = df["rhetoric"].apply(lambda r: float("hedged" in list(r)) if r is not None else 0.0)
    df["risk_orientation"] = concepts.apply(lambda s: float(any(str(c).startswith("risk_") for c in s)))
    df["governance_orientation"] = concepts.apply(lambda s: float(any(str(c).startswith("gov_") for c in s)))
    df["temporal_posture"] = (df["temporal"] == "realized").astype(float)
    df["ai_positioning"] = df["is_customer_facing"].astype(float)
    df["specificity"] = df["specificity"].apply(lambda s: len(s) / 5.0 if s is not None else 0.0)
    return df

_df_exp = _posture_frame_rates(_frames_exp)
_dp_exp = pd.read_parquet(CLUSTERS / "document_panel.parquet")
_dp_exp["quarter"] = pd.to_datetime(_dp_exp["fecha"]).dt.to_period("Q")

_quarters = sorted(_df_exp["quarter"].unique())
print(f"{len(_quarters)} quarterly cutoffs: {_quarters[0]} .. {_quarters[-1]}")

_exp_rows = []
for _qi, _cutoff in enumerate(_quarters):
    if _qi < 3:
        continue  # need enough history to fit a stable geometry
    _sub = _df_exp[_df_exp["quarter"] <= _cutoff]
    _counts = _sub.groupby("ticker").size()
    _active_tickers = _counts[_counts >= 5].index
    if len(_active_tickers) < 20:
        continue
    _rates = _sub.groupby("ticker")[_POSTURE_EXP].mean().loc[_active_tickers]
    # shrink using expanding().shift(1)-style: use counts as of this cutoff (already excludes anything after cutoff)
    def _shrink_to_prior_exp(rates, counts):
        out = {}
        for col in rates.columns:
            p = rates[col]
            mean, var = float(p.mean()), float(p.var(ddof=1))
            if var <= 0 or not 0 < mean < 1:
                out[col] = p
                continue
            strength = max(mean * (1 - mean) / var - 1, 1e-6)
            alpha, beta = mean * strength, (1 - mean) * strength
            out[col] = (p * counts + alpha) / (counts + alpha + beta)
        return pd.DataFrame(out, index=rates.index)
    _shrunk = _shrink_to_prior_exp(_rates, _counts.loc[_active_tickers])
    _words = _dp_exp[_dp_exp["quarter"] <= _cutoff].groupby("ticker").agg(
        w=("n_words", "sum"), fr=("n_frames", "sum")).reindex(_active_tickers)
    _fr1k = 1000 * _words["fr"] / _words["w"].replace(0, np.nan)
    _shrunk["disclosure_intensity"] = _fr1k.rank(pct=True)
    _train_df = _shrunk.dropna()
    if len(_train_df) < 20:
        continue
    _mu_e, _sd_e = _train_df.mean(), _train_df.std(ddof=0)
    _sd_e = _sd_e.replace(0, 1.0)
    _X_train = ((_train_df - _mu_e) / _sd_e).values
    _aa_e = _AA_exp(n_archetypes=3, random_state=42, max_iter=500)
    _aa_e.fit_transform(_X_train)
    _A_e = _aa_e.archetypes_
    _def_col = int(np.argmax(_A_e[:, _POSTURE_EXP.index("risk_orientation")]))
    _rem = [i for i in range(3) if i != _def_col]
    _gov_col = max(_rem, key=lambda i: _A_e[i, _POSTURE_EXP.index("governance_orientation")])
    _voc_col = [i for i in range(3) if i not in (_def_col, _gov_col)][0]
    _order_e = [_voc_col, _gov_col, _def_col]

    _X_cross = ((_train_df.loc[_active_tickers.intersection(_train_df.index)] - _mu_e) / _sd_e).values
    _tickers_cross = _active_tickers.intersection(_train_df.index)
    _W_cross = _aa_e.transform(_X_cross)
    _W_ordered = _W_cross[:, _order_e]
    _exp_rows.append(pd.DataFrame({
        "ticker": _tickers_cross, "cutoff_quarter": _cutoff,
        "w_voc": _W_ordered[:, 0], "w_gov": _W_ordered[:, 1], "w_def": _W_ordered[:, 2],
    }))
    if _qi % 4 == 0:
        print(f"  {_cutoff}: {len(_tickers_cross)} firms")

_panel_q = pd.concat(_exp_rows, ignore_index=True)
_panel_q.to_parquet(OUT, index=False)
print(f"\nSaved {len(_panel_q)} ticker-quarter soft weights to {OUT}")
