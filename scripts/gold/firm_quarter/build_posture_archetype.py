"""
scripts/gold/firm_quarter/build_posture_archetype.py — layer 2: posture
archetype weights per (ticker, quarter).

A posture is a firm's current stance, so the default window is the trailing
12 months (ttm); an expanding window anchors firms to what they said years
earlier and dilutes the governance vertex in 2022-2023.

Reads only layer-1 covariates (build_measures.py) of one channel family and
window: the `<family>_<window>_*` columns of posture_rates and
disclosure_volume. For every closed quarter q, using only
that quarter's rows (information published up to the quarter end):

  1. Training rows = firms with at least MIN_FRAMES posture frames. The
     shrinkage prior of the seven posture rates, the intensity scale (ECDF of
     frames per 1,000 words) and the standardization are fitted on them.
  2. Archetypal Analysis, k=3. The first quarter with MIN_TRAIN_ROWS rows uses
     FurthestSum with restarts (posture_features.fit_aa); later quarters start
     from the previous quarter's archetypes (warm_start_aa).
  3. Vertices named by profile: Governance-Led Disclosers = highest
     governance_orientation; Defensive Disclosers = highest risk_orientation
     among the other two; Vocal Substantives = the rest. Governance goes
     first because governance text is also risk text (the governance vertex
     can carry as much risk as the defensive one), while the two separate
     sharply on governance.
  4. Every firm with at least one posture frame gets its weights (w_voc,
     w_gov, w_def, on the simplex) and its argmax archetype; firms without
     frames (or no documents) have null weights; rows follow the firm-quarter spine.

Output: covariates/firm_quarter/posture_archetype, one
column set per variant in VARIANTS (`<family>_<window>_w_voc`, `_w_gov`,
`_w_def`, `_archetype`), and
models/posture_archetype_weights/<family>_<window>/quarter=YYYYQn/model.pkl.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from posture_features import CLUSTER_FEATURES, INTENSITY, MIN_FRAMES, POSTURE, fit_aa  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import layers as L  # noqa: E402
from warm_start_aa import WarmStartAA  # noqa: E402

BUILDER = "scripts/gold/firm_quarter/build_posture_archetype.py"
MIN_TRAIN_ROWS = 30
VARIANTS = [("posture", "ttm")]
NAMES = ("Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers")


def shrink_params(rates: pd.DataFrame) -> dict:
    """Beta prior (method of moments) per posture rate."""
    params = {}
    for col in rates.columns:
        mean, var = float(rates[col].mean()), float(rates[col].var(ddof=1))
        if var <= 0 or not 0 < mean < 1:
            params[col] = None
            continue
        strength = max(mean * (1 - mean) / var - 1, 1e-6)
        params[col] = (mean * strength, (1 - mean) * strength)
    return params


def apply_shrink(rates: pd.DataFrame, counts: pd.Series, params: dict) -> pd.DataFrame:
    out = rates.copy()
    for col, prior in params.items():
        if prior is not None:
            out[col] = (rates[col] * counts + prior[0]) / (counts + prior[0] + prior[1])
    return out


def name_vertices(archetypes: np.ndarray) -> dict[int, str]:
    risk, gov = CLUSTER_FEATURES.index("risk_orientation"), CLUSTER_FEATURES.index("governance_orientation")
    governance = int(np.argmax(archetypes[:, gov]))
    rest = [i for i in range(3) if i != governance]
    defensive = max(rest, key=lambda i: archetypes[i, risk])
    vocal = next(i for i in range(3) if i not in (defensive, governance))
    return {vocal: NAMES[0], governance: NAMES[1], defensive: NAMES[2]}


def build_variant(family: str, window: str, warm_start: bool = True) -> pd.DataFrame:
    tag = f"{family}_{window}"
    columns = {f"{tag}_{c}": c for c in ["n_posture_frames"] + POSTURE}
    rates = (pd.read_parquet(L.gold_path("covariates", "firm_quarter", "posture_rates"),
                             columns=["ticker", "quarter", "as_of_date"] + list(columns)).rename(columns=columns))
    volume = (pd.read_parquet(L.gold_path("covariates", "firm_quarter", "disclosure_volume"),
                              columns=["ticker", "quarter", f"{tag}_frames_per_1k"])
              .rename(columns={f"{tag}_frames_per_1k": "frames_per_1k"}))
    panel = rates.merge(volume, on=["ticker", "quarter"], how="left")
    model_root = REPO_ROOT / "models" / "posture_archetype_weights" / tag

    outputs, previous = [], None   # previous quarter archetypes, original units
    for quarter, rows in panel.groupby("quarter", sort=True):
        train = rows[(rows["n_posture_frames"].fillna(0) >= MIN_FRAMES) & rows["frames_per_1k"].notna()]
        if len(train) < MIN_TRAIN_ROWS:
            continue
        params = shrink_params(train[POSTURE])
        scale = np.sort(train["frames_per_1k"].values)

        def features(r: pd.DataFrame) -> pd.DataFrame:
            f = apply_shrink(r[POSTURE], r["n_posture_frames"], params)
            f[INTENSITY] = np.searchsorted(scale, r["frames_per_1k"].fillna(0).values, side="right") / len(scale)
            return f[CLUSTER_FEATURES]

        X = features(train)
        mu, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
        Z = ((X - mu) / sd).values
        if previous is None or not warm_start:
            model, _ = fit_aa(Z, 3)
        else:
            model = WarmStartAA().fit(Z, (previous - mu.values) / sd.values)
        names = name_vertices(model.archetypes_)
        previous = model.archetypes_ * sd.values + mu.values
        column = {name: j for j, name in names.items()}

        scored = rows[rows["n_posture_frames"].fillna(0) >= 1].reset_index(drop=True)
        W = model.transform(((features(scored) - mu) / sd).values)
        out = rows[["ticker", "quarter", "as_of_date", "n_posture_frames"]].merge(pd.DataFrame({
            "ticker": scored["ticker"],
            "w_voc": W[:, column[NAMES[0]]], "w_gov": W[:, column[NAMES[1]]], "w_def": W[:, column[NAMES[2]]],
            "archetype": [names[j] for j in W.argmax(axis=1)],
        }), on="ticker", how="left")
        outputs.append(out)

        target = model_root / f"quarter={quarter}"
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump({"archetypes": previous, "mean": mu, "std": sd, "vertex_names": names, "shrink_params": params,
                     "intensity_scale": scale, "feature_names": CLUSTER_FEATURES, "min_frames": MIN_FRAMES,
                     "quarter": quarter, "family": family, "window": window, "n_train": int(len(train))},
                    target / "model.pkl")
        prof = pd.DataFrame(previous, columns=CLUSTER_FEATURES, index=[names[j] for j in range(3)])
        print(f"{quarter}: train={len(train):3d} scored={len(scored):3d} | "
              f"Gov gov={prof.loc[NAMES[1], 'governance_orientation']:.2f} risk={prof.loc[NAMES[1], 'risk_orientation']:.2f} | "
              f"Def risk={prof.loc[NAMES[2], 'risk_orientation']:.2f} gov={prof.loc[NAMES[2], 'governance_orientation']:.2f} | "
              f"counts {out['archetype'].value_counts().reindex(list(NAMES)).fillna(0).astype(int).tolist()}", flush=True)

    result = pd.concat(outputs, ignore_index=True)[["ticker", "quarter", "w_voc", "w_gov", "w_def", "archetype"]]
    return result.rename(columns={c: f"{tag}_{c}" for c in ["w_voc", "w_gov", "w_def", "archetype"]})


def main(warm_start: bool = True) -> None:
    result = L.read_gold("firm_quarter")
    for family, window in VARIANTS:
        result = result.merge(build_variant(family, window, warm_start), on=["ticker", "quarter"], how="left",
                              validate="one_to_one")
    L.write_gold("covariates", "firm_quarter", "posture_archetype", result, builder=BUILDER,
                 extra={"columns": "<family>_<window>_<metric>", "variants": [f"{f}_{w}" for f, w in VARIANTS],
                        "min_frames": MIN_FRAMES, "warm_start": warm_start,
                        "usable_from": "as_of_date = quarter end + 1 day"})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-warm-start", action="store_true")
    args = parser.parse_args()
    main(not args.no_warm_start)
