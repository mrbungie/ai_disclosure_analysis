"""
scripts/gold/posture/build_posture_asof.py — point-in-time posture archetypes.

One snapshot per quarter start (scripts/common/pit.py). A snapshot dated
`as_of_date` q uses only AI frames and documents published before q:

  1. Posture of each firm at q = its seven posture rates over the frames of
     the posture channels (posture_features.POSTURE_FORMS: 10-K, 8-K, DEF 14A)
     published in the WINDOW_MONTHS before q, plus disclosure intensity
     (frames per 1,000 words over the documents of the same window).
  2. Training rows = the 1 January postures (calendar years, TRAIN_MONTHS) up
     to q with at least MIN_FRAMES frames: archetypes built on year c are
     available from the 1 January c+1 snapshot. The shrinkage prior, the
     standardization and the intensity scale are estimated on those rows only.
  3. Archetypal Analysis (k=3): the first snapshot uses FurthestSum with
     restarts (posture_features.fit_aa); every later snapshot starts from the
     previous snapshot's archetypes (warm_start_aa), so archetypes evolve
     with the data instead of jumping between near-equivalent optima.
  4. Vertices are named by their profile: Defensive Disclosers = highest
     risk_orientation, Governance-Led Disclosers = highest
     governance_orientation among the other two, Vocal Substantives = the
     remaining one.
  5. Every universe firm gets its archetype weights at q with the same
     construction; firms without frames in the window are "No AI".

Output: data/gold/predictions/firm_quarter/posture_archetype_asof.parquet
(ticker, as_of_date, n_frames_window, w_voc, w_gov, w_def, archetype) and one
model bundle per snapshot under models/posture_archetype_asof/. Join onto events
with pit.asof_join(event_date=...).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
from posture_features import (  # noqa: E402  (pins BLAS threads before numpy import)
    CLUSTER_FEATURES, INTENSITY, MIN_FRAMES, POSTURE, POSTURE_FORMS, fit_aa, frame_indicators, load_frames, write_gold,
)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import layers as L  # noqa: E402
from pit import snapshot_dates  # noqa: E402
from warm_start_aa import WarmStartAA  # noqa: E402

BUILDER = "scripts/gold/posture/build_posture_asof.py"
WINDOW_MONTHS = 12
MIN_TRAIN_ROWS = 30
# Training rows are the 1 January windows (= calendar years, no overlap). The
# archetypes fitted on year c exist from the 1 January c+1 snapshot onward;
# in-between snapshots reuse them and only refresh each firm's 12-month posture.
TRAIN_MONTHS = (1,)
NAMES = ("Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers")
OUT = L.gold_path("predictions", "firm_quarter", "posture_archetype_asof")
MODEL_DIR = REPO_ROOT / "models" / "posture_archetype_asof"


def shrink_params(rates: pd.DataFrame) -> dict:
    """Beta prior (method of moments) per posture rate, fitted on training rows."""
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


def snapshot_posture(indicators: pd.DataFrame, documents: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Raw posture rates, frame counts and intensity per firm over the window before `as_of`."""
    start = as_of - pd.DateOffset(months=WINDOW_MONTHS)
    window = indicators[(indicators["available_date"] < as_of) & (indicators["available_date"] >= start)]
    rates = window.groupby("ticker")[POSTURE].mean()
    rates["n_frames_window"] = window.groupby("ticker").size()
    docs = documents[(documents["fecha"] < as_of) & (documents["fecha"] >= start)]
    words = docs.groupby("ticker").agg(w=("n_words", "sum"), fr=("n_frames", "sum"))
    rates["frames_per_1k"] = (1000 * words["fr"] / words["w"].replace(0, np.nan)).reindex(rates.index)
    rates["as_of_date"] = as_of
    return rates.reset_index()


def name_vertices(archetypes_std: np.ndarray) -> dict[int, str]:
    risk, gov = CLUSTER_FEATURES.index("risk_orientation"), CLUSTER_FEATURES.index("governance_orientation")
    defensive = int(np.argmax(archetypes_std[:, risk]))
    rest = [i for i in range(3) if i != defensive]
    governance = max(rest, key=lambda i: archetypes_std[i, gov])
    vocal = next(i for i in range(3) if i not in (defensive, governance))
    return {vocal: NAMES[0], governance: NAMES[1], defensive: NAMES[2]}


def main(forms: tuple[str, ...] = POSTURE_FORMS, warm_start: bool = True, train_months: tuple[int, ...] = TRAIN_MONTHS,
         out_path: Path = OUT, model_dir: Path = MODEL_DIR) -> None:
    frames = load_frames(forms)
    indicators = frame_indicators(frames)
    documents = pd.read_parquet(L.gold_path("covariates", "document", "document_panel"),
                                columns=["ticker", "fecha", "n_words", "n_frames"])
    universe = L.read("silver.firm_universe").select("ticker").to_pandas()["ticker"]
    dates = snapshot_dates("2021-01-01", indicators["available_date"].max() + pd.offsets.QuarterBegin(startingMonth=1))

    history: list[pd.DataFrame] = []
    outputs: list[pd.DataFrame] = []
    previous = None   # previous snapshot archetypes, original units
    for as_of in dates:
        current = snapshot_posture(indicators, documents, as_of)
        if as_of.month in train_months:
            history.append(current)
        if not history:
            continue
        train = pd.concat(history, ignore_index=True)
        train = train[train["n_frames_window"] >= MIN_FRAMES].dropna(subset=["frames_per_1k"])
        if len(train) < MIN_TRAIN_ROWS:
            continue
        params = shrink_params(train[POSTURE])
        scale = np.sort(train["frames_per_1k"].values)

        def features(rows: pd.DataFrame) -> pd.DataFrame:
            f = apply_shrink(rows[POSTURE], rows["n_frames_window"], params)
            f[INTENSITY] = np.searchsorted(scale, rows["frames_per_1k"].fillna(0).values, side="right") / len(scale)
            return f[CLUSTER_FEATURES]

        X_train = features(train)
        mu, sd = X_train.mean(), X_train.std(ddof=0).replace(0, 1.0)
        Z = ((X_train - mu) / sd).values
        if previous is None or not warm_start:
            model, _ = fit_aa(Z, 3)
            archetypes = model.archetypes_
        else:
            model = WarmStartAA().fit(Z, (previous - mu.values) / sd.values)
            archetypes = model.archetypes_
        names = name_vertices(archetypes)
        previous = archetypes * sd.values + mu.values

        scored = current[current["n_frames_window"] >= 1].reset_index(drop=True)
        weights = model.transform(((features(scored) - mu) / sd).values)
        column = {name: j for j, name in names.items()}
        out = pd.DataFrame({"ticker": universe, "as_of_date": as_of})
        scored_out = pd.DataFrame({
            "ticker": scored["ticker"], "n_frames_window": scored["n_frames_window"].astype(int),
            "w_voc": weights[:, column[NAMES[0]]], "w_gov": weights[:, column[NAMES[1]]],
            "w_def": weights[:, column[NAMES[2]]],
            "archetype": [names[j] for j in weights.argmax(axis=1)],
        })
        out = out.merge(scored_out, on="ticker", how="left")
        out["n_frames_window"] = out["n_frames_window"].fillna(0).astype(int)
        out["archetype"] = out["archetype"].fillna("No AI")
        outputs.append(out)

        target = model_dir / f"as_of={as_of.date()}"
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump({"archetypes": previous, "mean": mu, "std": sd, "vertex_names": names,
                     "shrink_params": params, "intensity_scale": scale, "feature_names": CLUSTER_FEATURES,
                     "window_months": WINDOW_MONTHS, "min_frames": MIN_FRAMES, "as_of_date": str(as_of.date()),
                     "warm_start": previous is not None, "n_train": int(len(train))},
                    target / "model.pkl")
        prof = pd.DataFrame(previous, columns=CLUSTER_FEATURES, index=[names[j] for j in range(3)])
        print(f"{as_of.date()}: train={len(train):5d} scored={len(scored):3d} | "
              f"Gov gov={prof.loc[NAMES[1], 'governance_orientation']:.2f} risk={prof.loc[NAMES[1], 'risk_orientation']:.2f} | "
              f"Def risk={prof.loc[NAMES[2], 'risk_orientation']:.2f} gov={prof.loc[NAMES[2], 'governance_orientation']:.2f} | "
              f"Voc spec={prof.loc[NAMES[0], 'specificity']:.2f} | "
              f"counts {out['archetype'].value_counts().reindex(list(NAMES) + ['No AI']).fillna(0).astype(int).tolist()}",
              flush=True)

    result = pd.concat(outputs, ignore_index=True)
    result.insert(0, "id", result["ticker"] + "_" + result["as_of_date"].dt.strftime("%Y-%m-%d"))
    write_gold(out_path, result, keys=["ticker", "as_of_date"], inputs=[out_path], builder=BUILDER,
               extra={"grain": "firm_quarter (ticker x as_of_date)", "window_months": WINDOW_MONTHS,
                      "min_frames": MIN_FRAMES, "forms": list(forms), "warm_start": warm_start,
                      "train_months": list(train_months)})
    print(f"-> {out_path} ({len(result):,} rows, {result['as_of_date'].nunique()} snapshots)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--forms", nargs="+", default=list(POSTURE_FORMS), help="frame channels")
    parser.add_argument("--train-months", type=int, nargs="+", default=list(TRAIN_MONTHS),
                        help="snapshot months whose windows enter training (1 = non-overlapping calendar years)")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(tuple(args.forms), not args.no_warm_start, tuple(args.train_months), args.out, args.model_dir)
