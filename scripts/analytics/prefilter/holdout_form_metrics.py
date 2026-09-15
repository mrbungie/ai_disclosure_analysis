"""
scripts/analytics/prefilter/holdout_form_metrics.py — holdout evaluation and
decision curve for a deployed prefilter model, moved out of
`scripts/enrichment/ai_prefilter_deploy.py` (E-M3): fitting a model is
enrichment's job, reporting how well it does on held-out data is analytics'.

Loads an ALREADY-TRAINED model (`models/ai_classification/run=<run_id>/
model.joblib`, read-only — never refit here) and scores it against the
1,500-label DEF 14A / 8-K validation sample that never enters training
(`data/interim/golden_set_forms/`, read-only, the "." subdir — see
`ai_prefilter_deploy.load_form_labels`). Bronze features for those labels
come from `bronze.prefilter_scores`/`bronze.unique_paragraphs`/
`bronze.prefilter_sentence_scores` via the same feature-building code
`ai_prefilter_deploy.py` uses to fit, so a holdout row gets the exact
features the deployed model expects.

Output:
    data/results/prefilter/holdout_form_metrics.json  total + per-form F1/precision/recall/AP
    data/results/prefilter/decision_curve.csv          threshold -> recall/precision/F-beta off the deployed cut

Usage:
    uv run python scripts/analytics/prefilter/holdout_form_metrics.py
    uv run python scripts/analytics/prefilter/holdout_form_metrics.py --run-id 20260906T160624Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "enrichment"))
import layers as L  # noqa: E402
import ai_prefilter_deploy as deploy  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "ai_classification"
DECISION_CUTS = (0.5, 0.36, 0.25, 0.2, 0.15, 0.12, 0.1, 0.07, 0.05)


def latest_run_id() -> str:
    runs = sorted(p.name.removeprefix("run=") for p in MODEL_DIR.glob("run=*") if (p / "model.joblib").exists())
    if not runs:
        raise FileNotFoundError(f"no trained model under {MODEL_DIR}")
    return runs[-1]


def load_model(run_id: str) -> dict:
    payload = joblib.load(MODEL_DIR / f"run={run_id}" / "model.joblib")
    return {"model": payload["model"], "columns": payload["columns"], "threshold": payload["threshold"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None, help="models/ai_classification/run=<id> to evaluate "
                                                        "(default: newest with a model.joblib).")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Same F-beta weighting ai_prefilter_deploy.py uses to pick a threshold.")
    args = parser.parse_args()

    run_id = args.run_id or latest_run_id()
    deployed = load_model(run_id)
    print(f"Modelo: run={run_id} ({len(deployed['columns'])} señales, umbral {deployed['threshold']:.2f})")

    holdout = deploy.load_form_labels(".", "form_validation")
    if holdout.empty:
        print("No hay muestra de validación en data/interim/golden_set_forms/ — nada que evaluar.")
        return

    Xh = holdout[deployed["columns"]].astype(float).to_numpy()
    yh = holdout["y"].astype(bool).to_numpy()
    wh = holdout["inclusion_weight"].astype(float).to_numpy()
    proba_h = deployed["model"].predict_proba(Xh)[:, 1]
    threshold = deployed["threshold"]

    holdout_metrics = {"total": deploy.weighted_f1(yh, proba_h >= threshold, wh, args.beta)}
    holdout_metrics["total"]["ap"] = float(average_precision_score(yh, proba_h, sample_weight=wh))
    for form in sorted(holdout["form"].unique()):
        mask = (holdout["form"] == form).to_numpy()
        holdout_metrics[form] = deploy.weighted_f1(yh[mask], (proba_h >= threshold)[mask], wh[mask], args.beta)

    print(f"\nHOLDOUT ({len(holdout):,} etiquetas de DEF 14A / 8-K que nunca entraron al ajuste):")
    for name, values in holdout_metrics.items():
        print(f"  {name:10s} F1 {values['f1']:.3f} | prec {values['precision']:.3f} | "
              f"recall {values['recall']:.3f}" + (f" | AP {values['ap']:.3f}" if "ap" in values else ""))

    curve_rows = []
    for cut in DECISION_CUTS:
        point = deploy.weighted_f1(yh, proba_h >= cut, wh, 2.0)
        curve_rows.append({"threshold": cut, "recall": point["recall"], "precision": point["precision"],
                           "f2": point["fbeta"], "deployed": abs(cut - threshold) < 0.005})

    out_json = L.results_path("prefilter", "holdout_form_metrics.json")
    out_csv = L.results_path("prefilter", "decision_curve.csv")
    out_json.write_text(json.dumps({
        "run_id": run_id, "threshold": threshold, "beta": args.beta,
        "n_holdout": int(len(holdout)), "metrics": holdout_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, sort_keys=True) + "\n")
    with out_csv.open("w") as f:
        f.write("threshold,recall,precision,f2,deployed\n")
        for row in curve_rows:
            f.write(f"{row['threshold']},{row['recall']},{row['precision']},{row['f2']},{row['deployed']}\n")

    print(f"\n-> {out_json}\n-> {out_csv}")


if __name__ == "__main__":
    main()
