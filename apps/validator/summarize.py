"""Summarizes atomic annotations exported from the validation UI (`apps/validator/annotations/*.json`).

Calculates validation metrics by scope and category:
  1. Prefilter:
     - Precision, recall, and confusion matrix (raw and stratum-reweighted).
  2. Frames:
     - Real existence rate of extracted AI frames (no hallucination).
     - Accuracy of temporal, promotional, and specificity tags.
     - Validity of attributed textual evidence.
     - Omission rate in negative paragraphs.
  3. Activities:
     - Accuracy of core claim (action + object).
     - Accuracy of technology source (own vs third-party / vendor).
     - Accuracy of target beneficiaries, deployment stages, and named entities.
     - Direct textual evidence validity.
     - Omission rate of activities per paragraph.

Usage:
    uv run --frozen --no-sync python apps/validator/summarize.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA = HERE / "data.json"
ANN_DIR = HERE / "annotations"


def kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load_annotations() -> dict:
    merged: dict = {}
    for path in sorted(ANN_DIR.glob("*.json")) if ANN_DIR.exists() else []:
        obj = json.loads(path.read_text(encoding="utf-8"))
        items = obj.get("annotations") or obj
        for k, v in items.items():
            if not isinstance(v, dict):
                v = {"verdict": v, "ts": None}
            if k not in merged or (v.get("ts") or "") > (merged[k].get("ts") or ""):
                merged[k] = v
    return merged


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"{DATA} does not exist. Run build_sample.py first.")
    data = json.loads(DATA.read_text(encoding="utf-8"))
    ann = load_annotations()
    if not ann:
        print(f"Notice: No annotations found in {ANN_DIR}/. Create a JSON file or export from the UI.")
        return

    items_by_ambito = data.get("items", {})
    total_ann = len(ann)
    print(f"\n=======================================================")
    print(f" ATOMIC VALIDATION SUMMARY (n = {total_ann} responses)")
    print(f"=======================================================\n")

    # ---- 1. PREFILTER ----
    pf_items = items_by_ambito.get("prefilter", [])
    pf_ann = [it for it in pf_items if it["id"] in ann and ann[it["id"]].get("verdict")]
    if pf_ann:
        print(f"--- 1. PREFILTER (evaluated n = {len(pf_ann)} / {len(pf_items)}) ---")
        mentions = [it for it in pf_ann if it.get("category") == "mentions_ai"]

        raw = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "unc": 0}
        cells = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

        pred_run = data.get("predictions_run")
        pop_n = {}
        sample_n = Counter((it["form"], it["estrato"]) for it in mentions)
        if pred_run and pred_run != "existing":
            run_path = REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique" / pred_run
            if run_path.exists():
                pop = (
                    pl.scan_parquet(run_path)
                    .filter(pl.col("country_code") == "us")
                    .with_columns(
                        pl.when(pl.col("is_ai_prefiltered"))
                        .then(pl.lit("positive"))
                        .when(pl.col("predicted_proba") >= 0.05)
                        .then(pl.lit("gray_zone"))
                        .otherwise(pl.lit("negative"))
                        .alias("estrato")
                    )
                    .group_by("form", "estrato")
                    .len()
                    .collect()
                )
                pop_n = {(r["form"], r["estrato"]): r["len"] for r in pop.iter_rows(named=True)}

        for it in mentions:
            v = ann[it["id"]]["verdict"]
            if v == "U":
                raw["unc"] += 1
                continue
            human_says_ai = v == "Y"
            model_says_ai = bool(it.get("model_value"))

            if model_says_ai:
                k = "tp" if human_says_ai else "fp"
            else:
                k = "fn" if human_says_ai else "tn"
            raw[k] += 1

            if pop_n:
                strat_key = (it.get("form"), it.get("estrato"))
                w = pop_n.get(strat_key, 0) / max(1, sample_n[strat_key])
                cells[k] += w

        n_valid = raw["tp"] + raw["fp"] + raw["fn"] + raw["tn"]
        if n_valid > 0:
            prec_raw = raw["tp"] / max(1, raw["tp"] + raw["fp"])
            rec_raw = raw["tp"] / max(1, raw["tp"] + raw["fn"])
            acc_raw = (raw["tp"] + raw["tn"]) / n_valid
            print(f"  AI Mention (n={n_valid}, uncertain={raw['unc']}):")
            print(f"    - Raw Accuracy:   {100*acc_raw:.1f}%")
            print(f"    - Raw Precision:  {100*prec_raw:.1f}%  (TP={raw['tp']}, FP={raw['fp']})")
            print(f"    - Raw Recall:     {100*rec_raw:.1f}%   (FN={raw['fn']}, TN={raw['tn']})")

            if pop_n and (cells["tp"] + cells["fp"]) > 0:
                prec_w = cells["tp"] / max(1e-9, cells["tp"] + cells["fp"])
                rec_w = cells["tp"] / max(1e-9, cells["tp"] + cells["fn"])
                print(f"    - Reweighted:     Precision {100*prec_w:.1f}% | Recall {100*rec_w:.1f}%")
        print()

    # ---- 2. FRAMES ----
    fr_items = items_by_ambito.get("frames", [])
    fr_ann = [it for it in fr_items if it["id"] in ann and ann[it["id"]].get("verdict")]
    if fr_ann:
        print(f"--- 2. FRAMES (evaluated n = {len(fr_ann)} / {len(fr_items)}) ---")
        by_cat = defaultdict(list)
        for it in fr_ann:
            by_cat[it.get("category", "other")].append(it)

        for cat, label in [
            ("existence", "Frame existence (factual, not hallucinated)"),
            ("temporal", "Temporal classification accuracy"),
            ("promotional", "Promotional rhetoric agreement"),
            ("specificity", "Specificities (product, process, vendor, etc.)"),
            ("evidence", "Textual evidence validity"),
            ("missing_frame", "Omission of frames in negative paragraphs"),
        ]:
            c_items = by_cat.get(cat, [])
            if not c_items:
                continue
            y = sum(1 for it in c_items if ann[it["id"]]["verdict"] == "Y")
            n = sum(1 for it in c_items if ann[it["id"]]["verdict"] == "N")
            u = sum(1 for it in c_items if ann[it["id"]]["verdict"] == "U")
            tot = y + n + u
            rate = 100 * y / max(1, tot)
            print(f"  {label:48s}: {y:4d} Yes ({rate:5.1f}%) | {n:3d} No | {u:2d} Uncertain (n={tot})")
        print()

    # ---- 3. ACTIVITIES ----
    act_items = items_by_ambito.get("activities", [])
    act_ann = [it for it in act_items if it["id"] in ann and ann[it["id"]].get("verdict")]
    if act_ann:
        print(f"--- 3. ACTIVITIES (evaluated n = {len(act_ann)} / {len(act_items)}) ---")
        by_cat = defaultdict(list)
        for it in act_ann:
            by_cat[it.get("category", "other")].append(it)

        for cat, label in [
            ("existence", "Core claim (action + object stated)"),
            ("source", "Technology source (own vs third-party)"),
            ("action_object", "Action & object accuracy"),
            ("target_stage", "Target beneficiary & deployment stage"),
            ("entities", "Named entities & assigned roles"),
            ("evidence", "Direct textual evidence validity"),
            ("completeness", "Activity omissions in paragraph"),
        ]:
            c_items = by_cat.get(cat, [])
            if not c_items:
                continue
            y = sum(1 for it in c_items if ann[it["id"]]["verdict"] == "Y")
            n = sum(1 for it in c_items if ann[it["id"]]["verdict"] == "N")
            u = sum(1 for it in c_items if ann[it["id"]]["verdict"] == "U")
            tot = y + n + u
            rate = 100 * y / max(1, tot)
            print(f"  {label:48s}: {y:4d} Yes ({rate:5.1f}%) | {n:3d} No | {u:2d} Uncertain (n={tot})")
        print()


if __name__ == "__main__":
    main()
