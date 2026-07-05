"""
val_02_validate.py — Evaluate pipeline scores against LLM ground-truth labels

Treats LLM labels (from val_01) as ground truth and computes precision, recall,
F1, and accuracy for each binary pipeline dimension, plus Spearman ρ for the
continuous specificity score.

Only supports --variant rule_based (validating llm_full with another LLM
judge would be circular). See variant_utils.require_variant.

Outputs:
  - data/processed/variant_rule_based/validation/validation_report__rule_based.csv  — per-dimension metrics
  - reports/validation_confusion__rule_based.png     — confusion matrices grid
  - reports/validation_summary__rule_based.txt       — human-readable summary

Usage:
    uv run python scripts/val_02_validate.py --variant rule_based
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

try:
    import pipeline_logger
    import variant_utils
except ImportError:
    from scripts import pipeline_logger
    from scripts import variant_utils


BINARY_DIMS = [
    "is_ai_related",
    "is_substantive",
    "is_promotional",
    "is_risk_related",
    "is_governance_related",
]

SPECIFICITY_BINS = [-0.001, 0.33, 0.66, 1.001]
SPECIFICITY_LABELS_MAP = {"low": 0, "medium": 1, "high": 2}


def coerce_bool(val) -> int:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)) and not pd.isna(val):
        return int(bool(val))
    if isinstance(val, str):
        return int(val.strip().lower() in ("true", "yes", "1"))
    return 0


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = float((y_true == y_pred).mean())
    return {
        "dimension": name,
        "type": "binary",
        "n": len(y_true),
        "prevalence_llm": float(y_true.mean()),
        "prevalence_pipeline": float(y_pred.mean()),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "spearman_rho": None,
        "spearman_p": None,
    }


def load_config() -> dict:
    with open("configs/config.json") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline scores against LLM ground-truth labels"
    )
    variant_utils.add_variant_arg(parser)
    args = parser.parse_args()

    config = load_config()
    variant = variant_utils.resolve_variant(args.variant, config)
    variant_utils.require_variant(variant, allowed=("rule_based",), script_name="val_02")
    output_root = config.get("variants", {}).get("output_root", "data/processed")

    validation_dir = variant_utils.variant_dir(variant, output_root=output_root) / "validation"
    labeled_path = validation_dir / f"llm_labeled_sample__{variant}.parquet"
    scored_path  = variant_utils.variant_path(variant, "ai_scored_chunks", "parquet", output_root=output_root)
    out_dir      = validation_dir
    reports_dir  = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    if not labeled_path.exists():
        print(f"Error: {labeled_path} not found. Run val_01_sample_and_label.py first.")
        return
    if not scored_path.exists():
        print(f"Error: {scored_path} not found.")
        return

    labels = pd.read_parquet(labeled_path)
    pipe_cols = ["chunk_id"] + BINARY_DIMS + ["final_specificity"]
    scored = pd.read_parquet(scored_path, columns=pipe_cols)

    # chunk_id is a content hash (sha256(chunk_text)[:16]) with no ticker/section/
    # filing component, so verbatim-repeated boilerplate (e.g. the same paragraph
    # in both Item 1A and Item 7 of one filing) shares an id across multiple rows.
    # Dedupe both sides on chunk_id before merging, otherwise one labeled chunk
    # fans out into N identical evaluation rows and silently over-weights it.
    labels = labels.drop_duplicates(subset="chunk_id")
    scored = scored.drop_duplicates(subset="chunk_id")

    df = labels.merge(scored, on="chunk_id", how="inner")
    print(f"Validation set: {len(df)} chunks (of {len(labels)} labeled)\n")

    records = []
    cm_data = {}

    # --- Binary dimensions ---
    for dim in BINARY_DIMS:
        llm_col  = f"llm_{dim}"
        pipe_col = dim
        if llm_col not in df.columns or pipe_col not in df.columns:
            print(f"  Skipping {dim}: column missing")
            continue

        y_true = df[llm_col].apply(coerce_bool).values
        y_pred = df[pipe_col].apply(coerce_bool).values

        rec = evaluate_binary(y_true, y_pred, dim)
        records.append(rec)
        cm_data[dim] = confusion_matrix(y_true, y_pred, labels=[0, 1])

        print(
            f"{dim:30s}  P={rec['precision']:.3f}  R={rec['recall']:.3f}  "
            f"F1={rec['f1']:.3f}  Acc={rec['accuracy']:.3f}  "
            f"(LLM prev={rec['prevalence_llm']:.1%}, pipe prev={rec['prevalence_pipeline']:.1%})"
        )

    # --- Specificity (continuous vs ordinal) ---
    if "llm_specificity" in df.columns and "final_specificity" in df.columns:
        llm_spec_ord = df["llm_specificity"].map(SPECIFICITY_LABELS_MAP)
        valid = llm_spec_ord.notna()
        llm_spec_ord = llm_spec_ord[valid].astype(int)
        pipe_spec = df.loc[valid, "final_specificity"].astype(float)

        rho, p_val = spearmanr(llm_spec_ord, pipe_spec)
        print(f"\n{'specificity (Spearman ρ)':30s}  ρ={rho:.3f}  p={p_val:.4f}  n={valid.sum()}")

        records.append({
            "dimension": "specificity",
            "type": "continuous",
            "n": int(valid.sum()),
            "prevalence_llm": None,
            "prevalence_pipeline": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p_val), 6),
        })

    # --- Save CSV report ---
    report_df = pd.DataFrame(records)
    report_df.to_csv(out_dir / f"validation_report__{variant}.csv", index=False)
    print(f"\nReport → {out_dir}/validation_report__{variant}.csv")

    # --- Confusion matrix plot ---
    n_dims = len(cm_data)
    if n_dims > 0:
        fig, axes = plt.subplots(1, n_dims, figsize=(n_dims * 2.8 + 0.5, 3.2))
        if n_dims == 1:
            axes = [axes]
        for ax, (dim, cm) in zip(axes, cm_data.items()):
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
                annot_kws={"size": 11},
            )
            short = dim.replace("is_", "").replace("_", " ").title()
            f1_val = next((r["f1"] for r in records if r["dimension"] == dim), None)
            title = f"{short}\nF1={f1_val:.3f}" if f1_val is not None else short
            ax.set_title(title, fontsize=9)

        fig.suptitle(
            f"Pipeline vs LLM Ground Truth — Confusion Matrices (n={len(df)})",
            fontsize=11,
        )
        fig.tight_layout()
        out_png = reports_dir / f"validation_confusion__{variant}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Confusion matrices → {out_png}")

    # --- Human-readable summary ---
    binary_rows = [r for r in records if r["type"] == "binary"]
    spec_row = next((r for r in records if r["type"] == "continuous"), None)

    summary_lines = [
        "Pipeline Validation Report",
        "=" * 50,
        f"Validation set: {len(df)} chunks",
        "",
        "Binary dimensions (LLM as ground truth):",
    ]
    for r in binary_rows:
        summary_lines.append(
            f"  {r['dimension']:30s}  F1={r['f1']:.3f}  "
            f"P={r['precision']:.3f}  R={r['recall']:.3f}  Acc={r['accuracy']:.3f}"
        )
    if spec_row:
        summary_lines += [
            "",
            f"Specificity score:  Spearman ρ = {spec_row['spearman_rho']:.3f}  "
            f"(p = {spec_row['spearman_p']:.4f})",
        ]

    summary_text = "\n".join(summary_lines)
    summary_path = reports_dir / f"validation_summary__{variant}.txt"
    summary_path.write_text(summary_text)
    print(f"Summary → {summary_path}")
    print("\n" + summary_text)

    pipeline_logger.log_event(
        pipeline_step="validation_compare",
        level="SUCCESS",
        message=f"Validation complete. {len(df)} chunks, {len(binary_rows)} binary dims evaluated.",
        details={"variant": variant},
    )


if __name__ == "__main__":
    main()
