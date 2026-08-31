"""
agreement_check.py — Human audit of the reward labels.

The eval set's labels are the reward function the harness search optimizes
against, so their validity needs a human anchor: this exports a balanced
subsample of labeled rows to an Excel workbook for hand-labeling (the stored
labels live on a separate sheet so they can't anchor the rater), then scores
raw agreement and Cohen's kappa once filled in. The chain the thesis reports:
human <-> reward labels (kappa, here) -> harness <-> reward labels (test F1,
eval_harness) -> corpus measurement (apply_harness, deterministic).

Usage:
    uv run python scripts/agreement_check.py --cycle eval --make [--n 60] [--seed 42]
    uv run python scripts/agreement_check.py --cycle eval --score
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import harness_fit
except ImportError:
    from scripts import harness_fit


@dataclass
class CycleSpec:
    labeled_path: Path
    id_col: str
    text_col: str
    label_fields: list[str]
    workbook: Path
    report: Path


CYCLES = {
    "eval": CycleSpec(
        labeled_path=Path("data/interim/eval/eval_set.parquet"),
        id_col="paragraph_id",
        text_col="paragraph_text",
        label_fields=["is_ai_related", "is_substantive", "is_promotional", "is_risk_related",
                      "is_governance_related", "is_use_case_specific", "is_quantified"],
        workbook=Path("reports/agreement_eval.xlsx"),
        report=Path("reports/agreement_eval_report.txt"),
    ),
}


def auto_make(cycle: str, n: int = 60, seed: int = 42) -> None:
    """Called by build_eval_set right after a labeling run:
    export the human validation workbook automatically if it doesn't exist
    yet, so the to-be-validated sample is always generated without a separate
    manual step. Never overwrites an existing workbook (it may hold hand
    labels)."""
    spec = CYCLES[cycle]
    if spec.workbook.exists():
        print(f"Human validation workbook already exists (not overwritten): {spec.workbook}")
        return
    labeled = pd.read_parquet(spec.labeled_path)
    harness_fit.export_agreement_workbook(
        labeled, spec.id_col, spec.text_col, spec.label_fields,
        spec.workbook, n, seed,
    )
    print(f"ACTION NEEDED: hand-label the 'label_me' sheet in {spec.workbook} "
          f"(instructions inside). The fit script will pick it up automatically.")


def report_lines(cycle: str) -> list[str]:
    """Judge-validation lines for the fit scripts' final reports: Cohen's
    kappa per label if the workbook is filled in, an explicit UNVALIDATED
    warning otherwise. Also refreshes the standalone agreement report."""
    spec = CYCLES[cycle]
    if not spec.workbook.exists():
        return [f"JUDGE VALIDATION: no workbook at {spec.workbook} — the judge is UNVALIDATED "
                f"(it is generated automatically at the end of the labeling step)."]
    lines = harness_fit.score_agreement(spec.workbook, spec.id_col, spec.label_fields)
    spec.report.parent.mkdir(parents=True, exist_ok=True)
    spec.report.write_text("\n".join(lines) + "\n")
    if any("no human labels filled in yet" in line for line in lines):
        lines.append(f"JUDGE VALIDATION incomplete: fill in {spec.workbook} — "
                     f"until then the judge is UNVALIDATED for the fields above.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", choices=sorted(CYCLES), required=True)
    parser.add_argument("--make", action="store_true", help="Export the workbook to hand-label")
    parser.add_argument("--score", action="store_true", help="Score a filled-in workbook")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.make and not args.score:
        parser.error("Pass --make and/or --score")
    spec = CYCLES[args.cycle]

    if args.make:
        if not spec.labeled_path.exists():
            print(f"Error: {spec.labeled_path} not found — run the cycle's labeling step first.")
            return
        if spec.workbook.exists():
            print(f"Error: {spec.workbook} already exists. Refusing to overwrite a workbook "
                  f"that may hold hand labels — delete it yourself if you want a fresh one.")
            return
        labeled = pd.read_parquet(spec.labeled_path)
        harness_fit.export_agreement_workbook(
            labeled, spec.id_col, spec.text_col, spec.label_fields,
            spec.workbook, args.n, args.seed,
        )
        print("Fill in the your_* columns on the 'label_me' sheet, then run with --score.")

    if args.score:
        if not spec.workbook.exists():
            print(f"Error: {spec.workbook} not found — run with --make first.")
            return
        lines = harness_fit.score_agreement(spec.workbook, spec.id_col, spec.label_fields)
        spec.report.parent.mkdir(parents=True, exist_ok=True)
        spec.report.write_text("\n".join(lines) + "\n")
        print(f"Agreement report -> {spec.report}")


if __name__ == "__main__":
    main()
