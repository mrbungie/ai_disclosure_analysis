"""
variant_utils.py — shared helpers for the rule_based / llm_full variant selection.

No numeric prefix so it can be imported by the numbered scripts and by app.py
(numbered modules like 11_cluster_archetypes.py cannot be imported by other
scripts because their filenames start with a digit). Scripts 00-08 do not
import this module — variant selection does not apply upstream of the
combined-scoring step.

See docs/plans/variant_selection_infrastructure.md for the full design.
"""

import sys
from pathlib import Path

VALID_VARIANTS = ("rule_based", "llm_full")


def add_variant_arg(parser):
    """Add --variant {rule_based,llm_full} to an argparse parser. Default None
    means "use configs/config.json's variants.active"."""
    parser.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default=None,
        help="Which classification variant to run/read/write (default: config['variants']['active']).",
    )
    return parser


def resolve_variant(cli_value, config):
    """Resolve the active variant: explicit --variant flag > config default > error.

    Always prints `[variant] active variant: ... (source: ...)` before any
    other script output, so which variant a run used is never ambiguous.
    """
    if cli_value is not None:
        variant = cli_value
        source = "--variant flag"
    else:
        variant = config.get("variants", {}).get("active")
        source = "config default"
        if not variant:
            print("[variant] ERROR: no --variant passed and no configs/config.json variants.active set.")
            sys.exit(1)

    if variant not in VALID_VARIANTS:
        print(f"[variant] ERROR: '{variant}' is not a valid variant. Choices: {VALID_VARIANTS}")
        sys.exit(1)

    print(f"[variant] active variant: {variant} (source: {source})")
    return variant


def variant_dir(variant, output_root="data/processed"):
    """Return the Path to the variant's output directory, e.g. data/processed/variant_rule_based/."""
    return Path(output_root) / f"variant_{variant}"


def variant_path(variant, filename_stem, ext, subdir=None, output_root="data/processed"):
    """Return the Path for a variant-suffixed output file:
    data/processed/variant_{variant}/[subdir/]{filename_stem}__{variant}.{ext}
    """
    base_dir = variant_dir(variant, output_root=output_root)
    if subdir:
        base_dir = base_dir / subdir
    return base_dir / f"{filename_stem}__{variant}.{ext}"


def require_variant(variant, allowed, script_name):
    """Abort execution with an explanatory message if `variant` is not in `allowed`."""
    if variant not in allowed:
        print(
            f"[variant] ERROR: {script_name} only supports variant(s) {allowed}, got '{variant}'.\n"
            f"{script_name} solo valida la variante rule_based por diseño: usar un LLM-judge para\n"
            f"validar las mismas categorías que ya clasificó otro LLM (llm_full) es circular.\n"
            f"Pasá --variant rule_based. llm_full no tiene validación de este tipo disponible\n"
            f"por diseño, no por omisión."
        )
        sys.exit(1)
