"""Acuerdo entre jueces sobre las mismas filas del golden set.

El golden set terminó etiquetado por dos modelos —gemini-3.8-flash y
qwen3.7-flash— repartidos por estrato, y sin una sola fila en común: el acuerdo
entre ellos nunca se pudo medir, aunque el prefiltro se ajustó con la unión de
ambos. Al re-etiquetar todo con un solo juez (`golden_set.py label
--coverage-judge current`) las filas del juez viejo quedan en disco, así que
cada párrafo re-etiquetado pasa a tener DOS lecturas independientes del mismo
texto — y eso sí es medible.

Reporta, sobre las filas con dos etiquetas:
  - acuerdo bruto y kappa de Cohen para `is_ai_mention` (relevance != 'none'),
    que es el target del prefiltro, y para `is_ai_disclosure` (sustantivo);
  - la matriz de confusión de `relevance` (none/incidental/substantive), que
    muestra DÓNDE discrepan: el borde caro es none vs. incidental;
  - el desglose por tier léxico, para separar "discrepan en lo difícil" de
    "discrepan en todo".

Es la cifra que faltaba para poder decir algo sobre la confiabilidad de la
etiqueta, y la referencia contra la que se comparará la anotación humana
cuando exista.

Uso:
    uv run python scripts/verif/judge_agreement.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "data" / "interim" / "golden_set"
KEY = ["country_code", "form", "accession_number", "item_key", "paragraph_index"]
RELEVANCE_ORDER = ["none", "incidental", "substantive"]


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Kappa de Cohen: acuerdo por sobre el esperado por azar. Con etiquetas
    muy desbalanceadas el acuerdo bruto engaña — dos jueces que dicen 'no' el
    95% de las veces coinciden el 90% sin saber nada."""
    categories = sorted(set(a) | set(b))
    confusion = pd.crosstab(pd.Categorical(a, categories), pd.Categorical(b, categories),
                            dropna=False).to_numpy(float)
    total = confusion.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(confusion) / total
    expected = float((confusion.sum(0) * confusion.sum(1)).sum()) / total ** 2
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--judge-a", default="gemini-3.8-flash")
    parser.add_argument("--judge-b", default="qwen/qwen3.7-flash")
    args = parser.parse_args()

    files = sorted(glob.glob(str(args.golden_dir / "golden_set_labels__session=*.parquet")))
    labels = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    labels = labels[labels["error"].isna() & labels["judge_model"].notna()]
    labels["is_ai_mention"] = labels["relevance"] != "none"
    print(f"{len(labels):,} etiquetas sin error | "
          f"{labels['judge_model'].value_counts().to_dict()}")

    a = labels[labels["judge_model"] == args.judge_a].drop_duplicates(KEY)
    b = labels[labels["judge_model"] == args.judge_b].drop_duplicates(KEY)
    paired = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    print(f"\nfilas con las dos lecturas: {len(paired):,} "
          f"(de {len(a):,} de {args.judge_a} y {len(b):,} de {args.judge_b})")
    if paired.empty:
        print("Nada que comparar todavía.")
        return

    report = {"judge_a": args.judge_a, "judge_b": args.judge_b, "n_paired": int(len(paired))}
    print("\n--- acuerdo ---")
    for target in ("is_ai_mention", "is_ai_disclosure"):
        left, right = paired[f"{target}_a"].astype(bool), paired[f"{target}_b"].astype(bool)
        raw = float((left == right).mean())
        kappa = cohen_kappa(left, right)
        print(f"{target:20s} acuerdo bruto {raw:.3f} | kappa {kappa:.3f} | "
              f"positivos: {args.judge_a} {left.mean():.3f}, {args.judge_b} {right.mean():.3f}")
        report[target] = {"raw": raw, "kappa": kappa,
                          "rate_a": float(left.mean()), "rate_b": float(right.mean())}

    print(f"\n--- matriz de relevance (filas = {args.judge_a}, columnas = {args.judge_b}) ---")
    confusion = pd.crosstab(pd.Categorical(paired["relevance_a"], RELEVANCE_ORDER),
                            pd.Categorical(paired["relevance_b"], RELEVANCE_ORDER),
                            dropna=False)
    print(confusion.to_string())
    report["relevance_confusion"] = json.loads(confusion.to_json(orient="index"))
    report["relevance_kappa"] = cohen_kappa(paired["relevance_a"], paired["relevance_b"])
    print(f"kappa (3 categorías): {report['relevance_kappa']:.3f}")

    if "keyword_tier_a" in paired.columns:
        print("\n--- por tier léxico ---")
        rows = []
        for tier, group in paired.groupby("keyword_tier_a"):
            left = group["is_ai_mention_a"].astype(bool)
            right = group["is_ai_mention_b"].astype(bool)
            rows.append({"tier": tier, "n": len(group),
                         f"pos_{args.judge_a}": float(left.mean()),
                         f"pos_{args.judge_b}": float(right.mean()),
                         "acuerdo": float((left == right).mean()),
                         "kappa": cohen_kappa(left, right)})
        table = pd.DataFrame(rows)
        print(table.round(3).to_string(index=False))
        report["by_tier"] = table.to_dict("records")

    destination = REPO_ROOT / "data" / "interim" / "audits" / "golden_set" / "judge_agreement.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
