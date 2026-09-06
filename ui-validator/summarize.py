"""Resume las anotaciones exportadas desde la UI (`annotations/*.json`).

Reporta, para las actividades divulgadas: existencia y precisión por campo
(acción, objeto, función, destinatario, etapa, proveedor, evidencia). Para
los frames: existencia, acuerdo humano–juez y κ de Cohen en
promocional y temporal, acuerdo en specificity y evidencia. Para el
prefiltro: precisión y recall del prefiltro contra el humano, crudos y
reponderados por estrato (la muestra sobre-representa la zona gris y los
negativos con término; los pesos devuelven cada estrato a su tamaño real
en `predictions_run`).

    uv run --frozen --no-sync python ui-validator/summarize.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
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
        obj = json.loads(path.read_text())
        for k, v in (obj.get("annotations") or obj).items():
            if k not in merged or (v.get("ts") or "") > (merged[k].get("ts") or ""):
                merged[k] = v
    return merged


def main() -> None:
    data = json.loads(DATA.read_text())
    ann = load_annotations()
    if not ann:
        raise SystemExit(f"no hay anotaciones en {ANN_DIR}/ — exportá desde la UI y guardá el JSON ahí")

    # ---- frames ----
    exist, promo, temporal, spec, evid, missing = [], [], [], [], [], []
    for it in data["frames"]:
        a = ann.get(it["id"])
        if not a:
            continue
        if not it["frames"] and a.get("sin_frames") is not None:
            missing.append(bool(a["sin_frames"]))
        for f in it["frames"]:
            p = f"f{f['frame_index']}."
            if a.get(p + "existe") is not None:
                exist.append(bool(a[p + "existe"]))
            if a.get(p + "promocional") is not None:
                promo.append((bool(a[p + "promocional"]), bool(f["promotional"])))
            if a.get(p + "temporal") is not None:
                temporal.append((a[p + "temporal"], f["temporal"]))
            if a.get(p + "specificity_ok") is not None:
                spec.append(bool(a[p + "specificity_ok"]))
            if a.get(p + "evidencia_ok") is not None:
                evid.append(bool(a[p + "evidencia_ok"]))
    print("FRAMES")
    if exist:
        print(f"  existen: {sum(exist)}/{len(exist)} ({100*sum(exist)/len(exist):.1f}%)")
    if missing:
        print(f"  párrafos sin frame que debían tener uno: {sum(missing)}/{len(missing)}")
    if promo:
        print(f"  promocional: acuerdo {100*sum(a==b for a,b in promo)/len(promo):.1f}%, κ={kappa(promo):.3f} (n={len(promo)})")
        h_yes = sum(a for a, _ in promo); j_yes = sum(b for _, b in promo)
        print(f"    humano dice sí en {h_yes}, juez en {j_yes}")
    if temporal:
        print(f"  temporal: acuerdo {100*sum(a==b for a,b in temporal)/len(temporal):.1f}%, κ={kappa(temporal):.3f} (n={len(temporal)})")
    if spec:
        print(f"  specificity correcta: {100*sum(spec)/len(spec):.1f}% (n={len(spec)})")
    if evid:
        print(f"  evidencia correcta: {100*sum(evid)/len(evid):.1f}% (n={len(evid)})")

    # ---- actividades ----
    verd = Counter(); wrong = Counter(); n_fields = 0; falta = []
    for it in data.get("activities", []):
        a = ann.get(it["id"])
        if not a:
            continue
        if a.get("falta") is not None:
            falta.append(bool(a["falta"]))
        for x in it["activities"]:
            p = f"a{x['activity_index']}."
            v = a.get(p + "veredicto")
            if not v:
                continue
            verd[v] += 1
            if v in ("ok", "mal"):
                n_fields += 1
                for k, w in (a.get(p + "mal") or {}).items():
                    if w:
                        wrong[k] += 1
    if verd:
        n = sum(verd.values())
        print(f"\nACTIVIDADES (n anotado = {n} actividades en {len(falta)} párrafos)")
        if falta:
            print(f"  párrafos a los que falta alguna actividad (recall): {sum(falta)}/{len(falta)} ({100*sum(falta)/len(falta):.1f}%)")
        print(f"  existe y está bien: {verd['ok']}/{n} ({100*verd['ok']/n:.1f}%) | existe con algún campo mal: {verd['mal']} | no existe: {verd['no_existe']} ({100*verd['no_existe']/n:.1f}%)")
        print("  precisión por campo, entre las actividades que existen:")
        for k in ("action", "object", "function", "target", "stage", "provider", "own", "evidence_strength"):
            print(f"    {k:18s} {100*(1 - wrong[k]/max(1, n_fields)):.1f}%  ({wrong[k]} marcadas mal de {n_fields})")

    # ---- prefiltro, reponderado por estrato ----
    run = REPO_ROOT / "data" / "interim" / "prefilter_predictions_unique" / data["predictions_run"]
    con = duckdb.connect()
    pop = con.execute(f"""
        SELECT form, CASE WHEN is_ai_prefiltered THEN 'positivo' WHEN predicted_proba >= 0.05 THEN 'zona_gris' ELSE 'negativo' END estrato, count(*) n
        FROM read_parquet('{run}') WHERE country_code = 'us' GROUP BY 1, 2
    """).fetchall()
    pop_n = {(f, e): n for f, e, n in pop}
    sample_n = Counter((it["form"], it["estrato"]) for it in data["prefilter"])
    cells = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    raw = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    n_ann = 0
    for it in data["prefilter"]:
        a = ann.get(it["id"])
        if not a or a.get("menciona_ia") is None:
            continue
        n_ann += 1
        w = pop_n.get((it["form"], it["estrato"]), 0) / max(1, sample_n[(it["form"], it["estrato"])])
        k = ("tp" if a["menciona_ia"] else "fp") if it["prefilter_says_ai"] else ("fn" if a["menciona_ia"] else "tn")
        raw[k] += 1; cells[k] += w
    print("\nPREFILTRO (n anotado =", n_ann, ")")
    if n_ann:
        for label, c in (("crudo", raw), ("reponderado por estrato", cells)):
            prec = c["tp"] / max(1e-9, c["tp"] + c["fp"]); rec = c["tp"] / max(1e-9, c["tp"] + c["fn"])
            print(f"  {label:26s} precisión {100*prec:.1f}%  recall {100*rec:.1f}%  "
                  f"(tp {c['tp']:.0f}, fp {c['fp']:.0f}, fn {c['fn']:.0f}, tn {c['tn']:.0f})")
        print("  nota: el recall reponderado sólo cubre negativos CON término de IA; los sin término no se muestrean.")


if __name__ == "__main__":
    main()
