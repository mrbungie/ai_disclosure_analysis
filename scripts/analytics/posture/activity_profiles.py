"""Perfiles de actividades de IA divulgadas (capa conductual concreta) --
reporte descriptivo, sin modelo econométrico ni re-clustering.

Lee el inventario de actividades ya materializado en el grano `activity` de
gold (`data/gold/spines/activity/activity.parquet` +
`data/gold/covariates/activity/{extraction,taxonomy}.parquet`, construidos por
`scripts/gold/activity/build_activity.py` desde
`scripts/enrichment/ai_activities_from_frames.py` + `silver.ai_frames`) y el
archetype de postura de `data/gold/covariates/firm/posture_archetype_static.parquet`
(el mismo fit k=3 de `scripts/gold/firm/build_posture_archetype_static.py`).
No se re-extrae ni re-agrega nada que gold ya produjo: este script sólo
etiqueta y reporta.

Cuatro salidas, todas a `data/results/posture/`:

  1. inventario por empresa (`firm_activity_profiles.parquet`): acciones,
     funciones, etapa máxima, mezcla interno/cliente, proveedores, evidencia.
  2. top behaviours del S&P 500: % de empresas (sobre las 510 con filings)
     que divulgan cada tipo de actividad.
  3. composición conductual de los archetypes de `build_posture_archetype_static.py`.
  4. fichas de empresas ejemplares por archetype.

`function` y `provider_or_model` vienen libres del LLM; las familias
(FUNCTION_FAMILIES/PROVIDER_FAMILIES/OBJECT_FAMILIES) y `flags()` viven en
`scripts/gold/activity/build_activity.py` (que ya las aplicó al
covariate) y se reimportan aquí para no duplicar la taxonomía.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "activity"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
from build_activity import ACTIVITY_FAMILIES, GROUNDING, PROVIDER_FAMILIES, STAGE_RANK, family, flags  # noqa: E402

ACTIVITY_SPINE = L.gold_path("spines", "activity", "activity")


def ranked(s: pd.Series) -> pd.Index:
    """Valores de `s` de más a menos frecuente; los empates, en orden alfabético
    (no en el orden de filas de gold)."""
    vc = s.value_counts()
    return vc.sort_index().sort_values(ascending=False, kind="stable").index


def _n_texts_judged() -> int:
    manifest = ACTIVITY_SPINE.parent / f"{ACTIVITY_SPINE.name.removesuffix('.parquet')}._manifest.json"
    return int(json.loads(manifest.read_text())["n_texts_judged"])


def concreteness(prof: pd.DataFrame) -> pd.Series:
    """Concreción conductual: media de cinco proporciones de las actividades de la
    empresa (función declarada, desplegada o escalada, producto o proceso con
    nombre, resultado cuantificado, proveedor nombrado). Sólo interpretación y
    robustez; no reemplaza el eje de conducta de `03`."""
    return prof[[f"share_{c}" for c in GROUNDING]].mean(axis=1)


def firm_profiles(a: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    fl = flags(a)
    g = a.groupby("ticker")
    prof = pd.DataFrame({"n_activities": g.size(), "n_texts": g["text_hash"].nunique(),
                         "share_calls": g["channel"].apply(lambda s: float((s == "call").mean()))})
    for c in fl.columns:
        prof[f"any_{c}"] = fl.groupby(a["ticker"])[c].any()
        prof[f"share_{c}"] = fl.groupby(a["ticker"])[c].mean()
    prof["max_stage"] = g["stage_rank"].max().map({v: k for k, v in STAGE_RANK.items()})
    prof["share_scaled_or_deployed"] = g["stage_rank"].apply(lambda s: float((s >= 3).mean()))
    prof["share_customers"] = g["target"].apply(lambda s: float((s == "customers").mean()))
    prof["share_internal"] = g["target"].apply(lambda s: float(s.isin(["employees", "internal_process"]).mean()))
    top = lambda col, k: g[col].apply(lambda s: ", ".join(v for v in ranked(s)[:k] if v not in ("unspecified", "other")))
    prof["main_actions"] = top("action", 3)
    prof["main_functions"] = top("function_family", 3)
    prof["main_objects"] = top("object", 3)
    prof["providers"] = top("provider_family", 3)
    prof = universe.merge(prof, left_on="ticker", right_index=True, how="left")
    prof["n_activities"] = prof["n_activities"].fillna(0).astype(int)
    for c in [c for c in prof.columns if c.startswith("any_")]:
        prof[c] = prof[c].fillna(False).astype(bool)
    prof["max_stage"] = prof["max_stage"].fillna("none")
    prof["concrecion_conductual"] = concreteness(prof)
    return prof


def main() -> None:
    a = L.read_dataset("activity", ("covariates", "extraction"), ("covariates", "taxonomy"))
    a.attrs["n_texts"] = _n_texts_judged()
    a["stage_rank"] = a["stage"].map(STAGE_RANK).fillna(0).astype(int)
    # Universo de empresas + etiqueta de caracterización: el mismo archetype
    # k=3 (pooled, whole-panel) que usa el resto de la tesis -- no el
    # "segmento" del viejo K-means de Capítulo 3 (build_segments.py,
    # deprecado 2026-09-13).
    seg = L.read_gold("firm", ("covariates", "posture_archetype_static", ["archetype"]))[["ticker", "archetype"]]
    prof = firm_profiles(a, seg)
    RESULTS_DIR = L.results_path("posture", "firm_activity_profiles.parquet").parent
    prof.to_parquet(RESULTS_DIR / "firm_activity_profiles.parquet", index=False)
    n_firms = len(seg)
    print(f"actividades: {len(a):,} únicas por empresa, de {a.attrs['n_texts']:,} textos únicos procesados | "
          f"empresas con ≥1 actividad: {int((prof.n_activities > 0).sum())} de {n_firms} | "
          f"{(a.channel == 'call').mean():.0%} provienen sólo de calls")

    print("\nACCIONES (% de actividades) y ETAPA, DESTINATARIO, EVIDENCIA")
    dist = {k: (a[k].value_counts(normalize=True) * 100).round(1).to_dict() for k in ("action", "stage", "target", "evidence_type", "source")}
    for k, v in dist.items():
        print(f"  {k:18s} " + " | ".join(f"{kk} {vv}" for kk, vv in v.items()))
    fam = (a["function_family"].value_counts(normalize=True) * 100).round(1)
    prov = (a["provider_family"].value_counts(normalize=True) * 100).round(1)
    multi = float(a["provider_families"].map(lambda v: len(set(v) - {"proprietary", "unspecified", "third_party_unnamed"}) >= 2).mean())
    roles = pd.Series([e["role"] for es in a["entities"] for e in es]).value_counts()
    print(f"  actividades con ≥2 proveedores externos nombrados: {100 * multi:.1f}% | con marca propia nombrada: {100 * (a['own_brands'].map(len) > 0).mean():.1f}% | entidades nombradas por rol: " + ", ".join(f"{k} {v}" for k, v in roles.items()))
    print("  function family    " + " | ".join(f"{k} {v}" for k, v in fam.items()))
    print("  provider family    " + " | ".join(f"{k} {v}" for k, v in prov.items()))

    print("\nTOP BEHAVIOURS — % de las 510 empresas que divulgan al menos una actividad de cada tipo")
    any_cols = [c for c in prof.columns if c.startswith("any_")]
    top = (prof[any_cols].mean() * 100).round(1).rename(lambda c: c[4:]).sort_values(ascending=False)
    stage_share = (prof["max_stage"].value_counts(normalize=True) * 100).round(1)
    filings_only = firm_profiles(a[a.channel == "filing"], seg)
    top_f = (filings_only[any_cols].mean() * 100).round(1).rename(lambda c: c[4:])
    for k, v in top.items():
        print(f"  {k:32s} {v:5.1f}%   (sólo filings: {top_f[k]:5.1f}%)")

    def firm_share(frame: pd.DataFrame, key: str, denom: int, k: int = 30) -> pd.DataFrame:
        """% de empresas (sobre `denom`) con ≥1 actividad de cada valor de `key`,
        con los objetos literales más frecuentes como ejemplo."""
        g = frame.groupby(key)
        out = pd.DataFrame({"firms": g["ticker"].nunique(), "activities": g.size(),
                            "examples": g["object"].apply(lambda s: ", ".join(ranked(s)[:4]))})
        out["pct_firms"] = (100 * out["firms"] / denom).round(1)
        return out.sort_values("firms", ascending=False).head(k)

    print("\nACTIVIDADES CONCRETAS MÁS COMUNES — acción · objeto: % de las 510 empresas, con objetos literales de ejemplo")
    concrete = a[a["object_family"] != "AI, unspecified object"]
    top_act = firm_share(concrete, "activity", n_firms, 30)
    print(top_act[["pct_firms", "firms", "activities", "examples"]].to_string())
    print("\nACCIÓN · OBJETO · FUNCIÓN (con función declarada) — % de las 510 empresas")
    top_actf = firm_share(concrete[concrete["function_family"].isin(["unspecified", "other"]) == False], "activity_function", n_firms, 30)
    print(top_actf[["pct_firms", "firms", "activities", "examples"]].to_string())
    print("\nOBJETOS (familia) — % de empresas que nombran al menos uno")
    top_obj = firm_share(a, "object_family", n_firms, 20)
    print(top_obj[["pct_firms", "firms", "examples"]].to_string())
    print("\nPROVEEDORES NOMBRADOS — % de empresas que nombran cada familia, y los nombres literales más frecuentes")
    expl = a[["ticker", "object", "providers_or_models"]].explode("providers_or_models").dropna(subset=["providers_or_models"])
    expl["provider_family"] = expl["providers_or_models"].map(lambda v: family(v, PROVIDER_FAMILIES, "unspecified"))
    named = expl[~expl["provider_family"].isin(["proprietary", "unspecified", "third_party_unnamed"])]
    prov_firms = firm_share(named, "provider_family", n_firms, 12)
    prov_firms["examples"] = named.groupby("provider_family")["providers_or_models"].apply(lambda s: ", ".join(ranked(s)[:5]))
    print(prov_firms[["pct_firms", "firms", "examples"]].to_string())
    print("  empresas que nombran algún proveedor externo:", named["ticker"].nunique(), "de", n_firms)

    print("\nCOMPOSICIÓN CONDUCTUAL DE LOS ARQUETIPOS — % de empresas del archetype con ≥1 actividad de cada tipo")
    by_seg = (prof.groupby("archetype")[any_cols].mean() * 100).round(1).rename(columns=lambda c: c[4:])
    by_seg["n_firms"] = prof.groupby("archetype").size()
    by_seg["median_activities"] = prof.groupby("archetype")["n_activities"].median()
    by_seg["share_customers"] = (prof.groupby("archetype")["share_customers"].mean() * 100).round(1)
    by_seg["share_internal"] = (prof.groupby("archetype")["share_internal"].mean() * 100).round(1)
    non_ai = [s for s in prof["archetype"].dropna().unique() if s != "No AI"]
    order = sorted(non_ai, key=lambda s: -int(prof[prof["archetype"] == s].shape[0])) + (["No AI"] if "No AI" in prof["archetype"].values else [])
    by_seg = by_seg.reindex(order)
    print(by_seg.T.to_string())
    seg_stage = pd.crosstab(prof["archetype"], prof["max_stage"], normalize="index").reindex(order) * 100
    aseg = a.merge(seg, on="ticker")
    seg_func = pd.crosstab(aseg["archetype"], aseg["function_family"], normalize="index").reindex(order[:3]) * 100

    print("\nACTIVIDADES CONCRETAS POR ARQUETIPO — top 12 acción · objeto, % de empresas del archetype")
    seg_top = {}
    for s in order[:3]:
        sub = aseg[(aseg["archetype"] == s) & (aseg["object_family"] != "AI, unspecified object")]
        tbl = firm_share(sub, "activity", int(by_seg.loc[s, "n_firms"]), 12)
        seg_top[s] = json.loads(tbl.to_json(orient="index"))
        print(f"  [{s}]")
        for k, r in tbl.iterrows():
            print(f"    {r.pct_firms:5.1f}%  {k:45s} e.g. {r.examples}")
    seg_actf = {}
    print("\nACCIÓN · OBJETO · FUNCIÓN POR ARQUETIPO — top 10 con función declarada, % de empresas del archetype")
    for s in order[:3]:
        sub = aseg[(aseg["archetype"] == s) & (aseg["object_family"] != "AI, unspecified object") & ~aseg["function_family"].isin(["unspecified", "other"])]
        tbl = firm_share(sub, "activity_function", int(by_seg.loc[s, "n_firms"]), 10)
        seg_actf[s] = json.loads(tbl.to_json(orient="index"))
        print(f"  [{s}]")
        for k, r in tbl.iterrows():
            print(f"    {r.pct_firms:5.1f}%  {k}")

    print("\nFICHAS — inventario de actividades concretas por empresa ejemplar")
    cards = {}
    exemplars = {s: prof[prof["archetype"] == s].sort_values("n_activities", ascending=False)["ticker"].head(5).tolist()
                 for s in order[:3]}
    for s, tickers in exemplars.items():
        print(f"  [{s}]")
        for t_ in tickers:
            sub = a[a.ticker == t_]
            if sub.empty:
                continue
            g = sub.groupby(["action", "object_family"])
            inv = pd.DataFrame({"n": g.size(),
                                "functions": g["function_family"].apply(lambda x: ", ".join(v for v in ranked(x)[:2] if v not in ("unspecified", "other"))),
                                "targets": g["target"].apply(lambda x: ", ".join(v for v in ranked(x)[:2] if v != "unspecified")),
                                "objects": g["object"].apply(lambda x: ", ".join(ranked(x)[:3])),
                                "providers": g["providers_or_models"].apply(lambda x: ", ".join(v for v in ranked(pd.Series([p for lst in x for p in lst], dtype=object))[:3] if v.lower() not in ("unspecified", "proprietary"))),
                                "own": g["own_brands"].apply(lambda x: ", ".join(ranked(pd.Series([p for lst in x for p in lst], dtype=object))[:3])),
                                "source": g["source"].apply(lambda x: ranked(x)[0]),
                                "stage": g["stage_rank"].max().map({v: k for k, v in STAGE_RANK.items()}),
                                "named_or_metric": g["evidence_type"].apply(lambda x: float(x.isin(["named", "metric", "vendor"]).mean()))
                                }).sort_values("n", ascending=False).head(8).reset_index()
            lines = [f"{r.action} · {r.object_family} ({r.n}): {r.objects}" + (f" | for {r.functions}" if r.functions else "")
                     + (f" | {r.targets}" if r.targets else "") + f" | {r.stage}" + f" | {r.source}" + (f" | providers: {r.providers}" if r.providers else "") + (f" | own: {r.own}" if r.own else "")
                     + f" | concrete {r.named_or_metric:.0%}" for r in inv.itertuples()]
            cards[t_] = {"archetype": s, "n_activities": int(len(sub)), "lines": lines}
            print(f"    {t_} ({len(sub)} activities)")
            for ln in lines:
                print(f"      - {ln}")

    payload = {"n_activities": int(len(a)), "n_texts": a.attrs["n_texts"], "n_firms": n_firms,
               "firms_with_activity": int((prof.n_activities > 0).sum()), "share_from_calls": float((a.channel == "call").mean()),
               "distributions": dist, "function_families": fam.to_dict(), "provider_families": prov.to_dict(),
               "top_behaviours": top.to_dict(), "top_behaviours_filings_only": top_f.to_dict(), "max_stage": stage_share.to_dict(),
               "by_archetype": json.loads(by_seg.to_json(orient="index")), "stage_by_archetype": json.loads(seg_stage.round(1).to_json(orient="index")),
               "functions_by_archetype": json.loads(seg_func.round(1).to_json(orient="index")), "exemplars": cards,
               "top_concrete_activities": json.loads(top_act.to_json(orient="index")),
               "top_activity_function": json.loads(top_actf.to_json(orient="index")),
               "object_families": json.loads(top_obj.to_json(orient="index")),
               "named_providers": json.loads(prov_firms.to_json(orient="index")), "firms_naming_provider": int(named["ticker"].nunique()),
               "archetype_top_activities": seg_top, "archetype_top_activity_function": seg_actf}
    (RESULTS_DIR / "activity_profiles.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\n-> {RESULTS_DIR}/activity_profiles.json, firm_activity_profiles.parquet")


if __name__ == "__main__":
    main()
