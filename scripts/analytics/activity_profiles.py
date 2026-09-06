"""Perfiles de actividades de IA divulgadas (capa conductual concreta).

Lee las actividades extraídas por `scripts/common/ai_activities_from_frames.py`
(la empresa hace ACCIÓN sobre OBJETO para FUNCIÓN, con etapa, destinatario,
proveedor y fuerza de evidencia) y las une a todas las instancias de cada
texto único (`gold_ai_frames`) para saber de qué empresa y canal son. Cada
actividad se cuenta UNA vez por empresa (texto único × índice de actividad),
así que el boilerplate repetido entre filings no infla el inventario.

No hay modelo econométrico ni re-clustering. Cuatro salidas:

  1. inventario por empresa (`firm_activity_profiles.parquet`): acciones,
     funciones, etapa máxima, mezcla interno/cliente, proveedores, evidencia.
  2. top behaviours del S&P 500: % de empresas (sobre las 510 con filings)
     que divulgan cada tipo de actividad.
  3. composición conductual de los segmentos de `02_segmentacion.md`.
  4. fichas de empresas ejemplares por segmento.

`function` y `provider_or_model` vienen libres del LLM; acá se agrupan en
familias por palabras clave (tablas FUNCTION_FAMILIES y PROVIDER_FAMILIES).
Todo es conducta DIVULGADA: lo que la empresa dice que hace.

Salida: `data/processed/clusters/activity_profiles.json`,
`firm_activity_profiles.parquet`, `firm_activities.parquet` (todas las
actividades con empresa y canal).
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
ACTIVITIES = REPO_ROOT / "data" / "interim" / "ai_activities"
CALLS_MANIFEST = REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
SEGMENT_LABELS = {"desplegadores_de_producto": "Product Deployers", "adoptantes_con_gobernanza": "Governance Adopters",
                  "listadores_de_riesgo": "Risk Listers", "sin_ia": "No AI"}
EXEMPLARS = {"desplegadores_de_producto": ["MSFT", "HPE", "PAYX", "ETSY", "NOW"],
             "adoptantes_con_gobernanza": ["JPM", "STT", "PWR", "DHR", "CINF"],
             "listadores_de_riesgo": ["NKE", "TDG", "CZR", "DHI", "TJX"]}

FUNCTION_FAMILIES = [
    ("customer_service", r"customer[_ ]?(service|support|experience|engagement|care)|contact[_ ]center|call[_ ]center|chatbot|virtual[_ ]assistant|service[_ ]desk|help[_ ]desk"),
    ("software_development", r"software|coding|code|develop|engineering|devops|programming|testing|qa\b"),
    ("marketing_and_sales", r"market|sales|advertis|personali|recommend|merchandis|pricing|promotion|campaign|lead[_ ]gen|e[_-]?commerce"),
    ("fraud_and_risk", r"fraud|risk|credit|underwrit|compliance|aml|anti[_ ]money|surveillance|claims|loss"),
    ("operations_and_supply_chain", r"operation|supply|logistic|inventory|manufactur|maintenance|forecast|demand|planning|scheduling|route|fleet|warehouse|procurement|quality[_ ]control|process[_ ]automation|automation|efficien|productivity|workflow"),
    ("it_and_cybersecurity", r"cyber|security|threat|it[_ ]operation|infrastructure|cloud|network|data[_ ]center|observability|monitoring"),
    ("research_and_product_development", r"research|r&d|discovery|drug|clinical|molec|design|innovation|product[_ ]dev|simulation|materials"),
    ("data_and_analytics", r"analytic|data|insight|intelligence|reporting|model(ing|ling)?$|decision"),
    ("hr_and_talent", r"\bhr\b|human[_ ]resource|talent|recruit|hiring|training|employee[_ ](experience|productivity)|workforce|learning"),
    ("finance_legal_and_admin", r"financ|account|legal|contract|document|invoice|audit|tax|back[_ ]office|administrat|procure"),
    ("content_and_media", r"content|media|creative|writing|translation|video|image|summar|search"),
    ("product_feature", r"product[_ ]feature|feature|platform|application|app\b|offering|solution|device|vehicle|autonom|robot"),
    ("healthcare_delivery", r"health|patient|clinic|diagnos|medical|care[_ ]delivery"),
]
PROVIDER_FAMILIES = [
    ("proprietary", r"^proprietary$|in[_ -]house|own\b"),
    ("OpenAI / Microsoft", r"openai|chatgpt|gpt|copilot|microsoft|azure"),
    ("Google", r"google|gemini|vertex|bard|deepmind|palm"),
    ("Anthropic", r"anthropic|claude"),
    ("NVIDIA", r"nvidia|cuda"),
    ("Amazon / AWS", r"amazon|aws|bedrock|sagemaker"),
    ("Meta", r"\bmeta\b|llama"),
    ("IBM", r"ibm|watson"),
    ("Salesforce", r"salesforce|einstein"),
    ("other_named", r"."),
]
STAGE_RANK = {"unspecified": 0, "exploring": 1, "piloting": 2, "deployed": 3, "scaled": 4}


def family(value: str | None, table: list[tuple[str, str]], default: str) -> str:
    if value is None or str(value).strip().lower() in ("", "unspecified", "none", "nan", "n/a"):
        return default
    v = str(value).strip().lower()
    for name, pattern in table:
        if re.search(pattern, v):
            return name
    return "other"


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(ACTIVITIES / "ai_activities__session=*.parquet")))
    if not files:
        raise SystemExit("sin actividades extraídas")
    acts = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    acts = acts[acts["error"].isna()]
    # último veredicto por texto único (una corrida puede reprocesar un texto que dio error antes)
    last = acts.groupby("text_hash")["session_id"].transform("max")
    acts = acts[acts["session_id"] == last]
    n_texts = acts["text_hash"].nunique()
    acts = acts[acts["has_activity"]].copy()
    con = duckdb.connect(str(DB), read_only=True)
    try:
        con.register("acts", acts[["text_hash"]].drop_duplicates())
        inst = con.execute(f"""
            WITH docs AS (
                SELECT accession_number, ticker, form_type AS form FROM filing_manifest WHERE country_code='us'
                UNION ALL SELECT accession_number, ticker, '10-Q' FROM filing_manifest_10q WHERE country_code='us'
                UNION ALL SELECT document_id, ticker, 'Earnings call' FROM read_parquet('{CALLS_MANIFEST}')
            )
            SELECT DISTINCT g.text_hash, d.ticker, CASE WHEN d.form = 'Earnings call' THEN 'call' ELSE 'filing' END AS channel
            FROM (SELECT DISTINCT text_hash, accession_number FROM gold_ai_frames WHERE country_code='us' AND has_frame) g
            JOIN acts USING (text_hash) JOIN docs d USING (accession_number)
            WHERE d.ticker IS NOT NULL
        """).df()
    finally:
        con.close()
    inst["text_hash"] = inst["text_hash"].astype("uint64")
    a = acts.merge(inst, on="text_hash", how="inner")
    # una actividad única por empresa; si el mismo texto aparece en call y filing, se anota el canal 'filing'
    a["channel_rank"] = (a["channel"] == "call").astype(int)
    a = a.sort_values("channel_rank").drop_duplicates(["ticker", "text_hash", "activity_index"]).drop(columns="channel_rank")
    a["function_family"] = a["function"].map(lambda v: family(v, FUNCTION_FAMILIES, "unspecified"))
    a["provider_family"] = a["provider_or_model"].map(lambda v: family(v, PROVIDER_FAMILIES, "unspecified"))
    a["stage_rank"] = a["stage"].map(STAGE_RANK).fillna(0).astype(int)
    a.attrs["n_texts"] = int(n_texts)
    return a


def flags(a: pd.DataFrame) -> pd.DataFrame:
    """Indicadores por actividad de los 'top behaviours'."""
    f = pd.DataFrame(index=a.index)
    used = a["action"].isin(["deploy", "integrate", "scale"])
    f["customer_facing_deployment"] = used & (a["target"] == "customers")
    f["internal_deployment"] = used & a["target"].isin(["employees", "internal_process"])
    f["developer_tools"] = used & (a["target"] == "developers")
    f["proprietary_ai"] = (a["action"] == "develop") | (a["provider_family"] == "proprietary")
    f["third_party_named_provider"] = ~a["provider_family"].isin(["proprietary", "unspecified"])
    f["infrastructure_investment"] = a["action"] == "invest_infrastructure"
    f["acquisition_or_licensing"] = a["action"] == "buy_or_license"
    f["partnership"] = a["action"] == "partner"
    f["talent_or_training"] = a["action"] == "hire_or_train"
    f["quantified_outcome"] = (a["action"] == "measure_outcome") | (a["evidence_strength"] == "metric")
    f["governance_or_restriction"] = a["action"].isin(["govern_or_control", "restrict"])
    f["piloting_or_exploring"] = a["action"] == "pilot_or_explore"
    f["named_product_or_process"] = a["evidence_strength"] == "named_product_or_process"
    return f


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
    top = lambda col, k: g[col].apply(lambda s: ", ".join(v for v in s.value_counts().index[:k] if v not in ("unspecified", "other")))
    prof["main_actions"] = top("action", 3)
    prof["main_functions"] = top("function_family", 3)
    prof["main_objects"] = top("object", 3)
    prof["providers"] = top("provider_family", 3)
    prof = universe.merge(prof, left_on="ticker", right_index=True, how="left")
    prof["n_activities"] = prof["n_activities"].fillna(0).astype(int)
    for c in [c for c in prof.columns if c.startswith("any_")]:
        prof[c] = prof[c].fillna(False).astype(bool)
    prof["max_stage"] = prof["max_stage"].fillna("none")
    return prof


def main() -> None:
    a = load()
    seg = pd.read_parquet(OUT_DIR / "firm_segments.parquet")[["ticker", "segmento"]]
    prof = firm_profiles(a, seg)
    prof.to_parquet(OUT_DIR / "firm_activity_profiles.parquet", index=False)
    a.drop(columns=["stage_rank"]).to_parquet(OUT_DIR / "firm_activities.parquet", index=False)
    n_firms = len(seg)
    print(f"actividades: {len(a):,} únicas por empresa, de {a.attrs['n_texts']:,} textos únicos procesados | "
          f"empresas con ≥1 actividad: {int((prof.n_activities > 0).sum())} de {n_firms} | "
          f"{(a.channel == 'call').mean():.0%} provienen sólo de calls")

    print("\nACCIONES (% de actividades) y ETAPA, DESTINATARIO, EVIDENCIA")
    dist = {k: (a[k].value_counts(normalize=True) * 100).round(1).to_dict() for k in ("action", "stage", "target", "evidence_strength")}
    for k, v in dist.items():
        print(f"  {k:18s} " + " | ".join(f"{kk} {vv}" for kk, vv in v.items()))
    fam = (a["function_family"].value_counts(normalize=True) * 100).round(1)
    prov = (a["provider_family"].value_counts(normalize=True) * 100).round(1)
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
    print("  etapa máxima alcanzada: " + " | ".join(f"{k} {v}%" for k, v in stage_share.items()))

    print("\nCOMPOSICIÓN CONDUCTUAL DE LOS SEGMENTOS — % de empresas del segmento con ≥1 actividad de cada tipo")
    by_seg = (prof.groupby("segmento")[any_cols].mean() * 100).round(1).rename(columns=lambda c: c[4:])
    by_seg["n_firms"] = prof.groupby("segmento").size()
    by_seg["median_activities"] = prof.groupby("segmento")["n_activities"].median()
    by_seg["share_customers"] = (prof.groupby("segmento")["share_customers"].mean() * 100).round(1)
    by_seg["share_internal"] = (prof.groupby("segmento")["share_internal"].mean() * 100).round(1)
    order = ["desplegadores_de_producto", "adoptantes_con_gobernanza", "listadores_de_riesgo", "sin_ia"]
    by_seg = by_seg.reindex(order)
    print(by_seg.T.to_string())
    seg_stage = pd.crosstab(prof["segmento"], prof["max_stage"], normalize="index").reindex(order) * 100
    print("\netapa máxima por segmento (%):"); print(seg_stage.round(1).to_string())
    seg_func = pd.crosstab(a.merge(seg, on="ticker")["segmento"], a.merge(seg, on="ticker")["function_family"], normalize="index").reindex(order[:3]) * 100
    print("\nfunciones por segmento (% de actividades):"); print(seg_func.round(1).T.to_string())

    print("\nFICHAS — empresas ejemplares")
    cards = {}
    for s, tickers in EXEMPLARS.items():
        print(f"  [{SEGMENT_LABELS[s]}]")
        for t in tickers:
            r = prof[prof.ticker == t]
            if r.empty:
                continue
            r = r.iloc[0]
            sub = a[a.ticker == t]
            objs = ", ".join(v for v in sub["object"].value_counts().index[:5])
            provs = ", ".join(v for v in sub["provider_or_model"].value_counts().index[:4] if v.lower() not in ("unspecified",))
            card = {"segment": SEGMENT_LABELS[s], "n_activities": int(r.n_activities), "main_actions": r.main_actions,
                    "main_functions": r.main_functions, "main_objects": objs, "max_stage": r.max_stage,
                    "share_customers": round(float(r.share_customers or 0), 2), "share_internal": round(float(r.share_internal or 0), 2),
                    "providers": provs, "share_quantified": round(float(r.share_quantified_outcome or 0), 2),
                    "share_named": round(float(r.share_named_product_or_process or 0), 2)}
            cards[t] = card
            print(f"    {t:6s} n={card['n_activities']:4d} | {card['main_actions']} | {card['main_functions']} | objects: {objs} | "
                  f"stage {card['max_stage']} | customers {card['share_customers']:.0%} internal {card['share_internal']:.0%} | "
                  f"providers: {provs} | named {card['share_named']:.0%} quantified {card['share_quantified']:.0%}")

    payload = {"n_activities": int(len(a)), "n_texts": a.attrs["n_texts"], "n_firms": n_firms,
               "firms_with_activity": int((prof.n_activities > 0).sum()), "share_from_calls": float((a.channel == "call").mean()),
               "distributions": dist, "function_families": fam.to_dict(), "provider_families": prov.to_dict(),
               "top_behaviours": top.to_dict(), "top_behaviours_filings_only": top_f.to_dict(), "max_stage": stage_share.to_dict(),
               "by_segment": json.loads(by_seg.to_json(orient="index")), "stage_by_segment": json.loads(seg_stage.round(1).to_json(orient="index")),
               "functions_by_segment": json.loads(seg_func.round(1).to_json(orient="index")), "exemplars": cards}
    (OUT_DIR / "activity_profiles.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\n-> {OUT_DIR}/activity_profiles.json, firm_activity_profiles.parquet, firm_activities.parquet")


if __name__ == "__main__":
    main()
