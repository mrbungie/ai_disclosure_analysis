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
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_intensity import document_table  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DB = REPO_ROOT / "duckdb" / "thesis.duckdb"
ACTIVITIES = REPO_ROOT / "data" / "interim" / "ai_activities"
CALLS_MANIFEST = REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet"
OUT_DIR = REPO_ROOT / "data" / "processed" / "clusters"
SEGMENT_LABELS = {"desplegadores_de_producto": "Product Deployers", "adoptantes_con_gobernanza": "Governance Adopters",
                  "listadores_de_riesgo": "Risk Listers", "sin_ia": "No AI"}
EXEMPLARS = {"desplegadores_de_producto": ["MSFT", "HPE", "PAYX", "ETSY", "NOW"],
             "adoptantes_con_gobernanza": ["JPM", "STT", "LOW", "DHR", "CINF"],
             "listadores_de_riesgo": ["NKE", "BAC", "CMA", "HWM", "TDG"]}

FUNCTION_FAMILIES = [
    ("governance", r"govern|oversight|responsible[_ ]ai|ai[_ ]ethic|transparen|trust[_ ]and[_ ]safety|policy|compliance[_ ]program"),
    ("customer_service", r"customer[_ ]?(service|support|experience|engagement|care|facing|outcome)|client[_ ](service|experience|support)|contact[_ ]center|call[_ ]center|chatbot|virtual[_ ]assistant|service[_ ]desk|help[_ ]desk|service[_ ]delivery"),
    ("financial_services_operations", r"trading|loan|lending|underwrit|payment|transaction|payroll|banking|wealth|portfolio|insurance|claims|actuar|investment[_ ]management"),
    ("software_development", r"software|coding|code|develop|engineering|devops|programming|testing|qa\b"),
    ("marketing_and_sales", r"market|sales|advertis|personali|recommend|merchandis|pricing|promotion|campaign|lead[_ ]gen|e[_-]?commerce"),
    ("fraud_and_risk", r"fraud|risk|credit|underwrit|compliance|aml|anti[_ ]money|surveillance|claims|loss"),
    ("operations_and_supply_chain", r"operation|supply|logistic|inventory|manufactur|maintenance|forecast|demand|planning|scheduling|route|fleet|warehouse|procurement|quality[_ ]control|process|automation|efficien|productivity|workflow|cost[_ ]reduction|digital[_ ]transformation|energy[_ ]management|building[_ ]management|transportation|knowledge[_ ]management|collaboration"),
    ("it_and_cybersecurity", r"cyber|security|threat|malware|identity|authenticat|it[_ ]operation|infrastructure|cloud|network|data[_ ]center|observability|monitoring|incident|troubleshoot"),
    ("ai_compute_and_models", r"ai[_ ]workload|ai[_ ]comput|compute|high[_ ]performance|hpc|ai[_ ]inference|ai[_ ]training|ai[_ ]acceleration|ai[_ ]processing|scientific[_ ]computing|edge[_ ]computing|model[_ ](training|development)|generative[_ ]ai$|^ai$|llm"),
    ("research_and_product_development", r"research|r&d|discovery|drug|clinical|molec|design|innovation|product[_ ]dev|simulation|materials"),
    ("data_and_analytics", r"analytic|data|insight|intelligence|reporting|model(ing|ling)?$|decision"),
    ("hr_and_talent", r"\bhr|human[_ ]resource|talent|recruit|hir(e|ing)|training|upskill|employee|workforce|learning|performance[_ ]management"),
    ("finance_legal_and_admin", r"financ|account|legal|contract|document|invoice|audit|tax|back[_ ]office|administrat|procure"),
    ("content_and_media", r"content|media|creative|writing|translation|video|image|summar|search"),
    ("product_feature", r"product|feature|platform|application|app\b|offering|solution|device|vehicle|autonom|robot|driving|automotive|driver|gaming|commerce|surgery|consulting|revenue"),
    ("healthcare_delivery", r"health|patient|clinic|diagnos|medical|care[_ ]delivery"),
]
PROVIDER_FAMILIES = [
    ("proprietary", r"^proprietary|in[_ -]house|own\b"),
    ("third_party_unnamed", r"third[_ -]?part|vendor|external|open[_ ]source|generative ai$|^genai$|^llms?$|^ai$"),
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
OBJECT_FAMILIES = [
    ("copilot or assistant", r"copilot|assistant|assist\b|agentforce|ai overviews|advisor|concierge"),
    ("AI agents", r"agent"),
    ("chatbot", r"chatbot|chat bot|conversational|virtual agent"),
    ("LLM or foundation model", r"large language|llm|foundation model|generative ai model|gpt|language model"),
    ("ML or predictive model", r"machine learning model|ml model|predictive|algorithm|\bmodels?\b|forecasting|scoring"),
    ("compute and infrastructure", r"data cent|compute|gpu|accelerat|chip|server|infrastructure|cloud capacity|supercomput|workload|ai training|ai pc|silicon|processor|hardware"),
    ("talent", r"talent|employee|workforce|engineer|scientist|team|hiring|training program|upskill"),
    ("automation", r"automation|robotic process|rpa|automat"),
    ("search or recommendation", r"\bsearch|recommend|personali|ranking|discovery engine|matching"),
    ("external model or provider", r"openai|nvidia|gemini|anthropic|claude|microsoft|azure|google|aws|bedrock|llama|chatgpt|gpt|third.party|partner|vendor|supplier|hyperscaler"),
    ("financial outcome", r"revenue|saving|cost|margin|efficienc|productivity|profit|growth|booking|sales|return"),
    ("governance framework", r"governance|responsible ai|policy|policies|framework|ethic|oversight|control|guardrail|committee"),
    ("acquired company or license", r"acquisition|acquire|company|companies|license|startup|business unit"),
    ("use case or project", r"use case|project|proof of concept|poc|roadmap|pilot|initiative|program"),
    ("analytics or data platform", r"analytic|data (cloud|platform|lake)|insight|dashboard|data science|big data|platform"),
    ("security tool", r"security|firewall|threat|fraud|detection|malware|surveillance"),
    ("vision, robotics or autonomy", r"vision|autonom|self-driving|driving|robot|drone|imaging|sensor|vehicle|camera"),
    ("product feature or service", r"feature|product|offering|solution|service|application|app\b|tool|software|capabilit|experience|engine|system"),
    ("AI, unspecified object", r"^ai$|^artificial intelligence|^generative ai$|^gen ?ai$|^ai technolog|^ai and|^ai/ml|^machine learning$|^ai initiative|^ai strateg|^ai use|^ai program|^ai investment|^technolog|^new technolog|^ai$"),
]
STAGE_RANK = {"unspecified": 0, "exploring": 1, "piloting": 2, "deployed": 3, "scaled": 4}


GENERIC_TOKENS = {"ai", "artificial", "intelligence", "generative", "gen", "genai", "ml", "machine", "learning", "and", "or", "&",
                  "capabilities", "capability", "technologies", "technology", "tools", "tool", "solutions", "solution", "systems",
                  "system", "applications", "application", "services", "service", "products", "product", "offerings", "offering",
                  "initiatives", "initiative", "strategy", "strategies", "use", "uses", "usage", "programs", "program", "investments",
                  "investment", "features", "feature", "powered", "driven", "enabled", "based", "new", "advanced", "technological",
                  "capabilities", "efforts", "effort", "adoption", "innovation", "innovations", "opportunities", "techniques",
                  "approaches", "methods", "models", "model", "algorithms", "algorithm", "automation", "the", "of", "our", "in",
                  "across", "business", "operations", "various", "other", "related", "a", "an", "unspecified", "technologies,"}
GENERIC_KEEP = {"models", "model", "algorithms", "algorithm", "automation"}   # solos sí dicen algo: se evalúan por regex


def is_generic_object(value: str | None) -> bool:
    """'ai capabilities', 'generative ai tools', 'ai and machine learning' no
    nombran ningún objeto: son la palabra IA con un sustantivo vacío."""
    if value is None:
        return True
    tokens = [w.strip(",.;:()") for w in str(value).lower().replace("/", " ").replace("-", " ").split()]
    tokens = [w for w in tokens if w]
    if not tokens:
        return True
    if any(w in GENERIC_KEEP for w in tokens) and not all(w in GENERIC_TOKENS for w in tokens if w not in GENERIC_KEEP):
        return False
    if any(w in GENERIC_KEEP for w in tokens):
        return False
    return all(w in GENERIC_TOKENS for w in tokens)


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
        inst_docs = con.execute(f"""
            WITH docs AS (
                SELECT accession_number, ticker, form_type AS form FROM filing_manifest WHERE country_code='us'
                UNION ALL SELECT accession_number, ticker, '10-Q' FROM filing_manifest_10q WHERE country_code='us'
                UNION ALL SELECT document_id, ticker, 'Earnings call' FROM read_parquet('{CALLS_MANIFEST}')
            )
            SELECT DISTINCT g.text_hash, g.accession_number, d.ticker, CASE WHEN d.form = 'Earnings call' THEN 'call' ELSE 'filing' END AS channel
            FROM (SELECT DISTINCT text_hash, accession_number FROM gold_ai_frames WHERE country_code='us' AND has_frame) g
            JOIN acts USING (text_hash) JOIN docs d USING (accession_number)
            WHERE d.ticker IS NOT NULL
        """).df()
        docs = document_table(con)[["accession_number", "fecha", "fy", "n_paragraphs", "channel"]]
    finally:
        con.close()
    inst_docs["text_hash"] = inst_docs["text_hash"].astype("uint64")
    inst = inst_docs.drop(columns="accession_number").drop_duplicates()
    a = acts.merge(inst, on="text_hash", how="inner")
    # una actividad única por empresa; si el mismo texto aparece en call y filing, se anota el canal 'filing'
    a["channel_rank"] = (a["channel"] == "call").astype(int)
    a = a.sort_values("channel_rank").drop_duplicates(["ticker", "text_hash", "activity_index"]).drop(columns="channel_rank")
    a["function_family"] = a["function"].map(lambda v: family(v, FUNCTION_FAMILIES, "unspecified"))
    # proveedores: lista por actividad. `provider_families` es la lista de familias;
    # `provider_family` resume la actividad: la primera familia externa nombrada, si no
    # 'proprietary' si lo declara, si no 'unspecified'. Los conteos por proveedor usan la lista.
    a["providers_or_models"] = a["providers_or_models"].map(lambda v: [str(x) for x in (list(v) if v is not None else [])])
    a["provider_or_model"] = a["providers_or_models"].map(lambda v: ", ".join(v) if v else "unspecified")
    a["provider_families"] = a["providers_or_models"].map(lambda v: sorted({family(x, PROVIDER_FAMILIES, "unspecified") for x in v}))
    def _summary(fams):
        ext = [f for f in fams if f not in ("proprietary", "unspecified", "third_party_unnamed")]
        return ext[0] if ext else ("third_party_unnamed" if "third_party_unnamed" in fams else ("proprietary" if "proprietary" in fams else "unspecified"))
    a["provider_family"] = a["provider_families"].map(_summary)
    a["stage_rank"] = a["stage"].map(STAGE_RANK).fillna(0).astype(int)
    a["object_family"] = a["object"].map(lambda v: "AI, unspecified object" if is_generic_object(v) else family(v, OBJECT_FAMILIES, "AI, unspecified object"))
    # lo que no cae en ninguna familia pero trae producto o proceso con nombre es un objeto concreto con marca
    named = (a["object_family"] == "other") & (a["evidence_strength"] == "named_product_or_process")
    a.loc[named, "object_family"] = "named product or platform"
    a["activity"] = a["action"] + " · " + a["object_family"]
    a["activity_function"] = a["activity"] + " · " + a["function_family"]
    a.attrs["n_texts"] = int(n_texts)
    # instancias por documento (una fila por actividad × documento), para paneles por año y por canal
    inst_docs = inst_docs.merge(docs.drop(columns="channel"), on="accession_number", how="inner")
    inst_docs["year"] = inst_docs["fecha"].dt.year
    a.attrs["instances"] = a[["text_hash", "activity_index", "action", "target", "stage", "provider_or_model", "provider_family",
                              "provider_families", "evidence_strength", "function"]].drop_duplicates(["text_hash", "activity_index"]) \
        .merge(inst_docs, on="text_hash", how="inner")
    return a


def flags(a: pd.DataFrame) -> pd.DataFrame:
    """Indicadores por actividad de los 'top behaviours'."""
    f = pd.DataFrame(index=a.index)
    used = a["action"].isin(["deploy", "integrate", "scale"])
    f["customer_facing_deployment"] = used & (a["target"] == "customers")
    f["internal_deployment"] = used & a["target"].isin(["employees", "internal_process"])
    f["developer_tools"] = used & (a["target"] == "developers")
    f["proprietary_ai"] = (a["action"] == "develop") | a["provider_families"].map(lambda v: "proprietary" in v)
    f["third_party_named_provider"] = a["provider_families"].map(lambda v: any(x not in ("proprietary", "unspecified", "third_party_unnamed") for x in v))
    f["infrastructure_investment"] = a["action"] == "invest_infrastructure"
    f["acquisition_or_licensing"] = a["action"] == "buy_or_license"
    f["partnership"] = a["action"] == "partner"
    f["talent_or_training"] = a["action"] == "hire_or_train"
    f["quantified_outcome"] = (a["action"] == "measure_outcome") | (a["evidence_strength"] == "metric")
    f["governance_or_restriction"] = a["action"].isin(["govern_or_control", "restrict"])
    f["piloting_or_exploring"] = a["action"] == "pilot_or_explore"
    f["named_product_or_process"] = a["evidence_strength"] == "named_product_or_process"
    f["named_function"] = ~a["function"].fillna("unspecified").str.lower().isin(["unspecified", "", "none", "n/a"])
    f["deployed_or_scaled"] = a["stage"].isin(["deployed", "scaled"])
    return f


GROUNDING = ["named_function", "deployed_or_scaled", "named_product_or_process", "quantified_outcome", "third_party_named_provider"]
ACTIVITY_FAMILIES = ["customer_facing_deployment", "internal_deployment", "proprietary_ai", "third_party_named_provider",
                     "infrastructure_investment", "quantified_outcome", "talent_or_training", "governance_or_restriction",
                     "piloting_or_exploring"]


def concreteness(prof: pd.DataFrame) -> pd.Series:
    """Concreción conductual: media de cinco proporciones de las actividades de la
    empresa (función declarada, desplegada o escalada, producto o proceso con
    nombre, resultado cuantificado, proveedor nombrado). Sólo interpretación y
    robustez; no reemplaza el eje de conducta de `03`."""
    return prof[[f"share_{c}" for c in GROUNDING]].mean(axis=1)


def yearly_and_channel_panels(inst: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1) empresa-año de presentación, sólo filings: actividades de cada familia
    por 1.000 párrafos, para el bloque de actividades de `05`.
    (2) empresa × ejercicio fiscal × canal: conteos por familia, para la brecha
    de actividades de `06`."""
    inst = inst.copy()
    fl = flags(inst)
    for c in ACTIVITY_FAMILIES + ["named_product_or_process", "named_function", "deployed_or_scaled"]:
        inst[c] = fl[c].astype(float)
    inst["n_activities"] = 1.0
    cols = ACTIVITY_FAMILIES + ["named_product_or_process", "named_function", "deployed_or_scaled", "n_activities"]
    # (1) por año de presentación, filings; el denominador es el de firm_year_master_v2
    fil = inst[inst["channel"] == "filing"].groupby(["ticker", "year"])[cols].sum().reset_index()
    master = pd.read_parquet(OUT_DIR / "firm_year_master_v2.parquet")[["ticker", "year", "n_paragraphs"]]
    fy_panel = master.merge(fil, on=["ticker", "year"], how="left").fillna({c: 0.0 for c in cols})
    for c in cols:
        fy_panel[f"{c}_per_1k"] = 1000.0 * fy_panel[c] / fy_panel["n_paragraphs"]
    # (2) por ejercicio fiscal y canal
    ch = inst.groupby(["ticker", "fy", "channel"])[cols].sum().reset_index()
    return fy_panel, ch


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
    prof["concrecion_conductual"] = concreteness(prof)
    return prof


def main() -> None:
    a = load()
    instances = a.attrs.pop("instances")
    seg = pd.read_parquet(OUT_DIR / "firm_segments.parquet")[["ticker", "segmento"]]
    prof = firm_profiles(a, seg)
    prof.to_parquet(OUT_DIR / "firm_activity_profiles.parquet", index=False)
    a.drop(columns=["stage_rank"]).to_parquet(OUT_DIR / "firm_activities.parquet", index=False)
    fy_panel, ch = yearly_and_channel_panels(instances)
    fy_panel.to_parquet(OUT_DIR / "firm_year_activities.parquet", index=False)
    ch.to_parquet(OUT_DIR / "channel_activity_cells.parquet", index=False)
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
    multi = float(a["provider_families"].map(lambda v: len(set(v) - {"proprietary", "unspecified", "third_party_unnamed"}) >= 2).mean())
    print(f"  actividades con ≥2 proveedores externos nombrados: {100 * multi:.1f}%")
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
                            "examples": g["object"].apply(lambda s: ", ".join(s.value_counts().index[:4]))})
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
    prov_firms["examples"] = named.groupby("provider_family")["providers_or_models"].apply(lambda s: ", ".join(s.value_counts().index[:5]))
    print(prov_firms[["pct_firms", "firms", "examples"]].to_string())
    print("  empresas que nombran algún proveedor externo:", named["ticker"].nunique(), "de", n_firms)

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
    aseg = a.merge(seg, on="ticker")
    seg_func = pd.crosstab(aseg["segmento"], aseg["function_family"], normalize="index").reindex(order[:3]) * 100

    print("\nACTIVIDADES CONCRETAS POR SEGMENTO — top 12 acción · objeto, % de empresas del segmento")
    seg_top = {}
    for s in order[:3]:
        sub = aseg[(aseg["segmento"] == s) & (aseg["object_family"] != "AI, unspecified object")]
        tbl = firm_share(sub, "activity", int(by_seg.loc[s, "n_firms"]), 12)
        seg_top[s] = json.loads(tbl.to_json(orient="index"))
        print(f"  [{SEGMENT_LABELS[s]}]")
        for k, r in tbl.iterrows():
            print(f"    {r.pct_firms:5.1f}%  {k:45s} e.g. {r.examples}")
    seg_actf = {}
    print("\nACCIÓN · OBJETO · FUNCIÓN POR SEGMENTO — top 10 con función declarada, % de empresas del segmento")
    for s in order[:3]:
        sub = aseg[(aseg["segmento"] == s) & (aseg["object_family"] != "AI, unspecified object") & ~aseg["function_family"].isin(["unspecified", "other"])]
        tbl = firm_share(sub, "activity_function", int(by_seg.loc[s, "n_firms"]), 10)
        seg_actf[s] = json.loads(tbl.to_json(orient="index"))
        print(f"  [{SEGMENT_LABELS[s]}]")
        for k, r in tbl.iterrows():
            print(f"    {r.pct_firms:5.1f}%  {k}")

    print("\nFICHAS — inventario de actividades concretas por empresa ejemplar")
    cards = {}
    for s, tickers in EXEMPLARS.items():
        print(f"  [{SEGMENT_LABELS[s]}]")
        for t_ in tickers:
            sub = a[a.ticker == t_]
            if sub.empty:
                continue
            g = sub.groupby(["action", "object_family"])
            inv = pd.DataFrame({"n": g.size(),
                                "functions": g["function_family"].apply(lambda x: ", ".join(v for v in x.value_counts().index[:2] if v not in ("unspecified", "other"))),
                                "targets": g["target"].apply(lambda x: ", ".join(v for v in x.value_counts().index[:2] if v != "unspecified")),
                                "objects": g["object"].apply(lambda x: ", ".join(x.value_counts().index[:3])),
                                "providers": g["providers_or_models"].apply(lambda x: ", ".join(v for v in pd.Series([p for lst in x for p in lst], dtype=object).value_counts().index[:3] if v.lower() not in ("unspecified", "proprietary"))),
                                "stage": g["stage_rank"].max().map({v: k for k, v in STAGE_RANK.items()}),
                                "named_or_metric": g["evidence_strength"].apply(lambda x: float(x.isin(["named_product_or_process", "metric", "vendor"]).mean()))
                                }).sort_values("n", ascending=False).head(8).reset_index()
            lines = [f"{r.action} · {r.object_family} ({r.n}): {r.objects}" + (f" | for {r.functions}" if r.functions else "")
                     + (f" | {r.targets}" if r.targets else "") + f" | {r.stage}" + (f" | {r.providers}" if r.providers else "")
                     + f" | concrete {r.named_or_metric:.0%}" for r in inv.itertuples()]
            cards[t_] = {"segment": SEGMENT_LABELS[s], "n_activities": int(len(sub)), "lines": lines}
            print(f"    {t_} ({len(sub)} activities)")
            for ln in lines:
                print(f"      - {ln}")

    payload = {"n_activities": int(len(a)), "n_texts": a.attrs["n_texts"], "n_firms": n_firms,
               "firms_with_activity": int((prof.n_activities > 0).sum()), "share_from_calls": float((a.channel == "call").mean()),
               "distributions": dist, "function_families": fam.to_dict(), "provider_families": prov.to_dict(),
               "top_behaviours": top.to_dict(), "top_behaviours_filings_only": top_f.to_dict(), "max_stage": stage_share.to_dict(),
               "by_segment": json.loads(by_seg.to_json(orient="index")), "stage_by_segment": json.loads(seg_stage.round(1).to_json(orient="index")),
               "functions_by_segment": json.loads(seg_func.round(1).to_json(orient="index")), "exemplars": cards,
               "top_concrete_activities": json.loads(top_act.to_json(orient="index")),
               "top_activity_function": json.loads(top_actf.to_json(orient="index")),
               "object_families": json.loads(top_obj.to_json(orient="index")),
               "named_providers": json.loads(prov_firms.to_json(orient="index")), "firms_naming_provider": int(named["ticker"].nunique()),
               "segment_top_activities": seg_top, "segment_top_activity_function": seg_actf}
    (OUT_DIR / "activity_profiles.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\n-> {OUT_DIR}/activity_profiles.json, firm_activity_profiles.parquet, firm_activities.parquet")


if __name__ == "__main__":
    main()
