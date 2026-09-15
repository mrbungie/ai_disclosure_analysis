"""Disclosed AI activities (the concrete behavioural layer): the activity
spine, its extracted fields and its deterministic taxonomy.

Reads the activities extracted by `scripts/enrichment/ai_activities_from_frames.py`
(the company does ACTION on OBJECT for FUNCTION, with stage, target,
provider and evidence strength) from `silver.ai_activities` (one verdict per
text, already restricted to the analysis universe and the deployed
prefilter) and joins each unique text to every document instance carrying it
(the gold document spine) to know which firm and channel it came from. Each
activity is counted ONCE per firm (unique text x activity index), so repeated
boilerplate across filings does not inflate the inventory.

Outputs:
  spines/activity/activity          id, ticker, text_hash, activity_id, fecha
                                    (earliest date of a document of that
                                    ticker carrying the text), accession_number
                                    (the judged text's representative document)
  covariates/activity/extraction    the extracted fields as judged
  covariates/activity/taxonomy      channel (filing when the text appears in
                                    both a filing and a call), function,
                                    provider and object families, activity
                                    labels

`flags()` (the activity-family and grounding indicators) and the family
tables are imported by the document, firm-quarter and call builders and by
scripts/analytics/posture/activity_profiles.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "scripts" / "gold" / "document"))
from build_document import gold_document_table  # noqa: E402

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
EXTERNAL_ROLES = {"provider", "model", "partner"}

GENERIC_TOKENS = {"ai", "artificial", "intelligence", "generative", "gen", "genai", "ml", "machine", "learning", "and", "or", "&",
                  "capabilities", "capability", "technologies", "technology", "tools", "tool", "solutions", "solution", "systems",
                  "system", "applications", "application", "services", "service", "products", "product", "offerings", "offering",
                  "initiatives", "initiative", "strategy", "strategies", "use", "uses", "usage", "programs", "program", "investments",
                  "investment", "features", "feature", "powered", "driven", "enabled", "based", "new", "advanced", "technological",
                  "capabilities", "efforts", "effort", "adoption", "innovation", "innovations", "opportunities", "techniques",
                  "approaches", "methods", "models", "model", "algorithms", "algorithm", "automation", "the", "of", "our", "in",
                  "across", "business", "operations", "various", "other", "related", "a", "an", "unspecified", "technologies,"}
GENERIC_KEEP = {"models", "model", "algorithms", "algorithm", "automation"}


def is_generic_object(value: str | None) -> bool:
    """'ai capabilities', 'generative ai tools', 'ai and machine learning'
    name no object at all: it's the word AI with an empty noun."""
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


EXTRACTION = ["country_code", "form", "item_key", "paragraph_index", "frame_id", "has_activity", "action", "object",
              "function", "target", "stage", "source", "entities", "metrics", "evidence_type", "sentence_ids", "firm",
              "judge_model", "prompt_version", "session_id", "classified_at"]
TAXONOMY = ["channel", "function_family", "providers_or_models", "own_brands", "is_own_ai", "provider_or_model",
            "provider_families", "provider_family", "object_family", "activity", "activity_function"]


def instance_documents(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (judged text, document carrying it) with the document's
    ticker, channel, date and fiscal year (gold document spine)."""
    if raw is None:
        raw = L.scan("silver.ai_activities").select("text_hash", "accession_number").collect().to_pandas()
    docs = gold_document_table()[["accession_number", "fecha", "fy", "n_words", "channel", "ticker"]]
    inst_docs = (raw[["text_hash", "accession_number"]].drop_duplicates()
                 .merge(docs, on="accession_number", how="inner"))
    return inst_docs[inst_docs["ticker"].notna()]


def load() -> pd.DataFrame:
    """One row per (ticker, unique text, activity index), with the taxonomy
    families attached, `fecha` (earliest date of a document of that ticker
    carrying the text) and `a.attrs["n_texts"]`."""
    ACTIVITY_PASSTHROUGH = ["country_code", "form", "accession_number", "item_key", "paragraph_index",
                            "frame_id", "activity_id", "has_activity", "action", "object", "function",
                            "target", "stage", "source", "entities", "metrics", "evidence_type",
                            "sentence_ids", "firm", "judge_model", "prompt_version", "session_id", "classified_at"]
    raw = L.scan("silver.ai_activities").select("text_hash", *ACTIVITY_PASSTHROUGH).collect().to_pandas()
    # one verdict per (text_hash, activity_id) -- session_id/has_activity
    # resolution across LLM reruns already happened in bronze.ai_activities.
    n_texts = raw["text_hash"].nunique()
    inst_docs = instance_documents(raw)

    # one representative instance per (text_hash, activity_id) for the
    # passthrough/provenance columns -- silver broadcasts each judged text to
    # every document instance that carries it, so any one of them is an
    # equally valid document of origin.
    acts = (raw.loc[raw["has_activity"]]
            .drop_duplicates(["text_hash", "activity_id"]).copy())

    inst = inst_docs[["text_hash", "ticker", "channel"]].drop_duplicates()
    a = acts.merge(inst, on="text_hash", how="inner")
    # una actividad única por empresa; si el mismo texto aparece en call y
    # filing, se anota el canal 'filing'
    a["channel_rank"] = (a["channel"] == "call").astype(int)
    a = a.sort_values("channel_rank").drop_duplicates(["ticker", "text_hash", "activity_id"]).drop(columns="channel_rank")
    a["function_family"] = a["function"].map(lambda v: family(v, FUNCTION_FAMILIES, "unspecified"))
    a["entities"] = a["entities"].map(lambda v: [dict(e) for e in (list(v) if v is not None else [])])
    a["source"] = a["source"].fillna("unspecified")
    a["providers_or_models"] = a["entities"].map(lambda es: [e["name"] for e in es if e["role"] in EXTERNAL_ROLES])
    a["own_brands"] = a["entities"].map(lambda es: [e["name"] for e in es if e["role"] == "own_brand"])
    a["is_own_ai"] = a["source"].isin(["own", "mixed"])
    a["provider_or_model"] = a["providers_or_models"].map(lambda v: ", ".join(v) if v else "unspecified")
    a["provider_families"] = a.apply(lambda r: sorted({family(x, PROVIDER_FAMILIES, "unspecified") for x in r["providers_or_models"]} | ({"proprietary"} if r["is_own_ai"] else set())), axis=1)

    def _summary(fams):
        ext = [f for f in fams if f not in ("proprietary", "unspecified", "third_party_unnamed")]
        return ext[0] if ext else ("third_party_unnamed" if "third_party_unnamed" in fams else ("proprietary" if "proprietary" in fams else "unspecified"))

    a["provider_family"] = a["provider_families"].map(_summary)
    a["stage_rank"] = a["stage"].map(STAGE_RANK).fillna(0).astype(int)
    a["object_family"] = a["object"].map(lambda v: "AI, unspecified object" if is_generic_object(v) else family(v, OBJECT_FAMILIES, "AI, unspecified object"))
    named = (a["object_family"] == "other") & (a["evidence_type"] == "named")
    a.loc[named, "object_family"] = "named product or platform"
    a["activity"] = a["action"] + " · " + a["object_family"]
    a["activity_function"] = a["activity"] + " · " + a["function_family"]
    first_seen = inst_docs.groupby(["ticker", "text_hash"])["fecha"].min().rename("fecha").reset_index()
    a = a.merge(first_seen, on=["ticker", "text_hash"], how="left", validate="many_to_one")
    a.attrs["n_texts"] = int(n_texts)
    return a


def flags(a: pd.DataFrame) -> pd.DataFrame:
    """Indicadores por actividad de los 'top behaviours'."""
    f = pd.DataFrame(index=a.index)
    used = a["action"].isin(["deploy", "integrate", "scale"])
    f["customer_facing_deployment"] = used & (a["target"] == "customers")
    f["internal_deployment"] = used & a["target"].isin(["employees", "internal_process"])
    f["developer_tools"] = used & (a["target"] == "developers")
    f["proprietary_ai"] = (a["action"] == "develop") | (a["source"] == "own")
    f["own_brand_named"] = a["own_brands"].map(lambda v: len(v) > 0)
    f["third_party_ai"] = a["source"].isin(["third_party", "mixed", "open_source"])
    f["co_developed_or_acquired"] = a["source"].isin(["co_developed", "acquired"])
    f["named_customer"] = a["entities"].map(lambda es: any(e["role"] == "customer" for e in es))
    f["third_party_named_provider"] = a["provider_families"].map(lambda v: any(x not in ("proprietary", "unspecified", "third_party_unnamed") for x in v))
    f["infrastructure_investment"] = a["action"] == "invest"
    f["acquisition_or_licensing"] = a["action"] == "procure"
    f["partnership"] = a["action"] == "partner"
    f["talent_or_training"] = a["action"] == "staff"
    f["quantified_outcome"] = (a["action"] == "measure") | (a["evidence_type"] == "metric")
    f["governance_or_restriction"] = a["action"].isin(["govern", "restrict"])
    f["piloting_or_exploring"] = a["stage"].isin(["exploring", "piloting"])
    f["named_product_or_process"] = a["evidence_type"] == "named"
    f["named_function"] = ~a["function"].fillna("unspecified").str.lower().isin(["unspecified", "", "none", "n/a"])
    f["deployed_or_scaled"] = a["stage"].isin(["deployed", "scaled"])
    return f


GROUNDING = ["named_function", "deployed_or_scaled", "named_product_or_process", "quantified_outcome", "third_party_named_provider"]
ACTIVITY_FAMILIES = ["customer_facing_deployment", "internal_deployment", "proprietary_ai", "third_party_named_provider",
                     "infrastructure_investment", "quantified_outcome", "talent_or_training", "governance_or_restriction",
                     "piloting_or_exploring"]


def main() -> None:
    a = load()
    a.insert(0, "id", a["ticker"] + "_" + a["text_hash"].astype(str) + "_" + a["activity_id"].astype(str))
    a["fecha"] = a["fecha"].astype("datetime64[ns]")
    spine = L.GOLD_SPINE_COLUMNS["activity"]
    BUILDER = "scripts/gold/activity/build_activity.py"
    L.write_gold("spines", "activity", "activity", a[spine + ["accession_number"]], builder=BUILDER,
                 inputs=[L.path("silver.ai_activities")], extra={"n_texts_judged": a.attrs["n_texts"]})
    L.write_gold("covariates", "activity", "extraction", a[spine + EXTRACTION], builder=BUILDER,
                 inputs=[L.path("silver.ai_activities")])
    L.write_gold("covariates", "activity", "taxonomy", a[spine + TAXONOMY], builder=BUILDER)
    print(f"actividades: {len(a):,} únicas por empresa, de {a.attrs['n_texts']:,} textos únicos procesados | "
          f"{(a.channel == 'call').mean():.0%} provienen sólo de calls")


if __name__ == "__main__":
    main()
