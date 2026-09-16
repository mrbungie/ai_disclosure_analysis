"""Builds `data.json` for the atomic human validation UI (`apps/validator/index.html`).

Organizes validation across three main SCOPES:
  1. prefilter   (AI mention & substantive corporate disclosure)
  2. frames      (AI frames: existence, temporal, promotional, specificities, evidence)
  3. activities  (Atomic activities: action, object, target, stage, AI source, entities, evidence)

All facts extracted by models are flattened into atomic verification items (1 question at a time).
Each verification is answered strictly with:
  [Y] Yes
  [N] No
  [U] Uncertain

Usage:
    # Flatten from existing sample (fast):
    uv run --frozen --no-sync python apps/validator/build_sample.py --from-existing apps/validator/data.json

    # Sample from scratch across bronze and silver:
    uv run --frozen --no-sync python apps/validator/build_sample.py --frames 300 --prefilter 300 --activities 120

    # Serve UI:
    cd apps/validator && python -m http.server 8765
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))
import layers as L  # noqa: E402

PRED_DIR = L.ADDITIVE_SOURCES["prefilter_predictions_unique"]["path"]
CALLS_MANIFEST = REPO_ROOT / "data" / "interim" / "manifests" / "filing_manifest_earnings_calls.parquet"
ACTIVITIES_PATH = REPO_ROOT / "data" / "processed" / "clusters" / "firm_activities.parquet"
OUT = Path(__file__).resolve().parent / "data.json"
FORMS = ["10-K", "10-Q", "DEF 14A", "8-K", "Earnings call"]
PARAGRAPH_KEY = ["country_code", "form", "accession_number", "item_key", "paragraph_index"]

FRAME_SPECIFICITY = {
    "business_process": "process",
    "product_or_system": "product",
    "vendor_or_partner": "vendor",
    "quantified_metric": "metric",
    "date_or_timeline": "timeline",
}

ACTION_EN = {
    "deploy": "deploys / uses",
    "develop": "develops",
    "integrate": "integrates",
    "buy_or_license": "purchases or licenses",
    "partner": "partners with",
    "invest_infrastructure": "invests in infrastructure for",
    "hire_or_train": "hires or trains for",
    "pilot_or_explore": "pilots or explores",
    "scale": "scales",
    "measure_outcome": "reports outcomes for",
    "govern_or_control": "governs or controls",
    "restrict": "restricts",
    "unspecified": "involves",
}

TARGET_EN = {
    "employees": "employees",
    "customers": "customers",
    "developers": "developers",
    "internal_process": "internal processes",
    "partners_or_suppliers": "partners or suppliers",
    "unspecified": "unspecified",
}

STAGE_EN = {
    "exploring": "exploration",
    "piloting": "pilot / trial",
    "deployed": "deployed / in production",
    "scaled": "scaled / mature",
    "unspecified": "unspecified",
}

SOURCE_EN = {
    "own": "own (in-house / proprietary)",
    "third_party": "third-party (external vendor)",
    "co_developed": "co-developed",
    "acquired": "acquired",
    "open_source": "open source",
    "mixed": "mixed (own + external models)",
    "unspecified": "unspecified in text",
}

ROLE_EN = {
    "own_product_or_brand": "own product / brand",
    "external_provider": "external vendor / provider",
    "external_model": "external model",
    "partner": "partner / ally",
    "acquired_company": "acquired company",
    "customer": "customer",
    "distribution_channel": "distribution channel",
    "competitor_or_reference": "competitor / reference",
}

TEMPORAL_EN = {
    "realized": "implemented / in production (realized)",
    "planned": "planned / future initiative (planned)",
    "expected": "expected / projected (expected)",
    "hypothetical": "hypothetical / conditional (hypothetical)",
    "unspecified": "unspecified",
}

SPEC_LABELS_EN = {
    "business_process": (
        "Business Process",
        "Does the text describe a specific business process or workflow where AI is applied?",
    ),
    "product_or_system": (
        "Product or System",
        "Does the text name a specific AI product, software, or system?",
    ),
    "vendor_or_partner": (
        "Vendor or Partner",
        "Does the text name an external AI vendor, provider, or technology partner?",
    ),
    "quantified_metric": (
        "Quantified Metric",
        "Does the text report a numeric metric or measurable figure associated with AI?",
    ),
    "date_or_timeline": (
        "Timeline / Date",
        "Does the text specify a concrete date, timeline, or milestone for the AI initiative?",
    ),
}


def latest_predictions() -> Path:
    runs = sorted(PRED_DIR.glob("prefilter_predictions__run=*.parquet"))
    if not runs:
        raise SystemExit(f"no prefilter predictions in {PRED_DIR}")
    return runs[-1]


def _seeded_order(values: pl.Series, salt: str) -> pl.Series:
    return pl.Series(
        [
            int.from_bytes(
                hashlib.blake2b(f"{v}|{salt}".encode(), digest_size=8).digest(),
                "big",
                signed=False,
            )
            for v in values.to_list()
        ],
        dtype=pl.UInt64,
    )


def register_firm_lookup() -> dict[str, dict]:
    fm = (
        L.scan("bronze.filing_manifest")
        .filter(pl.col("country_code") == "us")
        .select("accession_number", "ticker")
    )
    fmq = (
        L.scan("bronze.filing_manifest_10q")
        .filter(pl.col("country_code") == "us")
        .select("accession_number", "ticker")
    )
    calls = pl.scan_parquet(CALLS_MANIFEST).select(
        pl.col("document_id").alias("accession_number"), "ticker"
    )
    docs = pl.concat([fm, fmq, calls], how="vertical_relaxed").unique()
    names = (
        L.scan("bronze.firm_universe")
        .filter(pl.col("country_code") == "us")
        .group_by("ticker")
        .agg(pl.col("company_name").drop_nulls().first())
    )
    doc_firm = (
        docs.join(names, on="ticker", how="left")
        .unique(subset=["accession_number"], keep="first")
        .collect(engine="streaming")
    )
    return {
        r["accession_number"]: {"ticker": r["ticker"], "company": r["company_name"]}
        for r in doc_firm.iter_rows(named=True)
    }


def firm_of(lookup: dict, accession_number: str | None) -> dict:
    return lookup.get(accession_number, {"ticker": None, "company": None})


def _sentences_by_paragraph(paragraph_keys: pl.DataFrame) -> pl.DataFrame:
    return (
        L.scan("bronze.sentences")
        .join(paragraph_keys.lazy().select(*PARAGRAPH_KEY), on=PARAGRAPH_KEY, how="semi")
        .group_by(PARAGRAPH_KEY)
        .agg(
            pl.struct(idx=pl.col("sentence_index"), text=pl.col("sentence_text"))
            .sort_by(pl.col("sentence_index"))
            .alias("sentences")
        )
        .collect(engine="streaming")
    )


def sample_frames(n: int, seed: int, firm_lookup: dict) -> list[dict]:
    per_form = max(1, n // len(FORMS))
    frames_all = (
        L.scan("silver.ai_frames")
        .filter((pl.col("country_code") == "us") & pl.col("has_frame"))
        .select(
            *PARAGRAPH_KEY,
            "text_hash",
            "frame_id",
            "subject",
            "ai_type",
            "temporal",
            "concepts",
            "specificity",
            "rhetoric",
            "sentence_ids",
        )
        .collect(engine="streaming")
    )
    paras = frames_all.select(*PARAGRAPH_KEY, "text_hash").unique()
    paras = paras.with_columns(
        pl.Series("_rnd", _seeded_order(paras["text_hash"], f"build_sample.frames|{seed}"))
    )
    paras = (
        paras.with_columns(pl.col("_rnd").rank("ordinal").over("form").alias("_rn"))
        .filter(pl.col("_rn") <= per_form)
    )
    paras = paras.join(
        _sentences_by_paragraph(paras.select(*PARAGRAPH_KEY)), on=PARAGRAPH_KEY, how="left"
    ).sort(PARAGRAPH_KEY)
    frames_all = frames_all.join(paras.select(*PARAGRAPH_KEY), on=PARAGRAPH_KEY, how="semi")

    items = []
    for r in paras.iter_rows(named=True):
        fr = frames_all.filter(
            (pl.col("accession_number") == r["accession_number"])
            & (pl.col("item_key") == r["item_key"])
            & (pl.col("paragraph_index") == r["paragraph_index"])
        ).sort("frame_id")
        sentences = [{"idx": s["idx"], "text": s["text"]} for s in (r["sentences"] or [])]
        frame_items = []
        for f in fr.iter_rows(named=True):
            spec_tags, rhet_tags = set(f["specificity"] or []), set(f["rhetoric"] or [])
            frame_items.append({
                "frame_index": int(f["frame_id"]),
                "subject": f["subject"],
                "ai_type": f["ai_type"],
                "temporal": f["temporal"],
                "domain": "",
                "concepts": [str(c) for c in (f["concepts"] or [])],
                "specificity": {k: (tag in spec_tags) for k, tag in FRAME_SPECIFICITY.items()},
                "promotional": "promotional" in rhet_tags,
                "strategic": "strategic" in rhet_tags,
                "evidence": [int(x) for x in (f["sentence_ids"] or [])],
            })
        items.append({
            "id": f"F:{r['form']}:{r['accession_number']}:{r['item_key']}:{int(r['paragraph_index'])}",
            "form": r["form"],
            "accession_number": r["accession_number"],
            "text_hash": str(r["text_hash"]),
            **firm_of(firm_lookup, r["accession_number"]),
            "sentences": sentences,
            "frames": frame_items,
        })
    random.Random(seed).shuffle(items)
    return items


def sample_activities(n: int, seed: int, firm_lookup: dict) -> list[dict]:
    per = max(1, n // (len(FORMS) * 3))
    a = (
        L.scan("bronze.ai_activities")
        .filter((pl.col("country_code") == "us") & pl.col("has_activity"))
        .select(*PARAGRAPH_KEY, "text_hash", "activity_id")
        .unique()
    )
    paras = (
        a.group_by(*PARAGRAPH_KEY, "text_hash")
        .agg(pl.len().alias("n_act"))
        .with_columns(
            pl.when(pl.col("n_act") == 1)
            .then(pl.lit("1"))
            .when(pl.col("n_act") == 2)
            .then(pl.lit("2"))
            .otherwise(pl.lit("3+"))
            .alias("estrato")
        )
        .collect(engine="streaming")
    )
    paras = paras.with_columns(
        pl.Series("_rnd", _seeded_order(paras["text_hash"], f"build_sample.activities|{seed}"))
    )
    paras = (
        paras.with_columns(pl.col("_rnd").rank("ordinal").over(["form", "estrato"]).alias("_rn"))
        .filter(pl.col("_rn") <= per)
    )
    paras = paras.join(
        _sentences_by_paragraph(paras.select(*PARAGRAPH_KEY)), on=PARAGRAPH_KEY, how="left"
    ).sort(PARAGRAPH_KEY)

    acts = (
        L.scan("bronze.ai_activities")
        .filter((pl.col("country_code") == "us") & pl.col("has_activity"))
        .join(paras.lazy().select("text_hash"), on="text_hash", how="semi")
        .select(
            "text_hash",
            "activity_id",
            "action",
            "object",
            "function",
            "target",
            "stage",
            "source",
            "entities",
            "evidence_type",
            "sentence_ids",
        )
        .unique()
        .sort(["text_hash", "activity_id"])
        .collect(engine="streaming")
    )

    items = []
    for r in paras.iter_rows(named=True):
        sents = [{"idx": s["idx"], "text": s["text"]} for s in (r["sentences"] or [])]
        activities = []
        for x in acts.filter(pl.col("text_hash") == r["text_hash"]).iter_rows(named=True):
            ev_pos = [int(v) for v in (x["sentence_ids"] or [])]
            activities.append({
                "activity_index": int(x["activity_id"]),
                "action": x["action"],
                "object": x["object"],
                "function": x["function"],
                "target": x["target"],
                "stage": x["stage"],
                "ai_source": x["source"],
                "entities": [{"name": str(e["name"]), "role": str(e["role"])} for e in (x["entities"] or [])],
                "evidence_strength": x["evidence_type"],
                "evidence": [sents[i]["idx"] for i in ev_pos if 0 <= i < len(sents)],
            })
        items.append({
            "id": f"A:{r['text_hash']}",
            "form": r["form"],
            "accession_number": r["accession_number"],
            **firm_of(firm_lookup, r["accession_number"]),
            "text_hash": str(r["text_hash"]),
            "sentences": sents,
            "activities": activities,
        })
    random.Random(seed).shuffle(items)
    return items


def sample_prefilter(n: int, seed: int, firm_lookup: dict) -> list[dict]:
    pred = latest_predictions()
    per = max(1, n // (len(FORMS) * 3))
    ai_terms = r"(^|[^a-z])(ai|artificial intelligence|machine learning|generative|llm|chatgpt|copilot)([^a-z]|$)"
    pred_us = (
        pl.scan_parquet(pred)
        .filter(pl.col("country_code") == "us")
        .select(
            "text_hash",
            "form",
            "predicted_proba",
            "is_ai_prefiltered",
            "threshold",
            "named_entity_match",
        )
        .collect()
    )

    matches = []
    reader = pq.ParquetFile(L.path("bronze.unique_paragraphs"))
    for batch in reader.iter_batches(batch_size=100_000, columns=["text_hash", "paragraph_text", "is_scorable"]):
        up = pl.from_arrow(batch).filter(
            pl.col("is_scorable") & pl.col("paragraph_text").str.len_chars().is_between(80, 2500)
        )
        if up.height == 0:
            continue
        joined = up.join(pred_us, on="text_hash", how="inner").filter(
            pl.col("is_ai_prefiltered")
            | (pl.col("predicted_proba") >= 0.05)
            | pl.col("paragraph_text").str.to_lowercase().str.contains(ai_terms)
        )
        if joined.height:
            matches.append(
                joined.select(
                    "text_hash",
                    "form",
                    "predicted_proba",
                    "is_ai_prefiltered",
                    "threshold",
                    "named_entity_match",
                    "paragraph_text",
                )
            )
    empty_schema = {**pred_us.schema, "paragraph_text": pl.String}
    p = (pl.concat(matches) if matches else pl.DataFrame(schema=empty_schema)).with_columns(
        pl.when(pl.col("is_ai_prefiltered"))
        .then(pl.lit("positive"))
        .when(pl.col("predicted_proba") >= 0.05)
        .then(pl.lit("gray_zone"))
        .otherwise(pl.lit("negative"))
        .alias("estrato")
    )
    p = p.with_columns(
        pl.Series("_rnd", _seeded_order(p["text_hash"], f"build_sample.prefilter|{seed}"))
    )
    p = (
        p.with_columns(pl.col("_rnd").rank("ordinal").over(["form", "estrato"]).alias("_rn"))
        .filter(pl.col("_rn") <= per)
        .sort("text_hash")
    )

    acc = (
        L.scan("bronze.paragraphs")
        .filter(pl.col("country_code") == "us")
        .join(p.lazy().select("text_hash"), on="text_hash", how="semi")
        .group_by("text_hash")
        .agg(pl.col("accession_number").first())
        .collect(engine="streaming")
    )
    acc_by_hash = {r["text_hash"]: r["accession_number"] for r in acc.iter_rows(named=True)}

    items = []
    for r in p.iter_rows(named=True):
        items.append({
            "id": f"P:{r['text_hash']}",
            "form": r["form"],
            "text_hash": str(r["text_hash"]),
            "estrato": r["estrato"],
            **firm_of(firm_lookup, acc_by_hash.get(r["text_hash"])),
            "text": r["paragraph_text"],
            "proba": round(float(r["predicted_proba"]), 3),
            "prefilter_says_ai": bool(r["is_ai_prefiltered"]),
            "threshold": float(r["threshold"]),
            "named_entity_match": bool(r["named_entity_match"]),
        })
    random.Random(seed).shuffle(items)
    return items


def flatten_all(raw_data: dict[str, Any]) -> dict[str, list[dict]]:
    """Flattens raw document/paragraph data into atomic single-fact verifications (Y/N/U)."""
    flattened: dict[str, list[dict]] = {"prefilter": [], "frames": [], "activities": []}

    # ---- 1. PREFILTER ----
    for it in raw_data.get("prefilter", []):
        pid = it["id"]
        base_meta = {
            "paragraph_id": pid,
            "form": it.get("form"),
            "ticker": it.get("ticker"),
            "company": it.get("company"),
            "accession_number": it.get("accession_number"),
            "text_hash": it.get("text_hash"),
            "sentences": [{"idx": 0, "text": it["text"]}],
            "estrato": it.get("estrato"),
            "proba": it.get("proba"),
            "threshold": it.get("threshold"),
        }
        pred_str = "AI" if it.get("prefilter_says_ai") else "Non-AI"
        flattened["prefilter"].append({
            "id": f"{pid}:mentions_ai",
            "ambito": "prefilter",
            "category": "mentions_ai",
            "category_label": "AI Mention",
            **base_meta,
            "highlight_sentences": [0],
            "title": "Artificial Intelligence (AI) Mention",
            "question": "Does this paragraph mention or discuss Artificial Intelligence, Machine Learning, or related technologies?",
            "model_claim": f"Model prediction: {pred_str} (p = {it.get('proba', 0):.3f}, threshold = {it.get('threshold', 0):.3f})",
            "model_value": bool(it.get("prefilter_says_ai")),
            "context_hint": "Check whether the text contains any explicit reference to AI, ML, machine learning, algorithmic models, or generative tools.",
        })
        if it.get("prefilter_says_ai") or it.get("proba", 0) >= 0.05:
            flattened["prefilter"].append({
                "id": f"{pid}:substantive",
                "ambito": "prefilter",
                "category": "substantive",
                "category_label": "Substantive Disclosure",
                **base_meta,
                "highlight_sentences": [0],
                "title": "Substantive Firm Disclosure",
                "question": "Is this a substantive disclosure about the company itself (its products, operations, strategy, or risks) rather than generic boilerplate or incidental mention?",
                "model_claim": "Corporate relevance evaluation",
                "model_value": True,
                "context_hint": "Mark 'No' if it is solely a standard legal disclaimer, general risk factor boilerplate, or tangential mention of third parties without firm operational impact.",
            })

    # ---- 2. FRAMES ----
    for it in raw_data.get("frames", []):
        pid = it["id"]
        sents = it.get("sentences", [])
        base_meta = {
            "paragraph_id": pid,
            "form": it.get("form"),
            "ticker": it.get("ticker"),
            "company": it.get("company"),
            "accession_number": it.get("accession_number"),
            "text_hash": it.get("text_hash"),
            "sentences": sents,
        }
        frames = it.get("frames", [])
        if not frames:
            flattened["frames"].append({
                "id": f"{pid}:missing_frame",
                "ambito": "frames",
                "category": "missing_frame",
                "category_label": "Missing Frame",
                **base_meta,
                "highlight_sentences": [],
                "title": "Paragraph with Zero Frames",
                "question": "The model extracted 0 AI frames from this paragraph. Should at least one AI frame have been extracted?",
                "model_claim": "Model determined 0 frames in this paragraph",
                "model_value": False,
                "context_hint": "Answer 'Yes' if the paragraph makes a clear claim regarding corporate AI adoption, development, or strategic impact that the model missed.",
            })
        for f in frames:
            fidx = f.get("frame_index", 0)
            ev = f.get("evidence", [])
            subj = f.get("subject", "the firm")
            ai_t = f.get("ai_type", "unspecified")

            # Frame Existence
            flattened["frames"].append({
                "id": f"{pid}:f{fidx}:existence",
                "ambito": "frames",
                "category": "existence",
                "category_label": "Frame Existence",
                **base_meta,
                "highlight_sentences": ev,
                "title": f"Frame #{fidx}: Existence ({subj} · {ai_t})",
                "question": f"Does the text support that {subj} has an AI initiative or relationship of type '{ai_t}'?",
                "model_claim": f"Subject: {subj} | AI Type: {ai_t} | Domain: {f.get('domain') or '—'}",
                "model_value": True,
                "context_hint": "Verify whether the frame describes a genuine disclosure in the text rather than a model hallucination.",
            })

            # Temporal
            temp = f.get("temporal", "unspecified")
            temp_en = TEMPORAL_EN.get(temp, temp)
            flattened["frames"].append({
                "id": f"{pid}:f{fidx}:temporal",
                "ambito": "frames",
                "category": "temporal",
                "category_label": "Temporal Dimension",
                **base_meta,
                "highlight_sentences": ev,
                "title": f"Frame #{fidx}: Temporal State ({temp})",
                "question": f"Does the AI initiative correspond to a '{temp_en}' state at the time of reporting?",
                "model_claim": f"Temporal classification: {temp}",
                "model_value": temp,
                "context_hint": "Realized = already deployed / in production; Planned = future goals or plans; Expected = forward-looking projections; Hypothetical = conditional.",
            })

            # Promotional
            promo = bool(f.get("promotional"))
            flattened["frames"].append({
                "id": f"{pid}:f{fidx}:promotional",
                "ambito": "frames",
                "category": "promotional",
                "category_label": "Promotional Rhetoric",
                **base_meta,
                "highlight_sentences": ev,
                "title": f"Frame #{fidx}: Promotional Rhetoric ({'Yes' if promo else 'No'})",
                "question": "Is the AI disclosure predominantly promotional or exaggerated (marketing hype / buzzwords without operational substance)?",
                "model_claim": f"Model classified promotional = {'Yes' if promo else 'No'}",
                "model_value": promo,
                "context_hint": "Answer 'Yes' if the language is self-congratulatory, grandiose, or vague without concrete factual backing.",
            })

            # Specificity tags
            for spec_k, spec_val in (f.get("specificity") or {}).items():
                if spec_val and spec_k in SPEC_LABELS_EN:
                    s_label, s_q = SPEC_LABELS_EN[spec_k]
                    flattened["frames"].append({
                        "id": f"{pid}:f{fidx}:spec_{spec_k}",
                        "ambito": "frames",
                        "category": "specificity",
                        "category_label": f"Specificity: {s_label}",
                        **base_meta,
                        "highlight_sentences": ev,
                        "title": f"Frame #{fidx}: Specificity ({s_label})",
                        "question": s_q,
                        "model_claim": f"Identified specificity: {spec_k} = True",
                        "model_value": True,
                        "context_hint": f"Verify whether the highlighted text specifically includes {s_label.lower()}.",
                    })

            # Evidence
            if ev:
                flattened["frames"].append({
                    "id": f"{pid}:f{fidx}:evidence",
                    "ambito": "frames",
                    "category": "evidence",
                    "category_label": "Textual Evidence",
                    **base_meta,
                    "highlight_sentences": ev,
                    "title": f"Frame #{fidx}: Evidence Validity [{', '.join(str(x) for x in ev)}]",
                    "question": f"Do the highlighted sentence(s) [{', '.join(str(x) for x in ev)}] contain the direct textual evidence supporting this frame?",
                    "model_claim": f"Evidence sentence indices: {ev}",
                    "model_value": ev,
                    "context_hint": "Verify whether the selected sentence(s) are sufficient to justify the claim, or if key evidence is missing.",
                })

    # ---- 3. ACTIVITIES ----
    for it in raw_data.get("activities", []):
        pid = it["id"]
        sents = it.get("sentences", [])
        base_meta = {
            "paragraph_id": pid,
            "form": it.get("form"),
            "ticker": it.get("ticker"),
            "company": it.get("company"),
            "accession_number": it.get("accession_number"),
            "text_hash": it.get("text_hash"),
            "sentences": sents,
        }
        acts = it.get("activities", [])
        flattened["activities"].append({
            "id": f"{pid}:missing_activity",
            "ambito": "activities",
            "category": "completeness",
            "category_label": "Activity Completeness",
            **base_meta,
            "highlight_sentences": [],
            "title": "Paragraph Activity Completeness",
            "question": "Are there any other concrete AI activities described in this paragraph that the model omitted?",
            "model_claim": f"Model extracted {len(acts)} activity/activities in this paragraph",
            "model_value": False,
            "context_hint": "Answer 'Yes' if the paragraph describes another distinct AI application, deployment, or development not listed among the extracted activities.",
        })
        for x in acts:
            aidx = x.get("activity_index", 0)
            ev = x.get("evidence", [])
            act_en = ACTION_EN.get(x.get("action"), x.get("action"))
            tgt_en = TARGET_EN.get(x.get("target"), x.get("target"))
            stg_en = STAGE_EN.get(x.get("stage"), x.get("stage"))
            src_en = SOURCE_EN.get(x.get("ai_source"), x.get("ai_source"))
            fn_str = (
                f" for {x.get('function')}"
                if x.get("function") and x.get("function") != "unspecified"
                else ""
            )
            firm_str = it.get("company") or it.get("ticker") or "the company"

            # Core claim
            flattened["activities"].append({
                "id": f"{pid}:a{aidx}:claim",
                "ambito": "activities",
                "category": "existence",
                "category_label": "Activity Existence",
                **base_meta,
                "highlight_sentences": ev,
                "title": f"Activity #{aidx}: Core Claim",
                "question": f"Does the text state that {firm_str} {act_en} {x.get('object')}{fn_str}?",
                "model_claim": f"{firm_str} {x.get('action')} {x.get('object')} (target: {x.get('target')})",
                "model_value": True,
                "context_hint": "Verify whether the main operational action and object are factual according to the text.",
            })

            # AI Source
            flattened["activities"].append({
                "id": f"{pid}:a{aidx}:source",
                "ambito": "activities",
                "category": "source",
                "category_label": "AI Source",
                **base_meta,
                "highlight_sentences": ev,
                "title": f"Activity #{aidx}: Technology Source ({src_en})",
                "question": f"Is the technology source accurately identified as '{src_en}'?",
                "model_claim": f"ai_source = {x.get('ai_source')}",
                "model_value": x.get("ai_source"),
                "context_hint": "Own = only if the company builds it, it is their proprietary model or product. Third-party = names or refers to an external vendor or tool. If not stated: 'unspecified'.",
            })

            # Action & Object
            flattened["activities"].append({
                "id": f"{pid}:a{aidx}:action_object",
                "ambito": "activities",
                "category": "action_object",
                "category_label": "Action & Object",
                **base_meta,
                "highlight_sentences": ev,
                "title": f"Activity #{aidx}: Action ({x.get('action')}) and Object ({x.get('object')})",
                "question": f"Are the action ('{act_en}') and technological object ('{x.get('object')}') accurately described?",
                "model_claim": f"action: {x.get('action')} | object: {x.get('object')}",
                "model_value": True,
                "context_hint": "Verify that the action accurately characterizes what the firm does and that the object is the specific tool or model.",
            })

            # Target & Stage
            flattened["activities"].append({
                "id": f"{pid}:a{aidx}:target_stage",
                "ambito": "activities",
                "category": "target_stage",
                "category_label": "Target & Stage",
                **base_meta,
                "highlight_sentences": ev,
                "title": f"Activity #{aidx}: Target ({tgt_en}) & Stage ({stg_en})",
                "question": f"Do the target beneficiary ('{tgt_en}') and deployment stage ('{stg_en}') accurately reflect the disclosure?",
                "model_claim": f"target: {x.get('target')} | stage: {x.get('stage')}",
                "model_value": True,
                "context_hint": "Verify whether the target is who uses/benefits from the AI, and if the stage (exploration, pilot, deployed) matches.",
            })

            # Entities
            for e in x.get("entities") or []:
                e_name = e.get("name", "")
                e_role = e.get("role", "")
                role_en = ROLE_EN.get(e_role, e_role)
                flattened["activities"].append({
                    "id": f"{pid}:a{aidx}:ent_{e_name}",
                    "ambito": "activities",
                    "category": "entities",
                    "category_label": "Named Entity",
                    **base_meta,
                    "highlight_sentences": ev,
                    "title": f"Activity #{aidx}: Entity '{e_name}' ({role_en})",
                    "question": f"Is the entity '{e_name}' accurately identified with the role '{role_en}'?",
                    "model_claim": f"Entity: {e_name} | Role: {e_role}",
                    "model_value": e_role,
                    "context_hint": "Note: A company's own products are never 'external vendor'. Partners, customers, or models must reflect their factual role.",
                })

            # Evidence
            if ev:
                flattened["activities"].append({
                    "id": f"{pid}:a{aidx}:evidence",
                    "ambito": "activities",
                    "category": "evidence",
                    "category_label": "Activity Evidence",
                    **base_meta,
                    "highlight_sentences": ev,
                    "title": f"Activity #{aidx}: Textual Evidence [{', '.join(str(i) for i in ev)}]",
                    "question": f"Do the highlighted sentence(s) [{', '.join(str(i) for i in ev)}] contain the direct textual evidence for this activity?",
                    "model_claim": f"Evidence sentence indices: {ev}",
                    "model_value": ev,
                    "context_hint": "Verify whether the highlighted sentence(s) directly support the extracted activity.",
                })

    return flattened


def _run_isolated(fn, *args):
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        return ex.submit(fn, *args).result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--prefilter", type=int, default=300)
    parser.add_argument("--activities", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--from-existing",
        type=Path,
        default=None,
        help="Flatten from an existing data.json instead of re-sampling bronze/silver",
    )
    args = parser.parse_args()

    if args.from_existing and args.from_existing.exists():
        print(f"Loading existing sample from {args.from_existing}...")
        existing_doc = json.loads(args.from_existing.read_text(encoding="utf-8"))
        raw_data = existing_doc.get("raw_paragraphs") or existing_doc
        seed = existing_doc.get("seed", args.seed)
        pred_run = existing_doc.get("predictions_run", "existing")
    else:
        firm_lookup = register_firm_lookup()
        frames = _run_isolated(sample_frames, args.frames, args.seed, firm_lookup)
        prefilter = _run_isolated(sample_prefilter, args.prefilter, args.seed, firm_lookup)
        activities = _run_isolated(sample_activities, args.activities, args.seed, firm_lookup)
        raw_data = {
            "seed": args.seed,
            "predictions_run": latest_predictions().name,
            "frames": frames,
            "prefilter": prefilter,
            "activities": activities,
        }
        seed = args.seed
        pred_run = raw_data["predictions_run"]

    flattened = flatten_all(raw_data)
    counts = {k: len(v) for k, v in flattened.items()}
    total = sum(counts.values())

    payload = {
        "version": "v2_atomic",
        "seed": seed,
        "predictions_run": pred_run,
        "ambitos": ["prefilter", "frames", "activities"],
        "counts": counts,
        "raw_paragraphs": raw_data,
        "items": flattened,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"Atomic flattened sample saved to {args.out}:\n"
        f"  - prefilter:  {counts['prefilter']} verifications\n"
        f"  - frames:     {counts['frames']} verifications\n"
        f"  - activities: {counts['activities']} verifications\n"
        f"  TOTAL:        {total} atomic verifications (Y/N/U)"
    )


if __name__ == "__main__":
    main()
