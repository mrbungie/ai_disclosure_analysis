import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import pipeline_logger
    import variant_utils
except ImportError:
    from scripts import pipeline_logger
    from scripts import variant_utils

MENTIONS_COLS = [
    "chunk_id", "is_ai_related", "is_substantive", "is_promotional",
    "is_risk_related", "is_governance_related", "mentions_copilot", "mentions_cloud",
    "mentions_vendor", "mentions_training", "is_financial_impact", "rationale_short",
    "amounts", "entities", "sentiment"
]


def load_config():
    config_path = Path("configs/config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def build_proxy_sql():
    """SQL fragment (shared by both variants) computing has_* derived proxy flags."""
    return f"""
    scored_intermediate AS (
        SELECT
            *,
            (has_metric_percentage OR has_metric_dollar) as has_metrics,
            (
                (has_model_training AND NOT has_workforce_talent AND NOT has_ai_hedge)
                OR has_specific_product
                OR has_proprietary_data
                OR (has_compute_infra AND has_word_ai
                    AND (has_word_gpu OR has_word_gpus OR has_realized_language OR has_deployment_verb)
                    AND NOT has_ai_demand_context)
                OR (has_compute_infra AND has_word_ai AND has_ai_quantified_claim AND NOT has_ai_demand_context)
                OR (has_ethics_policy AND has_word_ai AND NOT has_risk_factor)
                OR (has_classical_ml_mention AND (has_internal_productivity OR has_customer_facing) AND NOT has_workforce_talent)
                OR (has_ai_own_use AND NOT has_ai_hedge AND NOT has_competitor_mention AND NOT has_ai_demand_context)
                OR (has_classical_ml_operational AND NOT has_risk_factor)
                OR has_ai_acquisition
            ) as substantive_proxy,
            (
                has_board_oversight
                OR has_audit_committee
                OR (has_ethics_policy AND has_word_ai)
                OR (has_compliance AND has_word_ai)
            ) as is_governance_related_proxy,
            (
                has_risk_factor
                OR has_regulatory_risk
                OR has_ethics_bias_risk
                OR has_supply_infra_risk
                OR has_data_licensing
                OR (has_cyber_privacy_risk AND has_word_ai)
                OR (has_ip_copyright_risk AND has_word_ai)
            ) as is_risk_related_proxy,
            (
                -- Promotional TONE, not "mentions a deployed AI product" (the old
                -- deployment-verb/named-product formula conflated the two and fired
                -- on ~42% of chunks against an LLM-judged true rate of ~13%,
                -- F1=0.277/P=0.182 on the 236-chunk validation sample). Hype
                -- vocabulary (count_vague_words, the VAGUE_WORDS list) outweighing
                -- negative-tone words, outside risk-factor language, with
                -- non-negative overall sentiment tests as boosterish tone directly:
                -- F1=0.400/P=0.270/R=0.774 on the same sample.
                (count_vague_words > count_negative_words)
                AND NOT has_risk_factor
                AND bow_sentiment_score >= 0
            ) as is_promotional_proxy,
            has_model_training as mentions_training_proxy,
            TRUE as is_ai_related_proxy,
            (has_vendor_nvidia OR has_vendor_openai OR has_vendor_microsoft OR has_vendor_google OR has_vendor_deepseek OR has_vendor_amazon OR has_vendor_meta OR has_vendor_anthropic OR has_vendor_amd) as has_vendor_mention
        FROM merged
    )
    """


def build_output_cols(bow_df):
    bow_cols = [c for c in bow_df.columns if c != "chunk_id"]
    out_cols = [
        "chunk_id", "accession_number", "ticker", "filing_date", "section_name", "chunk_text",
        "final_specificity", "final_governance_score", "final_risk_score", "final_promotional_score",
        "has_ai_disclosure", "is_substantive", "is_promotional", "is_risk_related",
        "is_governance_related", "is_ai_related",
        "llm_is_ai_related", "llm_is_substantive", "llm_is_promotional", "llm_is_risk_related",
        "llm_is_governance_related", "llm_is_financial_impact", "llm_mentions_training",
        "llm_mentions_vendor", "llm_mentions_cloud", "llm_mentions_copilot",
        "amounts", "entities", "sentiment",
    ] + bow_cols
    seen = set()
    return [x for x in out_cols if not (x in seen or seen.add(x))]


def score_rule_based(chunks_df, bow_df, output_path):
    """rule_based variant: today's --bow-only behavior, unchanged logic. Ignores
    ai_disclosure_mentions.parquet entirely (LLM fields always fall back to the
    deterministic BoW proxy)."""
    import duckdb

    pipeline_logger.log_event(
        pipeline_step="combined_scoring", level="WARNING",
        message="rule_based variant: scoring relies entirely on BoW proxy features (no LLM merge).",
        details={"variant": "rule_based"},
    )
    print("rule_based variant: scoring will rely entirely on BoW features.")

    mentions_df = pd.DataFrame(columns=MENTIONS_COLS)
    for col in MENTIONS_COLS + ["accession_number"]:
        if col not in mentions_df.columns:
            mentions_df[col] = None

    con = duckdb.connect()
    con.register("chunks_df", chunks_df)
    con.register("bow_df", bow_df)
    con.register("mentions_df", mentions_df)

    sql_query = f"""
    WITH chunks AS (SELECT * FROM chunks_df),
    bow AS (
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY chunk_id) as rn FROM bow_df
        ) WHERE rn = 1
    ),
    mentions AS (
        SELECT * FROM (
            SELECT * EXCLUDE (accession_number) REPLACE (
                COALESCE(is_ai_related, FALSE) as is_ai_related,
                COALESCE(is_substantive, FALSE) as is_substantive,
                COALESCE(is_promotional, FALSE) as is_promotional,
                COALESCE(is_risk_related, FALSE) as is_risk_related,
                COALESCE(is_governance_related, FALSE) as is_governance_related,
                COALESCE(mentions_training, FALSE) as mentions_training,
                COALESCE(mentions_copilot, FALSE) as mentions_copilot,
                COALESCE(mentions_cloud, FALSE) as mentions_cloud,
                COALESCE(mentions_vendor, FALSE) as mentions_vendor,
                COALESCE(is_financial_impact, FALSE) as is_financial_impact
            ), row_number() OVER (PARTITION BY chunk_id) as rn
            FROM mentions_df
        ) WHERE rn = 1
    ),
    merged AS (
        SELECT c.*, b.* EXCLUDE (chunk_id, rn), m.* EXCLUDE (chunk_id, rn)
        FROM chunks c
        LEFT JOIN bow b ON c.chunk_id = b.chunk_id
        LEFT JOIN mentions m ON c.chunk_id = m.chunk_id
    ),
    {build_proxy_sql()},
    scored_final AS (
        SELECT
            chunk_id, accession_number, ticker, filing_date, section_name, chunk_text,
            COALESCE(is_substantive, substantive_proxy) as is_substantive,
            COALESCE(is_promotional, is_promotional_proxy) as is_promotional,
            COALESCE(is_risk_related, is_risk_related_proxy) as is_risk_related,
            COALESCE(is_governance_related, is_governance_related_proxy) as is_governance_related,
            COALESCE(is_ai_related, is_ai_related_proxy) as is_ai_related,
            COALESCE(sentiment, 'Neutral') as sentiment,
            amounts, entities,
            LEAST(COALESCE(is_substantive, substantive_proxy)::INT +
                  (has_metric_percentage OR has_metric_dollar)::INT +
                  COALESCE(mentions_training, mentions_training_proxy)::INT, 3) as final_specificity,
            LEAST(COALESCE(is_governance_related, is_governance_related_proxy)::INT +
                  has_board_oversight::INT + has_audit_committee::INT, 3) as final_governance_score,
            LEAST(COALESCE(is_risk_related, is_risk_related_proxy)::INT +
                  (COALESCE(is_risk_related, is_risk_related_proxy) AND (has_board_oversight OR has_audit_committee))::INT +
                  (COALESCE(is_risk_related, is_risk_related_proxy) AND has_vendor_mention)::INT, 3) as final_risk_score,
            LEAST(COALESCE(is_promotional, is_promotional_proxy)::INT +
                  (COALESCE(is_promotional, is_promotional_proxy) AND NOT COALESCE(is_substantive, substantive_proxy))::INT +
                  (COALESCE(is_promotional, is_promotional_proxy) AND NOT (has_metric_percentage OR has_metric_dollar))::INT, 3) as final_promotional_score,
            (COALESCE(is_ai_related, is_ai_related_proxy) AND (COALESCE(is_substantive, substantive_proxy) OR COALESCE(is_risk_related, is_risk_related_proxy))) as has_ai_disclosure,
            COALESCE(is_ai_related, FALSE) as llm_is_ai_related,
            COALESCE(is_substantive, FALSE) as llm_is_substantive,
            COALESCE(is_promotional, FALSE) as llm_is_promotional,
            COALESCE(is_risk_related, FALSE) as llm_is_risk_related,
            COALESCE(is_governance_related, FALSE) as llm_is_governance_related,
            COALESCE(is_financial_impact, FALSE) as llm_is_financial_impact,
            COALESCE(mentions_training, FALSE) as llm_mentions_training,
            COALESCE(mentions_vendor, FALSE) as llm_mentions_vendor,
            COALESCE(mentions_cloud, FALSE) as llm_mentions_cloud,
            COALESCE(mentions_copilot, FALSE) as llm_mentions_copilot,
            * EXCLUDE (
                chunk_id, accession_number, ticker, filing_date, section_name, chunk_text,
                is_substantive, is_promotional, is_risk_related, is_governance_related, is_ai_related,
                sentiment, amounts, entities,
                has_metrics, substantive_proxy, is_governance_related_proxy, is_risk_related_proxy, is_promotional_proxy,
                mentions_training_proxy, is_ai_related_proxy, has_vendor_mention, mentions_training,
                is_financial_impact, mentions_vendor, mentions_cloud, mentions_copilot
            )
        FROM scored_intermediate
    )
    SELECT * FROM scored_final;
    """

    print(f"Scoring {len(chunks_df)} chunks using BoW features only (rule_based variant) via DuckDB SQL...")
    df_duckdb = con.execute(sql_query).df()

    out_cols = [c for c in build_output_cols(bow_df) if c in df_duckdb.columns]
    scored_df = df_duckdb[out_cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(output_path, index=False)

    pipeline_logger.log_event(
        pipeline_step="combined_scoring", level="SUCCESS",
        message=f"rule_based combined scoring complete. Saved {len(scored_df)} scored chunks to {output_path}.",
        details={"variant": "rule_based"},
    )
    print(f"Success: Saved {len(scored_df)} scored chunks to {output_path}")


def score_llm_full(chunks_df, bow_df, mentions_df, output_path):
    """llm_full variant: uses ai_disclosure_mentions.parquet with NO fallback to
    the BoW proxy. Chunks without a real LLM label keep NULL for the 5 core
    boolean fields and are marked llm_label_missing = TRUE, instead of being
    silently completed via regex proxy."""
    import duckdb

    if len(mentions_df) == 0:
        print("Error: ai_disclosure_mentions.parquet is empty. Run script 08 (LLM classifier) first for llm_full variant.")
        pipeline_logger.log_event(
            pipeline_step="combined_scoring", level="ERROR",
            message="llm_full variant requested but ai_disclosure_mentions.parquet has no rows.",
            details={"variant": "llm_full"},
        )
        sys.exit(1)

    con = duckdb.connect()
    con.register("chunks_df", chunks_df)
    con.register("bow_df", bow_df)
    con.register("mentions_df", mentions_df)

    sql_query = f"""
    WITH chunks AS (SELECT * FROM chunks_df),
    bow AS (
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY chunk_id) as rn FROM bow_df
        ) WHERE rn = 1
    ),
    mentions AS (
        SELECT * FROM (
            SELECT * EXCLUDE (accession_number), row_number() OVER (PARTITION BY chunk_id) as rn
            FROM mentions_df
        ) WHERE rn = 1
    ),
    matched AS (SELECT DISTINCT chunk_id FROM mentions),
    merged AS (
        SELECT c.*, b.* EXCLUDE (chunk_id, rn), m.* EXCLUDE (chunk_id, rn),
               (mt.chunk_id IS NOT NULL) as llm_label_present
        FROM chunks c
        LEFT JOIN bow b ON c.chunk_id = b.chunk_id
        LEFT JOIN mentions m ON c.chunk_id = m.chunk_id
        LEFT JOIN matched mt ON c.chunk_id = mt.chunk_id
    ),
    {build_proxy_sql()},
    scored_final AS (
        SELECT
            chunk_id, accession_number, ticker, filing_date, section_name, chunk_text,
            is_substantive, is_promotional, is_risk_related, is_governance_related, is_ai_related,
            sentiment, amounts, entities,
            NOT llm_label_present as llm_label_missing,
            LEAST(is_substantive::INT + (has_metric_percentage OR has_metric_dollar)::INT + mentions_training::INT, 3) as final_specificity,
            LEAST(is_governance_related::INT + has_board_oversight::INT + has_audit_committee::INT, 3) as final_governance_score,
            LEAST(is_risk_related::INT +
                  (is_risk_related AND (has_board_oversight OR has_audit_committee))::INT +
                  (is_risk_related AND has_vendor_mention)::INT, 3) as final_risk_score,
            LEAST(is_promotional::INT +
                  (is_promotional AND NOT is_substantive)::INT +
                  (is_promotional AND NOT (has_metric_percentage OR has_metric_dollar))::INT, 3) as final_promotional_score,
            (is_ai_related AND (is_substantive OR is_risk_related)) as has_ai_disclosure,
            is_ai_related as llm_is_ai_related,
            is_substantive as llm_is_substantive,
            is_promotional as llm_is_promotional,
            is_risk_related as llm_is_risk_related,
            is_governance_related as llm_is_governance_related,
            is_financial_impact as llm_is_financial_impact,
            mentions_training as llm_mentions_training,
            mentions_vendor as llm_mentions_vendor,
            mentions_cloud as llm_mentions_cloud,
            mentions_copilot as llm_mentions_copilot,
            * EXCLUDE (
                chunk_id, accession_number, ticker, filing_date, section_name, chunk_text,
                sentiment, amounts, entities, llm_label_present,
                has_metrics, substantive_proxy, is_governance_related_proxy, is_risk_related_proxy, is_promotional_proxy,
                mentions_training_proxy, is_ai_related_proxy, has_vendor_mention,
                is_substantive, is_promotional, is_risk_related, is_governance_related, is_ai_related,
                mentions_training, is_financial_impact, mentions_vendor, mentions_cloud, mentions_copilot
            )
        FROM scored_intermediate
    )
    SELECT * FROM scored_final;
    """

    print(f"Scoring {len(chunks_df)} chunks using LLM labels only (llm_full variant, no proxy fallback) via DuckDB SQL...")
    df_duckdb = con.execute(sql_query).df()

    out_cols = [c for c in build_output_cols(bow_df) + ["llm_label_missing"] if c in df_duckdb.columns]
    scored_df = df_duckdb[out_cols]

    total = len(scored_df)
    n_missing = int(scored_df["llm_label_missing"].sum())
    coverage_pct = (total - n_missing) / total * 100 if total else 0.0

    coverage_msg = f"[variant] llm_full coverage: {coverage_pct:.1f}% — {n_missing:,} / {total:,} chunks missing LLM label"
    print(coverage_msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(output_path, index=False)

    coverage_report_path = output_path.parent / "coverage_summary__llm_full.txt"
    coverage_report_path.write_text(coverage_msg + "\n")
    print(f"Coverage summary written to {coverage_report_path}")

    level = "WARNING" if coverage_pct < 95 else "INFO"
    pipeline_logger.log_event(
        pipeline_step="combined_scoring", level=level,
        message=coverage_msg,
        details={"variant": "llm_full", "coverage_pct": coverage_pct, "n_missing": n_missing, "n_total": total},
    )

    pipeline_logger.log_event(
        pipeline_step="combined_scoring", level="SUCCESS",
        message=f"llm_full combined scoring complete. Saved {len(scored_df)} scored chunks to {output_path}.",
        details={"variant": "llm_full"},
    )
    print(f"Success: Saved {len(scored_df)} scored chunks to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Score chunks with LLM booleans or BoW proxies")
    # Design Decision: The LLM classifier is made optional using the --bow-only flag.
    # This allows running the entire pipeline using purely deterministic BoW proxies,
    # ensuring data homogeneity and avoiding temporal anomalies (such as the 2023 dip)
    # caused by incomplete LLM classifications across years.
    parser.add_argument("--bow-only", action="store_true", help="Force BoW-only mode, ignoring LLM classifications")
    variant_utils.add_variant_arg(parser)
    args = parser.parse_args()

    config = load_config()
    variant = variant_utils.resolve_variant(args.variant, config)

    if args.bow_only and variant == "llm_full":
        print("Error: --bow-only is incompatible with --variant llm_full.")
        sys.exit(1)

    output_root = config.get("variants", {}).get("output_root", "data/processed")
    output_path = variant_utils.variant_path(variant, "ai_scored_chunks", "parquet", output_root=output_root)

    chunks_path = Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet"
    bow_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_bow_features.parquet"
    mentions_path = Path(config["paths"]["candidate_chunks"]) / "ai_disclosure_mentions.parquet"

    missing_files = [p.name for p in [chunks_path, bow_path] if not p.exists()]
    if missing_files:
        pipeline_logger.log_event(
            pipeline_step="combined_scoring", level="ERROR",
            message=f"Missing input parquets: {', '.join(missing_files)}",
            details={"variant": variant},
        )
        print(f"Error: Missing input parquets: {', '.join(missing_files)}")
        return

    pipeline_logger.log_event(
        pipeline_step="combined_scoring", level="INFO",
        message="Loading input parquets for combined scoring...",
        details={"variant": variant},
    )

    chunks_df = pd.read_parquet(chunks_path)
    bow_df = pd.read_parquet(bow_path)

    if variant == "rule_based":
        score_rule_based(chunks_df, bow_df, output_path)
        return

    # llm_full
    if not mentions_path.exists():
        print(f"Error: {mentions_path} not found. Run script 08 (LLM classifier) first for llm_full variant.")
        pipeline_logger.log_event(
            pipeline_step="combined_scoring", level="ERROR",
            message=f"llm_full variant requires {mentions_path}, which does not exist.",
            details={"variant": "llm_full"},
        )
        return

    mentions_df = pd.read_parquet(mentions_path)
    for col in MENTIONS_COLS + ["accession_number"]:
        if col not in mentions_df.columns:
            mentions_df[col] = None

    score_llm_full(chunks_df, bow_df, mentions_df, output_path)


if __name__ == "__main__":
    main()
