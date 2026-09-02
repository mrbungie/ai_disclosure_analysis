"""Versioned semantic retrieval anchors for the AI paragraph prefilter."""

from __future__ import annotations

from collections.abc import Sequence


POSITIVE_ANCHORS: dict[str, tuple[str, ...]] = {
    "ai_use": (
        "The company is using or deploying artificial intelligence in business operations.",
        "Artificial intelligence is integrated into a product, service, or workflow.",
    ),
    "ai_exploration": (
        "The company is experimenting with, piloting, evaluating, or exploring artificial intelligence.",
    ),
    "ai_capability": (
        "The company develops proprietary artificial intelligence models, systems, or platforms.",
        "The company relies on third-party artificial intelligence providers or models.",
        "The company invests in AI infrastructure, computing, data, or talent.",
    ),
    "ai_outcome": (
        "Artificial intelligence improves productivity or operational efficiency.",
        "Artificial intelligence reduces costs.",
        "Artificial intelligence generates revenue or commercial growth.",
        "Artificial intelligence improves customer experience or personalization.",
    ),
    "ai_risk": (
        "Artificial intelligence creates cybersecurity or privacy risks.",
        "Artificial intelligence creates regulatory, legal, copyright, or intellectual-property risks.",
        "Artificial intelligence may produce inaccurate, biased, unreliable, or harmful outputs.",
        "Artificial intelligence creates competitive or workforce disruption.",
    ),
    "ai_governance": (
        "The company has governance, policies, oversight, controls, or human review for artificial intelligence.",
    ),
    "ai_strategy": (
        "Artificial intelligence is described as strategically important or transformative for the company.",
    ),
}

NEGATIVE_ANCHORS: tuple[str, ...] = (
    "The paragraph discusses a business model rather than machine learning.",
    "The paragraph discusses financial valuation models.",
    "The paragraph discusses economic forecasting without artificial intelligence.",
    "The paragraph discusses ordinary software automation without machine learning or AI.",
)


def anchor_rows() -> list[dict[str, str]]:
    """Return one stable, versioned row per semantic anchor."""
    rows = []
    for category, anchors in POSITIVE_ANCHORS.items():
        for number, text in enumerate(anchors, start=1):
            rows.append({
                "anchor_id": f"{category}_{number:02d}",
                "category": category,
                "polarity": "positive",
                "anchor_text": text,
            })
    for number, text in enumerate(NEGATIVE_ANCHORS, start=1):
        rows.append({
            "anchor_id": f"negative_{number:02d}",
            "category": "negative",
            "polarity": "negative",
            "anchor_text": text,
        })
    return rows


def register_anchors(connection, table_name: str = "ai_prefilter_anchors") -> Sequence[dict[str, str]]:
    """Create a DuckDB table so anchors are inspectable alongside scores."""
    rows = anchor_rows()
    connection.execute(f"DROP TABLE IF EXISTS {table_name}")
    connection.execute(
        f"CREATE TABLE {table_name} (anchor_id VARCHAR, category VARCHAR, polarity VARCHAR, anchor_text VARCHAR)"
    )
    connection.executemany(
        f"INSERT INTO {table_name} VALUES (?, ?, ?, ?)",
        [(row["anchor_id"], row["category"], row["polarity"], row["anchor_text"]) for row in rows],
    )
    return rows
