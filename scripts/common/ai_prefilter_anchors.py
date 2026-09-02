"""Anchors and lexical terms for the AI paragraph prefilter, loaded from config.

Both signals live in configs/ai_prefilter.yaml rather than in this file: tuning
a prefilter means rewriting anchors repeatedly, and that should not be a code
change. See that file's header for how to write an anchor — the short version
is that an anchor must read like the paragraphs it should retrieve, because it
is matched by cosine similarity against them.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ai_prefilter.yaml"


@functools.cache
def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for section, keys in (("lexical", ("strong", "weak")), ("semantic", ("positive", "negative"))):
        if section not in config:
            raise ValueError(f"{path}: missing '{section}' section")
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise ValueError(f"{path}: {section} is missing {missing}")
    if not config["semantic"]["positive"]:
        raise ValueError(f"{path}: semantic.positive has no categories")
    return config


def _normalize(text: str) -> str:
    """YAML folds a wrapped scalar into one line with single spaces; collapse any
    remaining newlines so an anchor's embedding never depends on line wrapping."""
    return " ".join(str(text).split())


def strong_terms(path: Path | str = DEFAULT_CONFIG) -> tuple[str, ...]:
    return tuple(str(term).lower() for term in load_config(path)["lexical"]["strong"])


def weak_terms(path: Path | str = DEFAULT_CONFIG) -> tuple[str, ...]:
    return tuple(str(term).lower() for term in load_config(path)["lexical"]["weak"])


def positive_anchors(path: Path | str = DEFAULT_CONFIG) -> dict[str, tuple[str, ...]]:
    """Category -> anchor texts, in config order (which is the column order of
    the score_<category> columns downstream)."""
    return {
        category: tuple(_normalize(text) for text in texts)
        for category, texts in load_config(path)["semantic"]["positive"].items()
    }


def negative_anchors(path: Path | str = DEFAULT_CONFIG) -> tuple[str, ...]:
    return tuple(_normalize(text) for text in load_config(path)["semantic"]["negative"])


def anchor_rows(path: Path | str = DEFAULT_CONFIG) -> list[dict[str, str]]:
    """One stable row per anchor. Ids are positional within a category, so
    reordering the config renames anchors — the content hash in ai_prefilter
    catches that and refuses to mix the old scores with the new."""
    rows = []
    for category, anchors in positive_anchors(path).items():
        for number, text in enumerate(anchors, start=1):
            rows.append({
                "anchor_id": f"{category}_{number:02d}",
                "category": category,
                "polarity": "positive",
                "anchor_text": text,
            })
    for number, text in enumerate(negative_anchors(path), start=1):
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


# Back-compat aliases: score_embeddings iterates POSITIVE_ANCHORS to fix the
# category column order.
POSITIVE_ANCHORS = positive_anchors()
NEGATIVE_ANCHORS = negative_anchors()
