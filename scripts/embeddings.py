"""
embeddings.py — Local sentence-embedding cache shared by Phase 0 (concept
discovery, scripts/phase0_discovery.py) and the seed screen's optional
semantic-hit signal (scripts/seed_screen.py). Not a harness candidate and
not scored — a plain utility module, same footing as harness_fit.py.

The embedding model is configurable (configs/config.json: phase0.
embedding_models / phase0.active_embedding_model) and embeddings are
NEVER deleted from storage: each model gets its own cache file
(data/interim/embeddings/<model_key>.parquet), so switching
active_embedding_model never touches another model's cached vectors, and
re-running never truncates or overwrites rows already on disk — only
missing paragraph_ids are computed and appended.

Usage:
    from embeddings import get_or_compute_embeddings, cosine_similarity

    vecs = get_or_compute_embeddings(df, model_key="bge-m3", config=config)
    # vecs: paragraph_id, embedding (np.ndarray[float32])
"""

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# HuggingFace Hub's Xet download backend (its default as of late 2026)
# fails on this network with "CAS Client Error: Format error: I/O error:
# error decoding response body" on some large-model downloads (BGE-M3
# reproduced it consistently) — falling back to the plain HTTP downloader
# fixes it. Must be set before sentence_transformers/huggingface_hub is
# imported anywhere in the process, so set it at module import time here
# rather than relying on every caller's shell environment.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_MODEL_CACHE: dict[str, object] = {}


def _resolve_device(config: dict) -> str:
    device = config["phase0"].get("embedding_device", "auto")
    if device != "auto":
        return device
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embedder(model_key: str, config: dict):
    """Load (and cache in-process) a SentenceTransformer for model_key, per
    configs/config.json: phase0.embedding_models. Loading is expensive
    (model download/weights-to-device) so one process reuses one instance
    per model_key rather than reloading per call."""
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]
    spec = config["phase0"]["embedding_models"][model_key]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(spec["hf_id"], device=_resolve_device(config))
    if spec.get("max_seq_length"):
        model.max_seq_length = spec["max_seq_length"]
    _MODEL_CACHE[model_key] = model
    return model


def _cache_path(model_key: str, config: dict) -> Path:
    return Path(config["paths"]["interim_embeddings"]) / f"{model_key}.parquet"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_or_compute_embeddings(df: pd.DataFrame, model_key: str, config: dict,
                              id_col: str = "paragraph_id", text_col: str = "paragraph_text") -> pd.DataFrame:
    """Append-only embedding cache. `df` must have `id_col` and `text_col`.
    Returns id_col + `embedding` (np.ndarray[float32], one row per id in
    df, order not guaranteed to match df). Rows already cached (matching
    id AND text_hash) are never recomputed; rows whose text changed since
    caching are recomputed and appended as a NEW row for that id, on top
    of — not replacing — the stale one (never deleted, per design)."""
    path = _cache_path(model_key, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(path) if path.exists() else None

    df = df[[id_col, text_col]].drop_duplicates(subset=[id_col]).copy()
    df["text_hash"] = df[text_col].map(text_hash)

    if existing is not None and len(existing):
        latest_cached = existing.sort_values("computed_at").drop_duplicates(subset=[id_col], keep="last")
        merged = df.merge(latest_cached[[id_col, "text_hash"]], on=id_col, how="left", suffixes=("", "_cached"))
        to_embed = df[merged["text_hash"] != merged["text_hash_cached"]]
    else:
        to_embed = df

    if len(to_embed):
        model = load_embedder(model_key, config)
        batch_size = config["phase0"].get("embedding_batch_size", 32)
        vectors = model.encode(to_embed[text_col].tolist(), batch_size=batch_size,
                               show_progress_bar=len(to_embed) > 50, convert_to_numpy=True)
        new_rows = pd.DataFrame({
            id_col: to_embed[id_col].to_numpy(),
            "model_key": model_key,
            "embedding": list(vectors.astype(np.float32)),
            "text_hash": to_embed["text_hash"].to_numpy(),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        })
        combined = pd.concat([existing, new_rows], ignore_index=True) if existing is not None else new_rows
        combined.to_parquet(path, index=False)
    else:
        combined = existing

    if combined is None or not len(combined):
        empty = df[[id_col]].copy()
        empty["embedding"] = None
        return empty

    latest = combined.sort_values("computed_at").drop_duplicates(subset=[id_col], keep="last")
    result = df[[id_col]].merge(latest[[id_col, "embedding"]], on=id_col, how="left")
    result["embedding"] = result["embedding"].map(lambda v: np.asarray(v, dtype=np.float32))
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n, d), b: (m, d) -> (n, m) cosine similarity matrix."""
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T
