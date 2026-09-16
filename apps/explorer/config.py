"""
apps/explorer/config.py — Configuration and environment settings for the dataset explorer.
Supports local filesystem and remote protocols (s3://, hf://, https://, http://, file://).
"""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path

# Repository root (two levels up from apps/explorer)
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]


def get_default_data_uri() -> str:
    """Detect default location: DATA_URI / DATA_URL / DATA_PATH / DATA_DIR or local data/."""
    for key in ("DATA_URI", "DATA_URL", "DATA_PATH", "DATA_DIR"):
        val = os.getenv(key)
        if val and val.strip():
            return val.strip()

    # Local fallback
    repo_data = REPO_ROOT / "data"
    if repo_data.exists():
        return str(repo_data)
    local_data = Path("data").resolve()
    if local_data.exists():
        return str(local_data)
    return str(repo_data)


DATA_URI = get_default_data_uri()


def is_remote_uri(uri: str) -> bool:
    """Check whether a URI is using a remote network protocol (s3, https, http, hf)."""
    parsed = urllib.parse.urlparse(uri)
    return parsed.scheme.lower() in ("s3", "https", "http", "hf")


# S3 / B2 / Cloud Object Storage credentials
S3_ENDPOINT = (
    os.getenv("S3_ENDPOINT")
    or os.getenv("AWS_ENDPOINT_URL")
    or os.getenv("B2_ENDPOINT")
    or ""
)
S3_ACCESS_KEY_ID = (
    os.getenv("S3_ACCESS_KEY_ID")
    or os.getenv("AWS_ACCESS_KEY_ID")
    or os.getenv("B2_KEY_ID")
    or ""
)
S3_SECRET_ACCESS_KEY = (
    os.getenv("S3_SECRET_ACCESS_KEY")
    or os.getenv("AWS_SECRET_ACCESS_KEY")
    or os.getenv("B2_APPLICATION_KEY")
    or ""
)
S3_REGION = (
    os.getenv("S3_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or os.getenv("B2_REGION")
    or "us-east-1"
)
S3_URL_STYLE = os.getenv("S3_URL_STYLE", "path" if "backblazeb2" in S3_ENDPOINT or "minio" in S3_ENDPOINT else "vhost")

# Hugging Face token for private spaces/datasets
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Server configuration
SERVER_NAME = os.getenv("EXPLORER_HOST", os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
SERVER_PORT = int(os.getenv("EXPLORER_PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
SHARE = os.getenv("EXPLORER_SHARE", "false").lower() in ("true", "1", "yes")
DEBUG = os.getenv("EXPLORER_DEBUG", "false").lower() in ("true", "1", "yes")
