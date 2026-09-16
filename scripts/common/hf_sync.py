"""Hugging Face synchronization utility for project data and models.

Manages private Hugging Face repositories:
  - Dataset: https://huggingface.co/datasets/mrbungie/ai-disclosure-analysys
  - Model:   https://huggingface.co/mrbungie/ai-disclosure-analysys

Supports simultaneous/concurrent upload of models, structured data layers,
and compressed raw data archives (.tar.gz).

Usage:
    # Sync models and structured data layers concurrently (default):
    uv run python scripts/common/hf_sync.py --both

    # Sync everything including compressed raw data archives (.tar.gz):
    uv run python scripts/common/hf_sync.py --all

    # Sync only raw data archives:
    uv run python scripts/common/hf_sync.py --raw

    # Sync only models:
    uv run python scripts/common/hf_sync.py --models

    # Sync specific data layers:
    uv run python scripts/common/hf_sync.py --data --layers gold silver results
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k == "HF_TOKEN" and v.startswith("hf_"):
                    os.environ[k] = v
                else:
                    os.environ.setdefault(k, v)


load_env()

DEFAULT_TOKEN = "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"
DEFAULT_DATASET_REPO = os.environ.get("HF_DATASET_REPO") or os.environ.get("HF_REPO") or "mrbungie/ai-disclosure-analysys"
DEFAULT_MODEL_REPO = os.environ.get("HF_MODEL_REPO") or os.environ.get("HF_REPO") or "mrbungie/ai-disclosure-analysys"


def resolve_token(arg_token: str | None = None) -> str:
    if arg_token:
        return arg_token.strip()
    env_token = os.environ.get("HF_TOKEN")
    if env_token and env_token.startswith("hf_"):
        return env_token.strip()
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_path.exists():
        t = cache_path.read_text().strip()
        if t.startswith("hf_"):
            return t
    return DEFAULT_TOKEN


def get_api(token: str) -> HfApi:
    os.environ["HF_TOKEN"] = token
    return HfApi(token=token)


def ensure_repos(api: HfApi, dataset_repo: str, model_repo: str, private: bool = True) -> None:
    print(f"Ensuring repositories exist...")
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=private, exist_ok=True)
    api.create_repo(repo_id=model_repo, repo_type="model", private=private, exist_ok=True)
    print(f"  ✓ Dataset: https://huggingface.co/datasets/{dataset_repo}")
    print(f"  ✓ Model:   https://huggingface.co/{model_repo}")


def sync_models(api: HfApi, model_repo: str, commit_message: str = "Sync models") -> str:
    models_dir = REPO_ROOT / "models"
    if not models_dir.exists():
        print("  [Models] 'models/' directory not found, skipping.")
        return ""

    print(f"\n[Models] Starting upload from {models_dir} to https://huggingface.co/{model_repo}...")
    t0 = time.time()
    readme_content = f"""---
license: mit
tags:
- finance
- corporate-disclosures
- ai-classification
- sp500
---

# S&P 500 AI Corporate Disclosure Models

This repository stores trained classification models, posture archetypes, expanding window cutoffs, and shrinkage weights for empirical thesis research on corporate AI disclosure.

## Architecture
- `ai_classification/`: Prefilter binary classifiers (TF-IDF + Ridge/LogReg/LightGBM runs) detecting corporate AI discussion.
- `posture_archetype_expanding/`: Expanding window posture archetypes (quarterly cutoffs 2021Q4 - 2026Q3).
- `posture_archetype_expanding_yearly/`: Yearly posture archetypes (cutoffs 2021 - 2026).
- `posture_archetype_static/`: Static benchmark posture archetypes.
- `posture_archetype_weights/`: Principal component and factor weights for posture dimensions.
- `incremental_signal/`: Incremental signal and beta models.
- `washing_grounding_shrinkage/`: AI washing and grounding empirical shrinkage models.

Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=model_repo,
        repo_type="model",
        commit_message="docs: update model card README",
    )

    url = api.upload_folder(
        folder_path=str(models_dir),
        repo_id=model_repo,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=[".DS_Store", "*.pyc", "__pycache__", ".git*"],
    )
    elapsed = time.time() - t0
    print(f"[Models] ✓ Upload completed in {elapsed:.1f}s: {url}")
    return str(url)


def sync_raw_archives(api: HfApi, dataset_repo: str, commit_message: str = "Sync raw archives") -> None:
    raw_dir = REPO_ROOT / "data" / "raw"
    if not raw_dir.exists():
        print("  [Data/Raw] 'data/raw' directory not found, skipping.")
        return

    archive_dir = REPO_ROOT / "scratch" / "raw_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Data/Raw] Packaging and uploading raw data archives (.tar.gz) to https://huggingface.co/datasets/{dataset_repo}/tree/main/raw...")
    t0 = time.time()

    # 1. Subdirectories -> .tar.gz archives
    for sub in sorted(raw_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            tar_path = archive_dir / f"{sub.name}.tar.gz"
            if not tar_path.exists() or tar_path.stat().st_mtime < sub.stat().st_mtime:
                print(f"  [Data/Raw] Compressing '{sub.name}' -> {tar_path.name}...")
                subprocess.run(["tar", "-czf", str(tar_path), "-C", str(raw_dir), sub.name], check=True)

            size_mb = tar_path.stat().st_size / (1024 * 1024)
            print(f"  [Data/Raw] Uploading '{tar_path.name}' ({size_mb:.1f} MB)...")
            sub_t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(tar_path),
                path_in_repo=f"raw/{tar_path.name}",
                repo_id=dataset_repo,
                repo_type="dataset",
                commit_message=f"{commit_message} (raw/{tar_path.name})",
            )
            print(f"  [Data/Raw] ✓ '{tar_path.name}' uploaded in {time.time() - sub_t0:.1f}s")

        elif sub.is_file() and not sub.name.startswith("."):
            print(f"  [Data/Raw] Uploading file '{sub.name}'...")
            api.upload_file(
                path_or_fileobj=str(sub),
                path_in_repo=f"raw/{sub.name}",
                repo_id=dataset_repo,
                repo_type="dataset",
                commit_message=f"{commit_message} (raw/{sub.name})",
            )

    print(f"[Data/Raw] ✓ Raw archives synced in {time.time() - t0:.1f}s!")


def sync_data(
    api: HfApi,
    dataset_repo: str,
    layers: list[str] | None = None,
    sync_raw: bool = False,
    commit_message: str = "Sync datasets",
) -> str:
    data_dir = REPO_ROOT / "data"
    if not data_dir.exists():
        print("  [Data] 'data/' directory not found, skipping.")
        return ""

    if layers is None:
        layers = ["gold", "silver", "results", "archive", "bronze"]

    print(f"\n[Data] Starting upload for layers: {', '.join(layers)} to https://huggingface.co/datasets/{dataset_repo}...")
    t0 = time.time()

    readme_content = f"""---
license: mit
task_categories:
- text-classification
- feature-extraction
language:
- en
tags:
- economics
- corporate-disclosures
- sp500
- 10-K
- 10-Q
- earnings-calls
size_categories:
- 100K<n<1M
---

# S&P 500 AI Corporate Disclosure Dataset

Structured analytical dataset and panels tracking corporate Artificial Intelligence disclosure across S&P 500 firms (2021 - 2026) across SEC filings (10-K, 10-Q, 8-K, DEF 14A) and quarterly earnings calls.

## Layer Structure
- `gold/`: Curated, business-ready joined panels (financials, posture, channel gaps, call beta, crash archetypes).
- `silver/`: S&P 500 universe definitions and paragraph-level LLM extractions (frames, activities, mentions).
- `results/`: Empirical regression estimates, robustness checks, bootstrap statistics, and figures.
- `bronze/`: Cleaned source manifests, paragraph partitions, market prices, and raw prefilter outputs.
- `interim/`: Intermediate parquet partitions (split sections, paragraph embeddings, raw prefilter scores/predictions, LLM prompt extractions).
- `archive/`: Metadata and universe references.
- `raw/`: Compressed source archives (`.tar.gz`) containing raw HTML/text filings, earnings call transcripts, and XBRL facts.

Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message="docs: update dataset card README",
    )

    tickers_file = data_dir / "sec_company_tickers.json"
    if tickers_file.exists():
        api.upload_file(
            path_or_fileobj=str(tickers_file),
            path_in_repo="sec_company_tickers.json",
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message="feat: sync sec_company_tickers.json",
        )

    last_url = ""
    for layer in layers:
        if layer == "raw":
            sync_raw = True
            continue
        layer_dir = data_dir / layer
        if not layer_dir.exists():
            print(f"  [Data] Layer '{layer}' not found at {layer_dir}, skipping.")
            continue
        print(f"  [Data] Uploading layer: '{layer}' ({layer_dir})...")
        sub_t0 = time.time()
        url = api.upload_folder(
            folder_path=str(layer_dir),
            path_in_repo=layer,
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message=f"{commit_message} ({layer})",
            ignore_patterns=[".DS_Store", "*.pyc", "__pycache__", ".git*", "deprecated/**"],
        )
        print(f"  [Data] ✓ Layer '{layer}' completed in {time.time() - sub_t0:.1f}s")
        last_url = str(url)

    if sync_raw:
        sync_raw_archives(api, dataset_repo, commit_message)

    elapsed = time.time() - t0
    print(f"[Data] ✓ Data layers sync completed in {elapsed:.1f}s!")
    return last_url


def sync_both(
    api: HfApi,
    dataset_repo: str,
    model_repo: str,
    layers: list[str] | None = None,
    sync_raw: bool = False,
    commit_message: str = "Sync data and models",
) -> None:
    print("\n=======================================================")
    print(" HUGGING FACE CONCURRENT SYNC (MODELS + DATASET)")
    print("=======================================================\n")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_models = executor.submit(sync_models, api, model_repo, commit_message)
        f_data = executor.submit(sync_data, api, dataset_repo, layers, sync_raw, commit_message)

        for future in as_completed([f_models, f_data]):
            try:
                res = future.result()
            except Exception as e:
                print(f"[Error in sync task]: {e}", file=sys.stderr)

    total_time = time.time() - t0
    print(f"\n✓ Concurrent sync completed in {total_time:.1f}s!")
    print(f"  Dataset: https://huggingface.co/datasets/{dataset_repo}")
    print(f"  Model:   https://huggingface.co/{model_repo}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or via HF_TOKEN)")
    parser.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO, help="Dataset repo ID")
    parser.add_argument("--model-repo", type=str, default=DEFAULT_MODEL_REPO, help="Model repo ID")
    parser.add_argument("--public", action="store_true", help="Make repos public (default: private)")
    parser.add_argument("--models", action="store_true", help="Sync models only")
    parser.add_argument("--data", action="store_true", help="Sync data layers only")
    parser.add_argument("--raw", action="store_true", help="Sync raw data archives (.tar.gz) only")
    parser.add_argument("--interim", action="store_true", help="Sync interim parquet data layer only")
    parser.add_argument("--both", action="store_true", help="Sync both models and structured data (default)")
    parser.add_argument("--all", action="store_true", help="Sync models, structured data, interim, and raw data archives")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["gold", "silver", "results", "archive", "bronze"],
        help="Specific data layers to sync (e.g. gold silver results interim)",
    )
    parser.add_argument("--commit-message", type=str, default="Sync from thesis pipeline")

    args = parser.parse_args()
    token = resolve_token(args.token)
    api = get_api(token)

    private = not args.public
    ensure_repos(api, args.dataset_repo, args.model_repo, private=private)

    if args.interim:
        sync_data(api, args.dataset_repo, ["interim"], sync_raw=False, commit_message=args.commit_message)
    elif args.raw and not args.models and not args.both and not args.all:
        sync_raw_archives(api, args.dataset_repo, args.commit_message)
    elif args.models and not args.data and not args.both and not args.all:
        sync_models(api, args.model_repo, args.commit_message)
    elif args.data and not args.models and not args.both and not args.all:
        sync_data(api, args.dataset_repo, args.layers, sync_raw=False, commit_message=args.commit_message)
    elif args.all:
        layers = list(dict.fromkeys(args.layers + ["interim"]))
        sync_both(api, args.dataset_repo, args.model_repo, layers, sync_raw=True, commit_message=args.commit_message)
    else:
        # Default: sync both models and structured data layers
        sync_both(api, args.dataset_repo, args.model_repo, args.layers, sync_raw=False, commit_message=args.commit_message)


if __name__ == "__main__":
    main()
