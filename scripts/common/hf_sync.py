"""Hugging Face synchronization utility for project data and models.

Manages Hugging Face repositories:
  - Dataset: https://huggingface.co/datasets/mrbungie/ai-disclosure-analysis
  - Model:   https://huggingface.co/mrbungie/ai-disclosure-analysis

Supports upload and download of models, structured data layers,
and compressed raw data archives (.tar.gz).

Usage:
    # Download models and datasets into models/ and data/:
    uv run python scripts/common/hf_sync.py --download

    # Download specific data layers (e.g. gold, silver, results):
    uv run python scripts/common/hf_sync.py --download-data --layers gold silver results

    # Sync models and structured data layers concurrently (default upload):
    uv run python scripts/common/hf_sync.py --both

    # Sync everything including compressed raw data archives (.tar.gz):
    uv run python scripts/common/hf_sync.py --all

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

from huggingface_hub import HfApi, snapshot_download

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

DEFAULT_DATASET_REPO = os.environ.get("HF_DATASET_REPO") or os.environ.get("HF_REPO") or "mrbungie/ai-disclosure-analysis"
DEFAULT_MODEL_REPO = os.environ.get("HF_MODEL_REPO") or os.environ.get("HF_REPO") or "mrbungie/ai-disclosure-analysis"


def resolve_token(arg_token: str | None = None, required: bool = True) -> str | None:
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
    if required:
        raise SystemExit(
            "Error: HF_TOKEN not found. Please set HF_TOKEN in your .env file or environment."
        )
    return None


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
- `ai_classification/`: Prefilter binary classifiers (TF-IDF + Ridge/Logistic Regression) detecting corporate AI discussion.
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
        delete_patterns="*",
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

    raw_categories = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    for cat_dir in sorted(raw_categories):
        archive_path = archive_dir / f"{cat_dir.name}.tar.gz"
        if not archive_path.exists():
            print(f"  [Data/Raw] Archiving {cat_dir.name} -> {archive_path.name}...")
            cmd = ["tar", "-czf", str(archive_path), "-C", str(raw_dir), cat_dir.name]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"    [Error] Failed to archive {cat_dir.name}: {res.stderr}", file=sys.stderr)
                continue

        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"  [Data/Raw] Uploading {archive_path.name} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=f"raw/{archive_path.name}",
            repo_id=dataset_repo,
            repo_type="dataset",
            commit_message=f"{commit_message} (raw/{archive_path.name})",
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
license: other
license_name: mixed-academic-research
license_link: LICENSE
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
- `bronze/`: Cleaned source manifests, paragraph partitions, market prices, and prefilter outputs.
- `archive/`: Metadata and universe references.

## Licensing & Redistribution

This dataset is compiled for academic and empirical research. Because it derives from multiple distinct primary sources, separate terms apply to different components:

1. **Research Code, Extraction Prompts, Schemas & Models (MIT License)**:
   All extraction prompts, schema definitions, classification model weights, aggregation scripts, and derivative empirical indexes/matrices developed as part of this thesis are released under the [MIT License](https://opensource.org/licenses/MIT).

2. **Derived Aggregated Panels & Research Outputs (CC-BY 4.0 / MIT)**:
   Summary panels in `gold/` and statistical results in `results/` are derivative statistical aggregations produced for academic research and are shared under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

3. **SEC Regulatory Filings (Public Domain / Fair Use)**:
   Paragraph extracts and metadata from SEC EDGAR filings (Forms 10-K, 10-Q, 8-K, DEF 14A) represent publicly accessible US regulatory disclosures and US government works not subject to domestic copyright (17 U.S.C. § 105), utilized here under fair use for scholarly research.

4. **Third-Party Commercial Content (Non-Redistributable)**:
   Raw transcripts and vendor financial time-series are proprietary to their respective providers. To respect third-party intellectual property and database terms, **raw source files and unaggregated third-party archives are NOT redistributed** in this repository. Pipeline replication scripts fetch or compute these locally.

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
            delete_patterns="*",
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


def download_models(model_repo: str, token: str | None = None) -> None:
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Models] Downloading from https://huggingface.co/{model_repo} to {models_dir}...")
    t0 = time.time()
    snapshot_download(
        repo_id=model_repo,
        repo_type="model",
        local_dir=str(models_dir),
        token=token,
        ignore_patterns=[".git*", "README.md"],
    )
    print(f"[Models] ✓ Download completed in {time.time() - t0:.1f}s!")


def download_data(
    dataset_repo: str,
    layers: list[str] | None = None,
    token: str | None = None,
) -> None:
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = None
    if layers:
        allow_patterns = [f"{layer}/**" for layer in layers] + ["sec_company_tickers.json"]
    print(
        f"\n[Data] Downloading from https://huggingface.co/datasets/{dataset_repo} "
        f"{f'(layers: {layers}) ' if layers else ''}to {data_dir}..."
    )
    t0 = time.time()
    snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        local_dir=str(data_dir),
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=[".git*", "README.md"],
    )
    print(f"[Data] ✓ Download completed in {time.time() - t0:.1f}s!")


def download_both(
    dataset_repo: str,
    model_repo: str,
    layers: list[str] | None = None,
    token: str | None = None,
) -> None:
    print("\n=======================================================")
    print(" HUGGING FACE DOWNLOAD (MODELS + DATASET)")
    print("=======================================================\n")
    t0 = time.time()
    download_models(model_repo, token=token)
    download_data(dataset_repo, layers=layers, token=token)
    print(f"\n✓ Download completed in {time.time() - t0:.1f}s!\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or via HF_TOKEN)")
    parser.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO, help="Dataset repo ID")
    parser.add_argument("--model-repo", type=str, default=DEFAULT_MODEL_REPO, help="Model repo ID")
    parser.add_argument("--public", action="store_true", help="Make repos public (default: private)")
    parser.add_argument("--download", action="store_true", help="Download both models and data from Hugging Face")
    parser.add_argument("--download-models", action="store_true", help="Download models only from Hugging Face")
    parser.add_argument("--download-data", action="store_true", help="Download data layers only from Hugging Face")
    parser.add_argument("--models", action="store_true", help="Sync models only (upload)")
    parser.add_argument("--data", action="store_true", help="Sync data layers only (upload)")
    parser.add_argument("--raw", action="store_true", help="Sync raw data archives (.tar.gz) only (upload)")
    parser.add_argument("--interim", action="store_true", help="Sync interim parquet data layer only (upload)")
    parser.add_argument("--both", action="store_true", help="Sync both models and structured data (default upload)")
    parser.add_argument("--all", action="store_true", help="Sync models, structured data, interim, and raw data archives (upload)")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["gold", "silver", "results", "archive", "bronze"],
        help="Specific data layers to sync or download (e.g. gold silver results interim)",
    )
    parser.add_argument("--commit-message", type=str, default="Sync from thesis pipeline")

    args = parser.parse_args()

    # Handle download operations
    is_download = args.download or args.download_models or args.download_data
    if is_download:
        token = resolve_token(args.token, required=False)
        if args.download_models:
            download_models(args.model_repo, token=token)
        elif args.download_data:
            download_data(args.dataset_repo, layers=args.layers, token=token)
        else:
            download_both(args.dataset_repo, args.model_repo, layers=args.layers, token=token)
        return

    token = resolve_token(args.token, required=True)
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
