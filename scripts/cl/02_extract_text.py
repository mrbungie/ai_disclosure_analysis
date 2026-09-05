"""
scripts/cl/02_extract_text.py — extracts paragraphs from every completed CMF
filing (Memoria Anual / Análisis Razonado) via scripts/common/pdf/, which is
country-agnostic: this script contributes the CMF manifest shape and the
`country_code="cl"` furniture profile, nothing else about PDF handling.

WHICH BACKEND runs is `pdf.backend` in configs/cl/config.yaml (or
--backend). The default is PaddleOCR-VL 1.6 with per-page triage, which
falls back to the CPU text-layer backend for pages that don't need a model
— see docs/analytics/pdf-backend-poc.md for the measured comparison and
scripts/common/pdf/pipeline.py's PageTriage for the routing rule.

CONCURRENCY IS THE BACKEND'S CALL, not this script's. The CPU backend is
pure-CPU and has always run across os.cpu_count() worker processes; a VLM
backend holds model weights on ONE GPU, and forking it 32 ways is an
out-of-memory crash rather than a speedup. So the pool is sized from
`backend.max_workers`, and a backend that reports 1 runs in-process,
loading the model once for the whole run instead of once per document.

Whole-document unit, not an Item like scripts/us/*/02_extract_sections.py —
the Memoria/Análisis Razonado are ALREADY the narrower disclosure documents
this project analyzes (see 01_fetch_filings.py's docstring: Estados
Financieros, Declaración de responsabilidad and Hechos Relevantes are
skipped at fetch time), so there's no further Item-level narrowing to do
here — every paragraph PyMuPDF's block segmentation finds is kept.

Different manifest shape than scripts/common/section_extraction.py's
run_extraction() expects (document_id/rut/nemo, not accession_number/
ticker/cik; no item_key or fixed target_items — one manifest row in, a
variable number of paragraph rows out) — so this is its own checkpoint/
part-file runner rather than a caller of that shared module. Same
conventions though (append-only globbable parts, safe_slug, a
parse_status column only ever set to "completed" once the paragraphs are
actually on disk), reused via import rather than duplicated.

Multi-part documents (large Memorias split across N pdf.gz files — see
01_fetch_filings.py's _save_pdf_parts) are handled by
scripts/common/pdf/render.py, which is where that logic belongs: a part is
its own standalone PDF with its own page numbering from 0, `page` is
renumbered densely across the whole document, and `source_part` is kept
alongside it so a page number can still be traced back to the physical
part that produced it. None of that is Chile-specific.

Usage:
    uv run python scripts/cl/02_extract_text.py [--limit N] [--comment "..."]
    uv run python scripts/cl/02_extract_text.py --backend pymupdf   # override config
    uv run python scripts/cl/02_extract_text.py --no-triage         # VLM on every page
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/cl/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # scripts/common/

import pipeline_logger
from section_extraction import safe_slug
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for scripts.common.pdf

from scripts.common.pdf import pipeline
from scripts.common.pdf.render import read_pdf_parts

CHECKPOINT_EVERY = 50

#: One PdfExtractor per worker process, built once by _init_worker and
#: reused for every document that worker handles. Module-level because
#: ProcessPoolExecutor's initializer has nowhere else to put per-worker
#: state — and building it per document would reload a VLM backend's 2.2 GB
#: of weights 2,100 times over a corpus run.
_EXTRACTOR = None


def _init_worker(extractor_kwargs: dict) -> None:
    global _EXTRACTOR
    _EXTRACTOR = pipeline.PdfExtractor(**extractor_kwargs)


def _process_one(row: dict) -> dict:
    """Runs in a worker process: extract one document's paragraphs, return
    a result dict. No shared state (df, logger file handle) touched here —
    same isolation contract as section_extraction.py's _process_one."""
    document_id = row["document_id"]
    local_path = row["local_path"]
    idx = row["_idx"]

    if not local_path:
        return {"idx": idx, "document_id": document_id, "parse_status": "failed: no local_path",
                "paragraphs": [], "error": None}

    part_paths = [p for p in local_path.split(";") if p]
    missing = [p for p in part_paths if not Path(p).exists()]
    if missing:
        return {"idx": idx, "document_id": document_id,
                "parse_status": f"failed: missing file(s) {missing}", "paragraphs": [], "error": None}

    try:
        # Multi-part documents (a large Memoria split across N .pdf.gz
        # files by 01_fetch_filings.py's _save_pdf_parts) are handled
        # inside the pipeline now: each part is its own standalone PDF with
        # its own page numbering from 0, and the pipeline renumbers `page`
        # densely across the whole document while keeping `source_part` so
        # a page number can still be traced back to the physical part.
        all_paragraphs = _EXTRACTOR.extract(read_pdf_parts(local_path))

        for para in all_paragraphs:
            para["document_id"] = document_id
            para["rut"] = row["rut"]
            para["nemo"] = row["nemo"]
            para["form_type"] = row["form_type"]
            para["filing_type"] = row["filing_type"]
            para["period_end_date"] = row["period_end_date"]

        parse_status = "completed" if all_paragraphs else "failed: no paragraphs extracted"
        return {"idx": idx, "document_id": document_id, "parse_status": parse_status,
                "paragraphs": all_paragraphs, "error": None}
    except Exception as e:
        return {"idx": idx, "document_id": document_id, "parse_status": f"failed: {e}",
                "paragraphs": [], "error": str(e)}


def _iter_results(jobs: list[dict], max_workers: int, extractor_kwargs: dict):
    """Yields one _process_one result per job, as each finishes.

    Two execution modes behind one generator, because the backend decides
    which one applies (see the module docstring):

    - max_workers > 1 (CPU backend): a real process pool, results in
      completion order, exactly what this script has always done.
    - max_workers == 1 (any VLM backend): run in THIS process, one document
      at a time. Not a one-process pool — that would still pickle every
      PDF's bytes across a pipe for no gain, and would put the model in a
      child process where a CUDA OOM kills the worker silently mid-run.

    A generator rather than a list so the inline path stays LAZY: a VLM
    run over the full corpus is hours long, and materializing every result
    up front would leave the progress bar at 0% for all of it and hold
    every document's paragraphs in memory at once.
    """
    if max_workers <= 1:
        _init_worker(extractor_kwargs)
        for job in jobs:
            yield _process_one(job)
        return

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker,
                             initargs=(extractor_kwargs,)) as executor:
        futures = [executor.submit(_process_one, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only the first N pending documents (smoke-testing)")
    parser.add_argument("--comment", default="run", help="Short label for this run, embedded in output filenames/columns.")
    parser.add_argument("--backend", default=None,
                        help="PDF backend, overriding configs/cl/config.yaml's pdf.backend "
                             "(pymupdf | paddleocr_vl | mineru | dots — see "
                             "scripts/common/pdf/backends/).")
    parser.add_argument("--no-triage", action="store_true",
                        help="Run the chosen backend on EVERY page instead of routing cheap "
                             "pages to the CPU backend. Slower; use to reproduce a clean "
                             "single-backend corpus.")
    args = parser.parse_args()

    with open("configs/cl/config.yaml") as f:
        config = yaml.safe_load(f)

    manifest_dir = Path(config["storage"]["interim_manifests"])
    manifest_path = manifest_dir / "filing_manifest.parquet"
    sections_dir = Path(config["storage"]["interim_sections"])
    sections_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        pipeline_logger.log_event(
            pipeline_step="cl_extract_text", level="ERROR",
            message=f"Manifest not found at {manifest_path}. Run scripts/cl/01_fetch_filings.py first.",
            log_dir=manifest_dir,
        )
        return

    df = pd.read_parquet(manifest_path)
    if "parse_status" not in df.columns:
        df["parse_status"] = "pending"
    df["parse_status"] = df["parse_status"].fillna("pending")

    pending = df[(df["download_status"] == "completed") & (df["parse_status"] == "pending")]
    if args.limit:
        pending = pending.head(args.limit)

    if len(pending) == 0:
        pipeline_logger.log_event(
            pipeline_step="cl_extract_text", level="INFO", message="No pending documents to extract.",
            log_dir=manifest_dir,
        )
        return

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_date = datetime.now().isoformat()
    comment_slug = safe_slug(args.comment)

    pipeline_logger.log_event(
        pipeline_step="cl_extract_text", level="INFO",
        message=f"Extracting paragraphs from {len(pending)} documents (run_id={run_id}, comment={args.comment})...",
        log_dir=manifest_dir,
    )

    pdf_config = config.get("pdf") or {}
    backend_name = args.backend or pdf_config.get("backend", "pymupdf")
    triage = None if args.no_triage else pipeline.triage_from_config(pdf_config)
    extractor_kwargs = {
        "backend": backend_name,
        "country_code": "cl",
        "triage": triage,
        "fallback_backend": pdf_config.get("fallback_backend", "pymupdf"),
        "backend_kwargs": {k: v for k, v in (
            ("dpi", pdf_config.get("dpi")),
            ("runtime", pdf_config.get("runtime")),
            ("server_url", pdf_config.get("server_url")),
        ) if v},
    }

    # The BACKEND decides the pool size, not this script — see the module
    # docstring. Built once here (cheaply: constructing a backend doesn't
    # load anything, that's what start() is for) purely to ask.
    probe = pipeline.PdfExtractor(**extractor_kwargs)
    max_workers = min(probe.max_workers, os.cpu_count() or 1)
    probe.close()

    pipeline_logger.log_event(
        pipeline_step="cl_extract_text", level="INFO",
        message=(f"PDF backend={backend_name} triage={'on' if triage else 'off'} "
                 f"workers={max_workers}"),
        log_dir=manifest_dir,
    )

    jobs = [{**row.to_dict(), "_idx": idx} for idx, row in pending.iterrows()]

    buffer = []
    part_num = 0
    completed_since_checkpoint = 0

    def flush_buffer():
        nonlocal buffer, part_num
        if not buffer:
            return
        part_df = pd.DataFrame(buffer)
        part_path = sections_dir / f"filing_paragraphs__run={run_id}__part={part_num:04d}__{comment_slug}.parquet"
        part_df["run_id"] = run_id
        part_df["run_date"] = run_date
        part_df["run_comment"] = args.comment
        part_df["part_num"] = part_num
        part_df["part_file"] = part_path.name
        part_df.to_parquet(part_path, index=False)
        pipeline_logger.log_event(
            pipeline_step="cl_extract_text", level="INFO",
            message=f"Flushed part {part_num:04d} ({part_path.name}): {len(part_df)} paragraph rows",
            log_dir=manifest_dir,
        )
        buffer = []
        part_num += 1

    for result in tqdm(_iter_results(jobs, max_workers, extractor_kwargs), total=len(jobs)):
        idx = result["idx"]

        df.at[idx, "parse_status"] = result["parse_status"]
        df.at[idx, "updated_at"] = datetime.now()
        buffer.extend(result["paragraphs"])

        if result["parse_status"] == "completed":
            pipeline_logger.log_event(
                pipeline_step="cl_extract_text", level="SUCCESS",
                message=f"Extracted {len(result['paragraphs'])} paragraphs",
                ticker=result["document_id"], log_dir=manifest_dir,
            )
        elif result["error"] is not None:
            pipeline_logger.log_event(
                pipeline_step="cl_extract_text", level="ERROR",
                message=f"Exception extracting: {result['error']}",
                ticker=result["document_id"], log_dir=manifest_dir,
            )
        else:
            pipeline_logger.log_event(
                pipeline_step="cl_extract_text", level="WARNING",
                message=result["parse_status"], ticker=result["document_id"], log_dir=manifest_dir,
            )

        completed_since_checkpoint += 1
        if completed_since_checkpoint >= CHECKPOINT_EVERY:
            # Paragraphs FIRST, then manifest — a row is only ever
            # marked "completed" once its text is actually on disk.
            flush_buffer()
            df.to_parquet(manifest_path, index=False)
            completed_since_checkpoint = 0

    flush_buffer()
    df.to_parquet(manifest_path, index=False)

    pipeline_logger.log_event(
        pipeline_step="cl_extract_text", level="SUCCESS",
        message=(
            f"Finished paragraph extraction run_id={run_id} ({args.comment}). "
            f"Parts written to {sections_dir}/filing_paragraphs__run={run_id}__part=*.parquet — "
            f"read the whole corpus with a wildcard glob over "
            f"{sections_dir}/filing_paragraphs__run=*__part=*.parquet"
        ),
        log_dir=manifest_dir,
    )


if __name__ == "__main__":
    main()
