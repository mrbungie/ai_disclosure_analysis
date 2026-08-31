"""
harness_fit.py — Shared machinery for the two-stage distillation cascade
(docs/distillation_map.html): a fixed seed screen and two frozen reference
classifiers (scope rule, six-dimension codebook) each distill into a cheap
program via the SAME outer loop:

    stratified sample -> LLM judge labels -> search-split search (fold-
    stability penalized) -> freeze artifact -> single holdout look

Stage 1 (detection) fits the prefilter against the scope rule, sampled from
the lexical seed screen's three strata (scripts/seed_screen.py). Stage 2
(classification) fits the six-dimension tagger against the codebook, sampled
as chunks built around stage 1's admitted paragraphs — and only after stage
1 is frozen. This module holds everything the two stages share, so the
cascade is one implementation with two instantiation sites, not two ad-hoc
copies.

Discipline encoded here:
  - Metrics come in two flavors: RAW (on the sample as drawn — internally
    consistent, but shaped by the sampling design) and WEIGHTED
    (inverse-probability weighted by each row's `sampling_weight` — an
    unbiased estimate of the population value). Sampling designs that
    oversample one stratum inflate raw recall; the weighted numbers undo
    that.
  - fold_stability_score() is deliberately NOT called cross-validation:
    nothing is trained per fold (a keyword list / boolean formula has no
    trainable parameters), and search candidates are mined from ALL of the
    search split, including every fold's test rows. The folds only measure
    how stable a candidate's score is across subsamples, penalizing
    candidates that win on one lucky fold. The single-look holdout is the
    only honest estimate.
  - The holdout lock: once a stage's holdout report exists, re-running
    against the same holdout is refused. Improving further requires a fresh
    labeled batch.
"""

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Harness candidates — two tasks, one pattern (harnesses/<task>/<name>/harness.py)
# ---------------------------------------------------------------------------
# Task 1 "detection":      classify(text) -> bool            (is_ai_related)
# Task 2 "classification": classify(text) -> dict[6 bools]   (dimension tags,
#                          defined on task 1's positives)
# Each candidate is one self-contained stdlib-only file the proposer can
# rewrite freely; harnesses/<task>/ACTIVE names the frozen candidate.

HARNESSES_DIR = Path("harnesses")


def paragraph_id(accession_number: str, section_name: str, para_idx: int, text: str) -> str:
    h = hashlib.sha256(f"{accession_number}|{section_name}|{para_idx}|{text}".encode("utf-8")).hexdigest()
    return h[:16]


_TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")
# A line made up ENTIRELY of one or more markdown links (plus stray
# whitespace/punctuation/digits around them) — markdownify's rendering of
# navigational chrome (TOC entries, exhibit-index links), never prose.
_PURE_LINK_LINE_RE = re.compile(r"^(\[[^\]]*\]\([^)]*\)[\s.,\d$]*)+$")


def _looks_like_table_row(line: str) -> bool:
    return line.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    return "-" in line and "|" in line and bool(_TABLE_SEPARATOR_RE.match(line))


def _is_pure_link_line(line: str) -> bool:
    return bool(_PURE_LINK_LINE_RE.match(line))


_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


def _has_no_letters(line: str) -> bool:
    """True for a line with no alphabetic character at all: a lone page
    number, a bare horizontal rule ('---'), a zero-width space, an unadorned
    bullet marker or footnote reference ('•', '(1)') — decoration or
    pagination artifacts, never prose. A bullet or heading WITH real words
    on it always has letters and is kept."""
    return not bool(_HAS_LETTER_RE.search(line))


_SENTENCE_END_RE = re.compile(r"[.!?:;\"'”’)\]]\s*$")


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text))


def markdown_to_paragraphs(text: str) -> list[str]:
    """Split markdown section text (markdownify's HTML->markdown output —
    see scripts/04_extract_sections.py) into paragraph units.

    The source has no blank lines between blocks — markdownify emits one
    line per original HTML block element, so a real block-level markdown
    parser (which needs blank lines to separate paragraphs under CommonMark)
    just collapses everything into a handful of giant paragraphs instead.
    One line IS already the right paragraph unit here: a heading, a body
    sentence, and a bullet each already occupy exactly one line. Two things
    must NOT be kept as paragraphs even though they're each one line:
      - GFM pipe tables — each row (header, `| --- | --- |` separator, and
        data rows) is its own line, and naively keeping them individually
        made ~58% of the corpus's "paragraphs" degenerate table-row
        fragments with no prose content.
      - Pure navigational links ("[Table of Contents](#anchor)", exhibit-
        index entries) — ~4% of lines, chrome rather than disclosure text.
      - Lines with no letters at all — page numbers, bare '---' rules,
        zero-width spaces, unadorned bullet/footnote markers.
    So: keep every line as its own paragraph, except drop a detected table
    block (rows included), pure-link lines, and letterless lines.

    One more wrinkle: these EDGAR-to-HTML-to-markdown conversions carry
    page-break furniture (a page number, a "---" rule, a "[Table of
    Contents]" link — exactly the three junk kinds above, back to back) that
    the original HTML injected in the MIDDLE of a paragraph at a page
    boundary, not between two real paragraphs. Dropping that furniture
    without reconnecting the prose on either side leaves a sentence
    truncated mid-clause (ending in a comma, "and", etc.) as one paragraph
    and its continuation as another, decontextualized one. So: whenever at
    least one line was skipped as junk right before the next kept line, and
    the last kept paragraph does not already end on sentence-final
    punctuation, merge the next line onto it instead of starting a new
    paragraph."""
    lines = [line.strip() for line in text.split("\n")]
    paragraphs: list[str] = []
    just_skipped_junk = False
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line:
            i += 1
            continue
        # Table-block detection must run before the letterless check: a
        # table's header row is very often itself letterless (blank cells,
        # "|  |  |  |"), so checking letterless first would eat just that
        # one line, break the header+separator pattern the block detector
        # looks for, and let every data row after it leak through as
        # individual "paragraphs" again.
        if _looks_like_table_row(line) and i + 1 < n and _is_table_separator(lines[i + 1]):
            i += 2
            while i < n and _looks_like_table_row(lines[i]):
                i += 1
            just_skipped_junk = True
            continue
        if _is_pure_link_line(line) or _has_no_letters(line):
            i += 1
            just_skipped_junk = True
            continue
        if just_skipped_junk and paragraphs and not _ends_sentence(paragraphs[-1]):
            paragraphs[-1] = f"{paragraphs[-1]} {line}"
        else:
            paragraphs.append(line)
        just_skipped_junk = False
        i += 1
    return paragraphs


def flatten_corpus_paragraphs(config: dict) -> pd.DataFrame:
    """One row per paragraph across the entire parsed corpus: paragraph_id,
    accession_number, ticker, industry_group, filing_date, section_name,
    paragraph_text. Shared by the seed screen and both eval-set builders so
    every stage samples from the same paragraph universe. Paragraphs are
    markdown blocks (markdown_to_paragraphs), not raw text lines."""
    manifest = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet")
    sections = pd.read_parquet(Path(config["paths"]["interim_sections"]) / "filing_sections.parquet")
    firm_universe = pd.read_parquet(Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet")
    ticker_industry = dict(zip(firm_universe["ticker"], firm_universe["industry_group"]))

    completed_acc = set(manifest.loc[manifest["parse_status"] == "completed", "accession_number"])
    sections = sections[sections["accession_number"].isin(completed_acc)]

    rows = []
    for _, row in sections.iterrows():
        paragraphs = markdown_to_paragraphs(row["section_text"])
        for i, p in enumerate(paragraphs):
            rows.append({
                "paragraph_id": paragraph_id(row["accession_number"], row["section_name"], i, p),
                "accession_number": row["accession_number"],
                "ticker": row["ticker"],
                "industry_group": ticker_industry.get(row["ticker"]),
                "filing_date": row["filing_date"],
                "section_name": row["section_name"],
                "paragraph_index": i,
                "paragraph_text": p,
            })
    return pd.DataFrame(rows)


TASK_LABELS = {
    "detection": ["is_ai_related"],
    "classification": ["is_substantive", "is_promotional", "is_risk_related",
                       "is_governance_related", "is_use_case_specific", "is_quantified"],
}
LABEL_FIELDS = TASK_LABELS["detection"] + TASK_LABELS["classification"]


def load_candidate(task: str, name: str):
    """Import classify() from harnesses/<task>/<name>/harness.py."""
    import importlib.util
    path = HARNESSES_DIR / task / name / "harness.py"
    if not path.exists():
        raise FileNotFoundError(f"no candidate at {path}")
    spec = importlib.util.spec_from_file_location(f"harness_{task}_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify


def active_candidate(task: str) -> str:
    return (HARNESSES_DIR / task / "ACTIVE").read_text().strip()


def set_active_candidate(task: str, name: str) -> None:
    (HARNESSES_DIR / task / "ACTIVE").write_text(name + "\n")


# ---------------------------------------------------------------------------
# Regex harness primitives (shared with scripts 05-06)
# ---------------------------------------------------------------------------


def build_regex(keywords: list[str]) -> re.Pattern:
    patterns = [r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b" for kw in keywords]
    return re.compile("|".join(patterns), re.IGNORECASE)


def clean_false_positives(text: str, false_positives: list[str]) -> str:
    for fp in false_positives:
        text = re.sub(r"\b" + re.escape(fp) + r"\b", "", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Metrics — raw and inverse-probability weighted
# ---------------------------------------------------------------------------


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray,
                        weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """Precision/recall/F1. With `weights` (each row's inverse sampling
    probability), the confusion counts become weighted sums, giving a
    population estimate instead of a sample-design-shaped one."""
    if weights is None:
        weights = np.ones(len(y_true))
    tp = float(np.sum(weights[y_true & y_pred]))
    fp = float(np.sum(weights[~y_true & y_pred]))
    fn = float(np.sum(weights[y_true & ~y_pred]))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    return precision_recall_f1(y_true, y_pred, weights)[2]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                      weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """Sensitivity, specificity, and their mean (balanced accuracy). Base
    rates differ enough across the six dimensions that raw agreement can be
    trivially high by predicting the majority class — balanced accuracy is
    the map's operational measure of per-label fidelity, not F1."""
    if weights is None:
        weights = np.ones(len(y_true))
    tp = float(np.sum(weights[y_true & y_pred]))
    fn = float(np.sum(weights[y_true & ~y_pred]))
    tn = float(np.sum(weights[~y_true & ~y_pred]))
    fp = float(np.sum(weights[~y_true & y_pred]))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return sensitivity, specificity, (sensitivity + specificity) / 2


def metrics_block(label: str, y_true: np.ndarray, y_pred: np.ndarray,
                  weights: np.ndarray | None) -> list[str]:
    """Report lines with the raw and (when weights exist) weighted metrics
    side by side, so no report ever shows the design-inflated number alone."""
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    lines = [f"{label}  [raw, sample as drawn]      F1={f1:.3f}  P={p:.3f}  R={r:.3f}"]
    if weights is not None:
        wp, wr, wf1 = precision_recall_f1(y_true, y_pred, weights)
        lines.append(f"{' ' * len(label)}  [weighted, population est.]  F1={wf1:.3f}  P={wp:.3f}  R={wr:.3f}")
    return lines


def sampling_weights(df: pd.DataFrame) -> np.ndarray | None:
    """Row weights from the `sampling_weight` column written at sample time,
    or None (with a warning) for batches labeled before weights existed."""
    if "sampling_weight" not in df.columns:
        print("NOTE: no sampling_weight column (batch predates stratum reweighting) — "
              "only raw sample metrics are available; they overstate recall/F1 whenever "
              "the sampling design oversampled the flagged stratum.")
        return None
    return df["sampling_weight"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Splits and fold-stability scoring
# ---------------------------------------------------------------------------


def stratified_split(df: pd.DataFrame, label_col: str, dev_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Proportional dev/holdout split, stratified by label so both sides have
    a representative positive rate."""
    rng = np.random.default_rng(seed)
    dev_parts, holdout_parts = [], []
    for _, group in df.groupby(label_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_dev = round(len(idx) * dev_frac)
        dev_parts.append(group.loc[idx[:n_dev]])
        holdout_parts.append(group.loc[idx[n_dev:]])
    return (pd.concat(dev_parts).sample(frac=1, random_state=seed),
            pd.concat(holdout_parts).sample(frac=1, random_state=seed))


def stability_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Stratified fold test-indices for fold-stability scoring. Returns only
    the per-fold evaluation indices — there ARE no train indices, because
    nothing is trained per fold (see module docstring)."""
    rng = np.random.default_rng(seed)
    folds_idx: list[list[int]] = [[] for _ in range(n_folds)]
    for label in (True, False):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for i, chunk in enumerate(np.array_split(idx, n_folds)):
            folds_idx[i].extend(chunk.tolist())
    return [np.array(sorted(f)) for f in folds_idx]


def fold_stability_score(y: np.ndarray, pred: np.ndarray, folds: list[np.ndarray]) -> tuple[float, float]:
    """Mean and std of F1 across fold subsamples for a FIXED predictor.

    NOT cross-validation: the predictor has no per-fold training, and the
    search that proposes candidates sees all of dev. This is a stability /
    variance check that penalizes candidates riding one lucky subsample; the
    dev mean is optimistic by construction and never reported as the final
    number — the single-look holdout is."""
    scores = [f1_score(y[idx], pred[idx]) for idx in folds]
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Search-trace and holdout discipline
# ---------------------------------------------------------------------------


def flush_trace(trace_path: Path, meta: dict, rounds: list[dict]) -> None:
    """Write the full search trace (every candidate evaluated, every round)
    to disk. Called after every round so an interrupted run stays auditable —
    the cycle's equivalent of the meta-harness paper's filesystem of prior
    candidates."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w") as f:
        json.dump({**meta, "rounds": rounds}, f, indent=2)


def refuse_if_holdout_spent(holdout_report_path: Path, resample_hint: str) -> bool:
    """True (and prints the refusal) if this cycle's holdout was already
    evaluated once. A disappointing holdout is reported as-is; 'trying again'
    against the same holdout is what this blocks."""
    if holdout_report_path.exists():
        print(f"Error: {holdout_report_path} already exists — this cycle's holdout has been "
              f"evaluated once and is spent. {resample_hint}")
        return True
    return False


# ---------------------------------------------------------------------------
# LLM judge (outer loop)
# ---------------------------------------------------------------------------


def build_batch_output_type(item_type):
    """Wrap a single-item Pydantic output type into {items: [item_type +
    index]} so one judge call can score several excerpts at once. Batching
    (not concurrency) is what actually pays off on some NIM-hosted models:
    benchmarked at ~2.9s/item and 100% success for batch_size=5 vs ~5.9s/item
    serially and ~50% failures (500s / schema-retry exhaustion) under
    concurrency=5 — the endpoint chokes on parallel connections, not on
    longer single requests."""
    from pydantic import Field, create_model
    indexed_item = create_model(
        f"Indexed{item_type.__name__}",
        __base__=item_type,
        index=(int, Field(description="the excerpt's number from the input, 0-based, matching exactly")),
    )
    return create_model(f"Batch{item_type.__name__}", items=(list[indexed_item], ...))


def build_judge(output_type, system_prompt: str):
    """Pydantic-AI agent for the outer-loop judge. Reads LLM_JUDGE_* env vars;
    the judge should be a different model family from anything the harness
    approximates, so it isn't grading a close relative of itself. `output_type`
    is the single-item schema; judge calls are always batched (see
    build_batch_output_type, judge_batch).

    `LLM_JUDGE_PROVIDER=google` selects the Gemini API (GOOGLE_API_KEY,
    LLM_JUDGE_MODEL e.g. gemini-3.1-flash-lite); anything else (the default)
    uses an OpenAI-compatible endpoint (LLM_JUDGE_API_KEY/BASE_URL/MODEL —
    this is how NVIDIA NIM models are reached). Switched to Gemini
    2026-08-31 after nvidia/nemotron-3-ultra-550b-a55b's free-tier NIM
    endpoint proved unstable (sustained 429/500/timeout/404 under sequential
    load, ~2.9s/item best case); Gemini Flash Lite benchmarked ~0.28s/item,
    zero errors, and 100% scope-label agreement with nemotron on a 20-item
    sample including the 3 positives seen so far."""
    batch_type = build_batch_output_type(output_type)
    batched_prompt = (
        f"{system_prompt}\n\nYou will receive several numbered excerpts in one message, each as "
        f"'[N] <text>'. Return exactly one item per excerpt, in `items`, with each item's `index` "
        f"field set to match its excerpt's number N exactly."
    )

    if os.environ.get("LLM_JUDGE_PROVIDER", "").lower() == "google":
        from pydantic_ai import Agent
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set. Configure it in .env.")
        model_name = os.environ.get("LLM_JUDGE_MODEL", "gemini-3.1-flash-lite")
        provider = GoogleProvider(api_key=api_key)
        model = GoogleModel(model_name, provider=provider)
        return Agent(model, output_type=batch_type, retries=3, system_prompt=batched_prompt)

    import httpx
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = os.environ.get("LLM_JUDGE_API_KEY")
    if not api_key:
        raise ValueError("LLM_JUDGE_API_KEY is not set. Configure it in .env.")
    base_url = os.environ.get("LLM_JUDGE_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")

    # Some OpenAI-"compatible" endpoints stall a connection with no response
    # and no error; the OpenAI SDK's default httpx timeout is long enough
    # (600s) that a stalled call otherwise looks indistinguishable from a
    # slow-but-working run for many minutes. A short explicit timeout turns
    # that into a fast, retried failure instead (see judge_one).
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0))
    provider = OpenAIProvider(base_url=base_url, api_key=api_key, http_client=http_client)
    # Several OpenAI-"compatible" NIM-hosted models 400 on strict tool-definition
    # mode (extra_forbidden on tools.0.function.strict) — disable it so the judge
    # isn't locked to the handful of models that happen to support it.
    profile = OpenAIModelProfile(openai_supports_strict_tool_definition=False)
    model = OpenAIChatModel(model_name, provider=provider, profile=profile)
    return Agent(model, output_type=batch_type, retries=3, system_prompt=batched_prompt)


async def judge_one(agent, prompt: str, max_retries: int = 5) -> dict | None:
    """One judge call (a batch of excerpts numbered into one prompt — see
    build_judge/judge_batch) with backoff on transient failures. The FULL
    text of every excerpt is judged — no truncation: a label for a truncated
    excerpt is a label for a different document than the harness will see."""
    retry_delay = 5.0
    for attempt in range(max_retries):
        try:
            result = await agent.run(prompt)
            return result.output.model_dump(mode="json")
        except Exception as e:
            error_str = str(e)
            is_transient = ("429" in error_str or "rate limit" in error_str.lower()
                            or "500" in error_str or "502" in error_str or "503" in error_str
                            or "504" in error_str or "timed out" in error_str.lower()
                            or "timeout" in type(e).__name__.lower()
                            or getattr(e, "status_code", None) in (429, 500, 502, 503, 504))
            if is_transient and attempt < max_retries - 1:
                print(f"  Transient judge error, retrying in {retry_delay}s: {e}", flush=True)
                await asyncio.sleep(retry_delay)
                retry_delay *= 2.0
                continue
            print(f"  Judge error: {e}", flush=True)
            return None
    return None


async def judge_batch(agent, to_label: pd.DataFrame, text_col: str, label_fields: list[str],
                      existing_records: list[dict], out_path: Path,
                      batch_size: int, delay: float) -> pd.DataFrame:
    """Sequential batched judge labeling: `batch_size` excerpts numbered into
    one prompt per call (see build_judge/build_batch_output_type) — measured
    far faster and more reliable than one-item-per-call concurrency, which
    overloads some NIM-hosted models into 500s/schema failures.

    Checkpoints `out_path` after every batch, so a run interrupted or killed
    partway through — a crash, a rate limit exhausting retries, Ctrl-C — only
    loses the batch in flight. Resumability across separate invocations is
    the caller's job: `existing_records` should already exclude ids present
    in a prior checkpoint (see build_eval_set._label_stage), so re-running
    this function on the same sample only labels what's still missing.

    A batch can fail as a whole even when only one excerpt in it is the
    problem (a garbage table-fragment paragraph derailing the model's
    structured output breaks the whole batch's JSON, not just its own item).
    Since the sample is fixed, the same items would land in the same batch
    on every resumed run and fail identically forever — so a failed batch is
    retried once item-by-item (batch_size=1) before anything in it is given
    up on."""
    from tqdm import tqdm

    records = existing_records
    rows = list(to_label.iterrows())
    bar = tqdm(total=len(rows), desc="labeling", file=sys.stdout)

    async def label_one_row(row) -> None:
        prompt = f"[0] {row[text_col]}"
        result = await judge_one(agent, prompt)
        if not result or not result["items"]:
            print(f"  WARNING: single-item retry also failed for one row — dropped, "
                  f"will be picked up by a resumed run.", flush=True)
            return
        item = result["items"][0]
        record = row.to_dict()
        for field in label_fields:
            record[f"llm_{field}"] = item[field]
        records.append(record)

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompt = "\n\n".join(f"[{i}] {row[text_col]}" for i, (_, row) in enumerate(chunk))
        result = await judge_one(agent, prompt)
        if result is not None:
            by_index = {item["index"]: item for item in result["items"]}
            for i, (_, row) in enumerate(chunk):
                item = by_index.get(i)
                if item is None:
                    print(f"  WARNING: no item at index {i} in a batch response — dropped, "
                          f"will be picked up by a resumed run.", flush=True)
                    continue
                record = row.to_dict()
                for field in label_fields:
                    record[f"llm_{field}"] = item[field]
                records.append(record)
        elif len(chunk) > 1:
            print(f"  Batch of {len(chunk)} failed — retrying item-by-item.", flush=True)
            for _, row in chunk:
                await label_one_row(row)
                if delay > 0:
                    await asyncio.sleep(delay)
        else:
            await label_one_row(chunk[0][1])
        bar.update(len(chunk))
        pd.DataFrame(records).to_parquet(out_path, index=False)
        if delay > 0:
            await asyncio.sleep(delay)
    bar.close()

    labeled_df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(out_path, index=False)
    return labeled_df


# ---------------------------------------------------------------------------
# Stratified sampling with recorded weights
# ---------------------------------------------------------------------------


def proportional_by_group(df: pd.DataFrame, group_col: str, n: int, seed: int,
                          min_per_group: int = 1) -> pd.DataFrame:
    """Proportional allocation by group_col (frac = n/N within each group),
    with a floor so small groups aren't zeroed out."""
    df = df.copy()
    frac = min(1.0, n / len(df)) if len(df) else 0.0

    def sample_group(g):
        target = max(min_per_group, round(len(g) * frac))
        return g.sample(min(target, len(g)), random_state=seed)

    sampled = df.groupby(group_col, group_keys=False).apply(sample_group, include_groups=False)
    sampled = df.loc[sampled.index]
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)
    return sampled


def attach_sampling_weights(sample: pd.DataFrame, population: pd.DataFrame,
                            strata_cols: list[str]) -> pd.DataFrame:
    """Attach `sampling_weight` = N_stratum / n_stratum for each stratum cell
    actually drawn from. Recorded AT SAMPLE TIME so every later metric can be
    inverse-probability weighted back to the population — the fix for
    balanced-by-design draws overstating recall/F1."""
    pop_counts = population.groupby(strata_cols, dropna=False).size()
    samp_counts = sample.groupby(strata_cols, dropna=False).size()
    weights = (pop_counts / samp_counts).rename("sampling_weight").reset_index()
    out = sample.merge(weights, on=strata_cols, how="left")
    assert out["sampling_weight"].notna().all(), "every sampled row must fall in a counted stratum"
    return out


def weighted_search_sample(df: pd.DataFrame, stratum_col: str, hit_count_col: str,
                           n: int, seed: int,
                           stratum_fracs: dict[str, float] | None = None) -> pd.DataFrame:
    """Stage-1 search draw, deliberately skewed toward hard cases (map §2):
    weak hits (a single keyword hit, within the hit stratum) and no-hit
    paragraphs inside filings the screen hit elsewhere, over the
    overwhelmingly-easy clean-filing stratum. Not a probability sample — the
    search sample is spent freely, never used for a population estimate.

    Fixed per-stratum quotas (`stratum_fracs`), not a multiplicative weight
    on the raw population: the hit stratum is ~0.5% of the corpus, so any
    weight multiplier small enough to leave the other two strata sane gets
    swamped by the ~140x population-size gap and barely moves the draw (a 3x
    weight produced 5/560 hit rows — indistinguishable from unweighted
    prevalence). A quota guarantees real representation regardless of how
    skewed the underlying strata sizes are."""
    if stratum_fracs is None:
        stratum_fracs = {"hit": 0.40, "no_hit_filing_hits": 0.40, "no_hit_filing_clean": 0.20}

    parts = []
    for stratum, frac in stratum_fracs.items():
        pool = df[df[stratum_col] == stratum]
        target = min(round(n * frac), len(pool))
        if target == 0:
            continue
        if stratum == "hit":
            # Within hit, favor weak (single-keyword) hits — the ambiguous,
            # informative cases — over strong multi-keyword ones.
            weights = 1.0 / pool[hit_count_col].clip(lower=1)
            parts.append(pool.sample(n=target, weights=weights, random_state=seed))
        else:
            parts.append(pool.sample(n=target, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def stratified_holdout_with_coverage(df: pd.DataFrame, population: pd.DataFrame, stratum_col: str,
                                     cross_cols: list[str], n: int, seed: int,
                                     min_per_stratum: int = 10) -> pd.DataFrame:
    """Probability draw, proportional within stratum_col x cross_cols cells,
    with a floor per stratum_col value so every seed-screen stratum —
    including paragraphs in filings the screen never flagged at all — has
    guaranteed holdout coverage (map §2, §7). Records `sampling_weight`
    against `population` for inverse-probability-weighted corpus estimates."""
    df = df.copy()
    df["_cross"] = list(zip(*[df[c].astype(str) for c in cross_cols])) if cross_cols else "_all"
    n_strata = max(df[stratum_col].nunique(), 1)
    per_stratum = max(min_per_stratum, n // n_strata)
    parts = [proportional_by_group(g, "_cross", per_stratum, seed) for _, g in df.groupby(stratum_col)]
    sample = pd.concat(parts, ignore_index=True).drop(columns="_cross")
    return attach_sampling_weights(sample, population, [stratum_col] + cross_cols)


def select_stage1_candidate(candidate_scores: dict[str, dict], recall_floor: float) -> str | None:
    """Stage-1's constrained objective (map §3): minimize candidate volume
    subject to weighted search recall >= recall_floor. `candidate_scores` is
    {name: {"recall_weighted": float, "volume_weighted": float}}. Returns
    the eligible candidate with the smallest volume, or None if no candidate
    clears the floor — unconstrained recall has a trivial optimum (admit
    everything), so a candidate is never selected on volume alone."""
    eligible = {name: v for name, v in candidate_scores.items() if v["recall_weighted"] >= recall_floor}
    if not eligible:
        return None
    return min(eligible, key=lambda name: eligible[name]["volume_weighted"])


# ---------------------------------------------------------------------------
# Chunking — stage-2 labeling unit only (map §3: population = "chunks built
# around admitted paragraphs"); tags are broadcast back to member paragraphs
# so downstream analysis stays paragraph-indexed.
# ---------------------------------------------------------------------------


def build_chunks(paragraphs: pd.DataFrame, admit_col: str,
                 window: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge +/-window paragraph neighborhoods around admitted paragraphs
    into chunks (port of the original 06_chunk_candidates.py windowing/merge
    logic, generalized from a fixed keyword regex to any admit predicate).

    `paragraphs` must contain every paragraph in each (accession_number,
    section_name) group, admitted or not, so a window can borrow context
    from non-admitted neighbors — but only admitted paragraphs seed a window
    and only admitted paragraphs are members whose tags get broadcast back;
    non-admitted neighbors contribute text only.

    Returns (chunks, membership):
      chunks: chunk_id, accession_number, ticker, filing_date, section_name,
              chunk_text, member_paragraph_ids (list, admitted only)
      membership: one row per admitted paragraph, paragraph_id -> chunk_id
    """
    chunk_rows: list[dict] = []
    membership_rows: list[dict] = []
    for (acc, sec), group in paragraphs.groupby(["accession_number", "section_name"], sort=False):
        g = group.sort_values("paragraph_index").reset_index(drop=True)
        admitted_positions = g.index[g[admit_col].astype(bool)].tolist()
        if not admitted_positions:
            continue
        windows = [(max(0, p - window), min(len(g) - 1, p + window)) for p in admitted_positions]
        merged: list[list[int]] = []
        for start, end in sorted(windows):
            if not merged or merged[-1][1] < start:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        for start, end in merged:
            block = g.iloc[start:end + 1]
            chunk_text = "\n\n".join(block["paragraph_text"])
            chunk_id = hashlib.sha256(f"{acc}|{sec}|{start}|{end}".encode("utf-8")).hexdigest()[:16]
            members = block.loc[block[admit_col].astype(bool), "paragraph_id"].tolist()
            chunk_rows.append({
                "chunk_id": chunk_id, "accession_number": acc, "ticker": block["ticker"].iloc[0],
                "filing_date": block["filing_date"].iloc[0], "section_name": sec,
                "chunk_text": chunk_text, "member_paragraph_ids": members,
            })
            for pid in members:
                membership_rows.append({"paragraph_id": pid, "chunk_id": chunk_id})
    chunks = pd.DataFrame(chunk_rows)
    membership = pd.DataFrame(membership_rows)
    return chunks, membership


# ---------------------------------------------------------------------------
# Human agreement sample (judge validation)
# ---------------------------------------------------------------------------


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters."""
    a = a.astype(bool)
    b = b.astype(bool)
    po = float(np.mean(a == b))
    p_yes = float(np.mean(a)) * float(np.mean(b))
    p_no = (1 - float(np.mean(a))) * (1 - float(np.mean(b)))
    pe = p_yes + p_no
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def export_agreement_workbook(labeled: pd.DataFrame, id_col: str, text_col: str,
                              label_fields: list[str], out_path: Path,
                              n: int, seed: int, priority: pd.Series | None = None,
                              exclude_ids: set | None = None) -> None:
    """Excel workbook for hand-labeling a subsample of judge-labeled rows.

    Sheet 'label_me': id, text, and one EMPTY TRUE/FALSE column per label
    field — the LLM's labels are deliberately NOT on this sheet, so the human
    rater isn't anchored. Sheet 'llm_labels' holds the judge's labels for the
    same ids; score_agreement() joins the two after the human sheet is filled.

    `exclude_ids` drops rows already answered in a prior audit (see
    load_ground_truth_ids/save_ground_truth) so every export asks about NEW
    cases instead of re-drawing ones already hand-labeled.

    `priority` (index-aligned to `labeled`, higher = more worth auditing)
    picks the TOP-priority rows within each class instead of a uniform
    random draw — e.g. cases where a cheap independent signal (a keyword
    screen) disagrees with the judge are far more informative to a human
    auditor than an easy case both would obviously agree on. Falls back to
    uniform random per class when not given.

    The subsample is drawn BALANCED on the first label field (up to n/2 per
    class): kappa needs disagreement opportunities in both classes, and a
    proportional draw of a rare-positive population would leave a handful of
    positives and a uselessly wide kappa. This deliberately makes the raw
    agreement number non-representative of the population — kappa corrects
    for marginals, raw agreement here is descriptive only. Degenerately short
    texts (< 40 chars) are excluded: they carry no signal for either rater."""
    rng_seed = seed
    llm_cols = [f"llm_{f}" for f in label_fields]
    strat = f"llm_{label_fields[0]}"
    pool = labeled[labeled[text_col].str.len() >= 40]
    if exclude_ids:
        pool = pool[~pool[id_col].isin(exclude_ids)]

    def pick(group: pd.DataFrame, k: int) -> pd.DataFrame:
        k = min(k, len(group))
        if priority is None:
            return group.sample(k, random_state=rng_seed)
        return group.loc[priority.reindex(group.index).fillna(0).sort_values(ascending=False).index[:k]]

    parts = [pick(group, n // 2) for _, group in pool.groupby(pool[strat].astype(bool))]
    sub = pd.concat(parts)
    if len(sub) < n:  # one class exhausted — top up from the other
        rest = pool.drop(sub.index)
        sub = pd.concat([sub, pick(rest, n - len(sub))])
    sub = sub.sample(frac=1, random_state=rng_seed)

    human = sub[[id_col, text_col]].copy()
    for f in label_fields:
        human[f"your_{f}"] = ""
    llm = sub[[id_col] + llm_cols].copy()

    instructions = pd.DataFrame({
        "How to fill this in": [
            "1. Work ONLY on the 'label_me' sheet. Do not open 'llm_labels' until you are done",
            "   (it holds the LLM judge's answers; peeking anchors your labels and voids the check).",
            f"2. For each row, read the text and fill every your_* column with 1 (yes) or 0 (no).",
            "3. Save the file in place, then run:  make agreement-score  (or scripts/agreement_check.py --score)",
            "   to get raw agreement and Cohen's kappa per label.",
            "4. Label what the TEXT says, not what you know about the company.",
        ]
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="instructions", index=False)
        human.to_excel(writer, sheet_name="label_me", index=False)
        llm.to_excel(writer, sheet_name="llm_labels", index=False)
        from openpyxl.styles import Alignment
        ws = writer.book["label_me"]
        ws.column_dimensions["B"].width = 120
        for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        writer.book["instructions"].column_dimensions["A"].width = 110
    print(f"Agreement workbook ({len(sub)} rows) -> {out_path}")


def _to_bool_one(v) -> bool:
    """Accepts 0/1 (as typed into Excel, which stores them as numbers, not
    strings — str(1.0) == '1.0', not '1', so a pure string-set check would
    silently misread a numeric 1) as well as TRUE/FALSE/yes/no text."""
    try:
        f = float(v)
        return f != 0.0 and not pd.isna(f)
    except (TypeError, ValueError):
        return str(v).strip().lower() in {"true", "yes", "y", "t"}


def _to_bool_series(series: pd.Series) -> pd.Series:
    return series.map(_to_bool_one)


def load_ground_truth_ids(store_path: Path, id_col: str) -> set:
    """Ids already hand-labeled in a prior audit round — never re-drawn into
    a future workbook, and never lost even if the workbook itself, or the
    eval set it was drawn from, gets regenerated (e.g. a corpus/paragraph
    definition fix invalidates and rebuilds the eval set)."""
    if not store_path.exists():
        return set()
    return set(pd.read_parquet(store_path)[id_col])


def save_ground_truth(store_path: Path, id_col: str, filled: pd.DataFrame, label_fields: list[str]) -> int:
    """Append newly hand-labeled rows (id + one bool column per label field)
    to the persistent ground-truth store, deduping by id_col — existing
    entries are kept as-is (a human label is never silently overwritten by a
    later run touching the same id). Returns how many NEW ids were added."""
    new_rows = filled[[id_col] + label_fields].copy()
    if store_path.exists():
        existing = pd.read_parquet(store_path)
        new_only = new_rows[~new_rows[id_col].isin(set(existing[id_col]))]
        combined = pd.concat([existing, new_only], ignore_index=True)
    else:
        new_only = new_rows
        combined = new_rows
    store_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(store_path, index=False)
    return len(new_only)


def score_agreement(workbook_path: Path, id_col: str, label_fields: list[str],
                    ground_truth_path: Path | None = None) -> list[str]:
    """Score a filled-in agreement workbook: raw agreement + Cohen's kappa per
    label field. Returns report lines (also printed). When `ground_truth_path`
    is given, every filled human label is also persisted there (see
    save_ground_truth) so it's never lost or re-asked."""
    human = pd.read_excel(workbook_path, sheet_name="label_me")
    llm = pd.read_excel(workbook_path, sheet_name="llm_labels")
    merged = human.merge(llm, on=id_col, how="inner")

    lines = [f"Judge agreement check — {workbook_path.name}, n={len(merged)}"]
    fully_filled_mask = pd.Series(True, index=merged.index)
    for f in label_fields:
        col = merged[f"your_{f}"]
        filled = col.notna() & (col.astype(str).str.strip() != "")
        fully_filled_mask &= filled
        if filled.sum() == 0:
            lines.append(f"  {f}: no human labels filled in yet")
            continue
        sub = merged[filled]
        h = _to_bool_series(sub[f"your_{f}"]).to_numpy()
        m = sub[f"llm_{f}"].astype(bool).to_numpy()
        kappa = cohens_kappa(h, m)
        agree = float(np.mean(h == m))
        lines.append(f"  {f}: n={len(sub)}  raw agreement={agree:.3f}  Cohen's kappa={kappa:.3f}")

    if ground_truth_path is not None and fully_filled_mask.any():
        to_save = merged[fully_filled_mask].copy()
        for f in label_fields:
            to_save[f] = _to_bool_series(to_save[f"your_{f}"])
        n_new = save_ground_truth(ground_truth_path, id_col, to_save, label_fields)
        lines.append(f"  Ground truth: +{n_new} new hand-labeled rows saved -> {ground_truth_path}")

    for line in lines:
        print(line)
    return lines
