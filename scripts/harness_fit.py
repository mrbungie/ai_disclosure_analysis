"""
harness_fit.py — Shared machinery for the harness-optimization cycle.

The pipeline contains two rule-based harnesses — keyword machinery standing in
for an LLM at corpus scale — and both are fit by the SAME cycle (the
meta-harness framing, docs/meta_harness_plan.md):

    stratified sample -> LLM judge labels -> dev-only search (fold-stability
    penalized) -> freeze artifact -> single holdout look -> write to config

Cycle 1 (scripts 07-08) fits the detection keywords (`ai_keywords`: "is this
text AI-related?"). Cycle 2 (scripts 09-12) fits the classification formulas
("what does the AI text say?", six dimensions). This module holds everything
the two cycles share, so the cycle itself is one implementation with two
instantiation sites, not two ad-hoc copies.

Discipline encoded here:
  - Metrics come in two flavors: RAW (on the sample as drawn — internally
    consistent, but shaped by the sampling design) and WEIGHTED
    (inverse-probability weighted by each row's `sampling_weight` — an
    unbiased estimate of the population value). Sampling designs that
    oversample one stratum (e.g. 07's 50/50 candidate/excluded draw) inflate
    raw recall; the weighted numbers undo that.
  - fold_stability_score() is deliberately NOT called cross-validation:
    nothing is trained per fold (a keyword list / boolean formula has no
    trainable parameters), and search candidates are mined from ALL of dev,
    including every fold's test rows. The folds only measure how stable a
    candidate's F1 is across subsamples, penalizing candidates that win on
    one lucky fold. The single-look holdout is the only honest estimate.
  - The holdout lock: once a cycle's holdout report exists, re-running the
    fit against the same holdout is refused. Improving further requires a
    fresh labeled batch.
"""

import asyncio
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_judge(output_type, system_prompt: str):
    """Pydantic-AI agent for the outer-loop judge. Reads LLM_JUDGE_* env vars;
    the judge should be a different model family from anything the harness
    approximates, so it isn't grading a close relative of itself."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = os.environ.get("LLM_JUDGE_API_KEY")
    if not api_key:
        raise ValueError("LLM_JUDGE_API_KEY is not set. Configure it in .env.")
    base_url = os.environ.get("LLM_JUDGE_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    # Several OpenAI-"compatible" NIM-hosted models 400 on strict tool-definition
    # mode (extra_forbidden on tools.0.function.strict) — disable it so the judge
    # isn't locked to the handful of models that happen to support it.
    profile = OpenAIModelProfile(openai_supports_strict_tool_definition=False)
    model = OpenAIChatModel(model_name, provider=provider, profile=profile)
    return Agent(model, output_type=output_type, retries=3, system_prompt=system_prompt)


async def judge_one(agent, prompt: str, max_retries: int = 5) -> dict | None:
    """One judge call with rate-limit backoff. The FULL text is judged — no
    truncation: a label for a truncated excerpt is a label for a different
    document than the harness will see."""
    retry_delay = 5.0
    for attempt in range(max_retries):
        try:
            result = await agent.run(prompt)
            return result.output.model_dump(mode="json")
        except Exception as e:
            error_str = str(e)
            is_rate_limit = ("429" in error_str or "rate limit" in error_str.lower()
                             or getattr(e, "status_code", None) == 429)
            if is_rate_limit and attempt < max_retries - 1:
                print(f"  Rate limited. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2.0
                continue
            print(f"  Judge error: {e}")
            return None
    return None


async def judge_batch(agent, to_label: pd.DataFrame, make_prompt, label_fields: list[str],
                      existing_records: list[dict], out_path: Path,
                      concurrency: int, delay: float) -> pd.DataFrame:
    """Concurrent judge labeling with periodic checkpointing to `out_path`.
    `make_prompt(row)` builds the judge prompt; each judge output dict's
    `label_fields` are stored as llm_<field> columns."""
    records = existing_records
    write_lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    for _, row in to_label.iterrows():
        queue.put_nowait(row)

    progress = {"done": 0, "total": len(to_label)}

    async def worker():
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            label = await judge_one(agent, make_prompt(row))
            async with write_lock:
                if label is not None:
                    record = row.to_dict()
                    for field in label_fields:
                        record[f"llm_{field}"] = label[field]
                    records.append(record)
                progress["done"] += 1
                print(f"[{progress['done']}/{progress['total']}] labeled")
                if len(records) % 20 == 0:
                    pd.DataFrame(records).to_parquet(out_path, index=False)
            if delay > 0:
                await asyncio.sleep(delay)
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]
    await asyncio.gather(*workers)

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
                              n: int, seed: int) -> None:
    """Excel workbook for hand-labeling a subsample of judge-labeled rows.

    Sheet 'label_me': id, text, and one EMPTY TRUE/FALSE column per label
    field — the LLM's labels are deliberately NOT on this sheet, so the human
    rater isn't anchored. Sheet 'llm_labels' holds the judge's labels for the
    same ids; score_agreement() joins the two after the human sheet is filled.

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
    parts = []
    for _, group in pool.groupby(pool[strat].astype(bool)):
        parts.append(group.sample(min(len(group), n // 2), random_state=rng_seed))
    sub = pd.concat(parts)
    if len(sub) < n:  # one class exhausted — top up from the other
        rest = pool.drop(sub.index)
        sub = pd.concat([sub, rest.sample(min(len(rest), n - len(sub)), random_state=rng_seed)])
    sub = sub.sample(frac=1, random_state=rng_seed)

    human = sub[[id_col, text_col]].copy()
    for f in label_fields:
        human[f"your_{f}"] = ""
    llm = sub[[id_col] + llm_cols].copy()

    instructions = pd.DataFrame({
        "How to fill this in": [
            "1. Work ONLY on the 'label_me' sheet. Do not open 'llm_labels' until you are done",
            "   (it holds the LLM judge's answers; peeking anchors your labels and voids the check).",
            f"2. For each row, read the text and fill every your_* column with TRUE or FALSE.",
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


def score_agreement(workbook_path: Path, id_col: str, label_fields: list[str]) -> list[str]:
    """Score a filled-in agreement workbook: raw agreement + Cohen's kappa per
    label field. Returns report lines (also printed)."""
    human = pd.read_excel(workbook_path, sheet_name="label_me")
    llm = pd.read_excel(workbook_path, sheet_name="llm_labels")
    merged = human.merge(llm, on=id_col, how="inner")

    def to_bool(series: pd.Series) -> pd.Series:
        return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes", "y", "t"})

    lines = [f"Judge agreement check — {workbook_path.name}, n={len(merged)}"]
    for f in label_fields:
        col = merged[f"your_{f}"]
        filled = col.notna() & (col.astype(str).str.strip() != "")
        if filled.sum() == 0:
            lines.append(f"  {f}: no human labels filled in yet")
            continue
        sub = merged[filled]
        h = to_bool(sub[f"your_{f}"]).to_numpy()
        m = sub[f"llm_{f}"].astype(bool).to_numpy()
        kappa = cohens_kappa(h, m)
        agree = float(np.mean(h == m))
        lines.append(f"  {f}: n={len(sub)}  raw agreement={agree:.3f}  Cohen's kappa={kappa:.3f}")
    for line in lines:
        print(line)
    return lines
