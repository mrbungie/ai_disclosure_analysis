"""Shared GPU/batching runtime for the embedding stages.

Split out of ai_prefilter.py when embedding and scoring became two separate
passes: embedding is the expensive, GPU-bound half whose output must be kept
(re-embedding 3.28M paragraphs costs ~40 min of RTX 5090 time), while scoring
against anchors is a 3.28M x 1024 by 1024 x 20 matmul that runs in seconds.
Both stages need the same device probing, token-budget batching, and live
throughput reporting, so it lives here rather than being imported sideways
from one script into the other.
"""

from __future__ import annotations

import os

# Read once at CUDA init: long-lived runs fragment the caching allocator badly
# when batch shapes vary, which is exactly what token-budget batching does.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PARAGRAPH_KEY = ("country_code", "form", "accession_number", "item_key", "paragraph_index")

TORCH_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def _load_model(model_name: str, device: str | None, dtype: str = "fp32") -> SentenceTransformer:
    """Load the encoder at the requested precision.

    Measured on this corpus (6k real paragraphs, RTX 5090) against fp32:
        fp32+TF32  1.28x, min cosine 0.999872, 0.10% of rows change anchor
        fp16       2.90x, min cosine 0.999775, 0.28% change anchor
        bf16       2.92x, min cosine 0.999229, 2.62% change anchor
    fp16 is the sweet spot — same speed as bf16 with far better fidelity,
    because bf16 trades mantissa bits for range this model does not need.
    The rows that change anchor are ones where two categories were already
    tied to within 5e-4, not rows whose meaning moved. fp32 stays the default
    so precision is never downgraded without someone asking for it.
    """
    torch_dtype = TORCH_DTYPES[dtype]
    if torch_dtype is not None:      # reduced precision already ignores TF32 for matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = SentenceTransformer(model_name, device=device)
    if torch_dtype is torch.float16:
        model = model.half()
    elif torch_dtype is torch.bfloat16:
        model = model.bfloat16()
    return model


def _encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Hardware probing and token-budget batching
#
# Nothing here hardcodes a VRAM size, a GPU generation or a batch constant.
# At startup we (1) ask whatever backend we landed on how much memory is free
# RIGHT NOW, (2) grow a real probe batch geometrically, measuring peak bytes
# AND throughput on this very model, and (3) stop at the SMALLEST batch that
# already saturates throughput. Filling the device is not the goal: measured
# on an RTX 5090, bge-m3 plateaus around 85k tok/s at ~10k tokens/batch and
# gains nothing from 145k, so a bigger batch would only buy OOM risk and
# coarser progress reporting.
#
# Batches are then packed by TOKENS, not row count, so a batch of 20-token
# boilerplate lines is wide and a batch of 8k-token tables is narrow — same
# memory ceiling either way. A row count can only ever be right for one text
# length, and this corpus runs from 20-char headings to 210k-char tables.
# ---------------------------------------------------------------------------

PROBE_START_TOKENS = 4096
PROBE_MAX_STEPS = 8
SATURATION_GAIN = 0.05          # <5% tok/s improvement => stop growing the batch
MAX_ROWS_PER_BATCH = 4096       # kernel-launch/overhead ceiling, not a memory one
MIN_TOKEN_BUDGET = 2048
DEFAULT_MEMORY_FRACTION = 0.6
CPU_ROW_CAP = 64                # CPU throughput plateaus early; big batches only cost RAM
TOKEN_SAMPLE = 512


def resolve_device(requested: str | None) -> str:
    """Pick the best backend present when the caller didn't name one."""
    if requested:
        return requested
    for kind in ("cuda", "xpu", "mps"):
        backend = getattr(torch, kind, None)
        if backend is not None and callable(getattr(backend, "is_available", None)) and backend.is_available():
            return kind
    return "cpu"


def device_kind(device: str) -> str:
    return str(device).split(":", 1)[0]


def _backend(device: str):
    return getattr(torch, device_kind(device), None)


def _cpu_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def device_memory(device: str) -> tuple[int | None, int | None]:
    """(free, total) bytes for `device`; (None, None) when the backend hides it."""
    kind = device_kind(device)
    backend = _backend(device)
    if backend is not None and callable(getattr(backend, "mem_get_info", None)):
        try:
            free, total = backend.mem_get_info()
            return int(free), int(total)
        except (RuntimeError, TypeError, ValueError):
            pass
    if kind == "mps" and backend is not None:
        try:
            total = int(backend.recommended_max_memory())
            used = int(backend.current_allocated_memory())
            return max(total - used, 0), total
        except (RuntimeError, AttributeError):
            pass
    if kind == "cpu":
        available = _cpu_available_bytes()
        return available, available
    return None, None


def _release(device: str) -> None:
    backend = _backend(device)
    for name in ("synchronize", "empty_cache"):
        call = getattr(backend, name, None)
        if callable(call):
            try:
                call()
            except (RuntimeError, TypeError):
                pass


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024   # Linux reports KiB


def _measure(model: SentenceTransformer, texts: list[str], device: str) -> tuple[int, float]:
    """(peak bytes, seconds) for encoding `texts` as ONE batch on this backend."""
    backend = _backend(device)
    trackable = backend is not None and callable(getattr(backend, "max_memory_allocated", None))
    _release(device)
    if trackable:
        reset = getattr(backend, "reset_peak_memory_stats", None)
        if callable(reset):
            try:
                reset()
            except (RuntimeError, TypeError):
                pass
        base = int(backend.memory_allocated()) if callable(getattr(backend, "memory_allocated", None)) else 0
    else:
        base = _rss_bytes()

    start = time.perf_counter()
    _encode(model, texts, len(texts))
    if trackable:
        sync = getattr(backend, "synchronize", None)
        if callable(sync):
            try:
                sync()
            except (RuntimeError, TypeError):
                pass
    seconds = time.perf_counter() - start
    peak = (int(backend.max_memory_allocated()) - base) if trackable else (_rss_bytes() - base)
    return max(peak, 0), seconds


def count_tokens(model: SentenceTransformer, texts, max_seq_length: int) -> np.ndarray:
    """Exact token count per row, clipped to the model's ceiling.

    Estimating tokens from character length was tried and abandoned: on this
    corpus chars-per-token ranges from 0.50 (dense numeric tables) to 5.07
    (English prose), so any single ratio is wrong by up to 10x — and because
    batch cost is rows x widest-row, a 3x token underestimate becomes a 9x
    memory overshoot. Real tokenization of 20k rows costs 0.51s against ~60s
    of encoding for the same rows: ~1% for exactness. Rows are clipped to
    max_seq_length * 8 chars first, which cannot change min(tokens, max_seq)
    but keeps 200k-char tables from dominating the tokenizer.
    """
    clipped = [text[: max_seq_length * 8] for text in texts]
    if not clipped:
        return np.zeros(0, dtype=np.int64)
    try:
        encoded = model.tokenizer(clipped, add_special_tokens=True, truncation=False)["input_ids"]
    except (AttributeError, TypeError, ValueError):     # fallback: median-ratio estimate
        lengths = np.fromiter((len(text) for text in clipped), dtype=np.float64, count=len(clipped))
        return np.clip(np.ceil(lengths / 4.0) + 2, 1, max_seq_length).astype(np.int64)
    return np.clip(np.fromiter((len(ids) for ids in encoded), dtype=np.int64, count=len(encoded)),
                   1, max_seq_length)


def token_stats(model: SentenceTransformer, texts: list[str], sample: int = TOKEN_SAMPLE) -> list[tuple[str, int]]:
    """(text, exact token count) pairs for a corpus sample, for probe batches."""
    probe = [text for text in texts if text][:sample]
    if not probe:
        return [("paragraph text", 4)]
    max_seq = int(getattr(model, "max_seq_length", 512) or 512)
    return list(zip(probe, count_tokens(model, probe, max_seq).tolist()))


def _build_probe_batch(pairs: list[tuple[str, int]], target_tokens: int, max_rows: int) -> tuple[list[str], int]:
    """A batch of near-uniform rows totalling ~`target_tokens` PADDED tokens.

    Uniform rows because that is what the real batches look like — plan_batches
    sorts by length before packing — and because a mixed batch's cost is set by
    its widest row, which would make the measurement meaningless.
    """
    ordered = sorted((tokens, text) for text, tokens in pairs if tokens > 0)
    if not ordered:
        return ["paragraph text"], 4
    width, text = ordered[len(ordered) // 2]        # median row length
    rows = max(1, min(target_tokens // width, max_rows))
    return [text] * rows, rows * width


def autotune_batching(
    model: SentenceTransformer,
    probe_texts: list[str],
    device: str,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> dict:
    """Find the smallest per-batch token budget that already saturates throughput.

    Grows the batch geometrically and keeps measuring. Growth stops on whichever
    comes first: throughput stops improving, the projected peak would exceed the
    memory share we are allowed, or the probe OOMs (in which case the last good
    step stands). The linear fit over the measured steps is kept only to project
    the next step and to report bytes/token.
    """
    kind = device_kind(device)
    free, total = device_memory(device)
    max_seq_length = int(getattr(model, "max_seq_length", 512) or 512)
    pairs = token_stats(model, probe_texts)
    max_rows = CPU_ROW_CAP if kind == "cpu" else MAX_ROWS_PER_BATCH
    budget_bytes = None if free is None else max(free * memory_fraction, 0.0)

    steps: list[dict] = []
    best_rate = 0.0
    chosen = MIN_TOKEN_BUDGET
    target = PROBE_START_TOKENS
    stop_reason = "max_steps"
    for _ in range(PROBE_MAX_STEPS):
        texts, tokens = _build_probe_batch(pairs, target, max_rows)
        if tokens <= 0:
            break
        try:
            peak, seconds = _measure(model, texts, device)
        except torch.OutOfMemoryError:
            _release(device)
            stop_reason = "oom"
            break
        rate = tokens / seconds if seconds > 0 else 0.0
        steps.append({
            "target_tokens": target, "tokens": tokens, "rows": len(texts),
            "peak_bytes": peak, "seconds": seconds, "tokens_per_second": rate,
        })
        gain = (rate - best_rate) / best_rate if best_rate > 0 else float("inf")
        if rate > best_rate:
            best_rate = rate
        chosen = tokens
        if gain < SATURATION_GAIN:
            stop_reason = "throughput_saturated"
            break
        if len(texts) >= max_rows:
            stop_reason = "row_cap"
            break
        if budget_bytes is not None and peak * 2 > budget_bytes:
            stop_reason = "memory_ceiling"
            break
        target *= 2

    if len(steps) >= 2:
        first, last = steps[0], steps[-1]
        span = last["tokens"] - first["tokens"]
        per_token = (last["peak_bytes"] - first["peak_bytes"]) / span if span > 0 else last["peak_bytes"] / last["tokens"]
        fixed = max(last["peak_bytes"] - per_token * last["tokens"], 0.0)
    elif steps:
        per_token = steps[0]["peak_bytes"] / max(steps[0]["tokens"], 1)
        fixed = 0.0
    else:
        per_token, fixed = 45_000.0, 0.0
    per_token = max(per_token, 1.0)

    # Hard ceiling regardless of what the throughput search picked.
    if budget_bytes is not None:
        chosen = min(chosen, int(max(budget_bytes - fixed, 0.0) / per_token))
    token_budget = max(int(chosen), MIN_TOKEN_BUDGET, max_seq_length)

    return {
        "device": device,
        "device_kind": kind,
        "free_bytes": free,
        "total_bytes": total,
        "memory_fraction": memory_fraction,
        "budget_bytes": None if budget_bytes is None else int(budget_bytes),
        "bytes_per_token": float(per_token),
        "fixed_overhead_bytes": int(fixed),
        "token_budget": token_budget,
        "projected_peak_bytes": int(fixed + per_token * token_budget),
        "max_rows_per_batch": int(max_rows),
        "max_seq_length": max_seq_length,
        "saturated_tokens_per_second": best_rate,
        "stop_reason": stop_reason,
        "probe_steps": steps,
    }


def plan_batches(token_counts: np.ndarray, token_budget: int, max_rows: int) -> list[np.ndarray]:
    """Group row indices into padded batches that each fit `token_budget`.

    Rows are ordered by length first so a batch's padding waste stays small;
    cost is `rows * longest_row_in_batch` because padding to the widest row is
    what actually gets materialized on the device.
    """
    order = np.argsort(token_counts, kind="stable")
    batches: list[np.ndarray] = []
    start = 0
    while start < len(order):
        widest = 0
        end = start
        while end < len(order):
            candidate = max(widest, int(token_counts[order[end]]))
            rows = end - start + 1
            if rows > max_rows or (rows > 1 and candidate * rows > token_budget):
                break
            widest = candidate
            end += 1
        if end == start:        # a single row wider than the whole budget: take it alone
            end = start + 1
        batches.append(order[start:end])
        start = end
    return batches


def _gib(value) -> str:
    return "n/a" if value is None else f"{value / 1024 ** 3:.2f} GiB"


class ThroughputReporter:
    """Speed WHILE the run is in flight, not only in the JSON written at the end."""

    def __init__(self, total: int, label: str, interval: float = 2.0, device: str = "cpu", stream=None):
        self.total = max(int(total), 0)
        self.label = label
        self.interval = max(interval, 0.0)
        self.device = device
        self.stream = stream or sys.stderr
        self.rows = 0
        self.tokens = 0
        self.encode_seconds = 0.0
        self.started = time.perf_counter()
        self.last_emit = self.started
        self.last_rows = 0
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def _mem(self) -> str:
        free, total = device_memory(self.device)
        if free is None or total is None:
            return ""
        return f" | mem {_gib(total - free)}/{_gib(total)}"

    def update(self, rows: int, encode_seconds: float, tokens: int = 0) -> None:
        self.rows += rows
        self.tokens += tokens
        self.encode_seconds += encode_seconds
        now = time.perf_counter()
        if self.interval and now - self.last_emit < self.interval:
            return
        self.emit(now)

    def emit(self, now: float | None = None) -> None:
        now = now or time.perf_counter()
        wall = max(now - self.started, 1e-9)
        window = max(now - self.last_emit, 1e-9)
        inst = (self.rows - self.last_rows) / window
        avg = self.rows / wall
        encode_rate = self.rows / self.encode_seconds if self.encode_seconds > 0 else 0.0
        token_rate = self.tokens / self.encode_seconds if self.encode_seconds > 0 else 0.0
        parts = [f"[{self.label}] {self.rows:,}"]
        if self.total:
            parts.append(f"/{self.total:,} ({100 * self.rows / self.total:5.1f}%)")
        parts.append(f" | {inst:,.0f} par/s now | {avg:,.0f} par/s avg | {encode_rate:,.0f} par/s encode")
        if token_rate:
            parts.append(f" | {token_rate / 1000:,.1f}k tok/s")
        if self.total and avg > 0:
            eta = (self.total - self.rows) / avg
            parts.append(f" | ETA {int(eta // 3600):02d}:{int(eta % 3600 // 60):02d}:{int(eta % 60):02d}")
        parts.append(self._mem())
        line = "".join(parts)
        self.stream.write(("\r" + line + "    ") if self.tty else (line + "\n"))
        self.stream.flush()
        self.last_emit = now
        self.last_rows = self.rows

    def close(self) -> None:
        self.emit()
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()


def _encode_batch(model: SentenceTransformer, texts: list[str], plan: dict) -> np.ndarray:
    """Encode one planned batch, halving on OOM so a bad estimate degrades
    instead of killing a multi-hour run. A survived OOM permanently lowers the
    budget: whatever the probe concluded, the device just said otherwise."""
    try:
        return _encode(model, texts, len(texts))
    except torch.OutOfMemoryError:
        _release(plan["device"])
        if len(texts) <= 1:
            raise
        plan["token_budget"] = max(plan["token_budget"] // 2, plan["max_seq_length"])
        plan["oom_backoffs"] = plan.get("oom_backoffs", 0) + 1
        middle = len(texts) // 2
        first = _encode_batch(model, texts[:middle], plan)
        second = _encode_batch(model, texts[middle:], plan)
        return np.concatenate([first, second], axis=0)


def encode_planned(
    model: SentenceTransformer,
    texts: list[str],
    plan: dict,
    reporter: ThroughputReporter | None = None,
) -> tuple[np.ndarray, float]:
    """Encode `texts` under the token budget, returning rows in their ORIGINAL order."""
    token_counts = count_tokens(model, texts, plan["max_seq_length"])
    batches = plan_batches(token_counts, plan["token_budget"], plan["max_rows_per_batch"])
    output: np.ndarray | None = None
    elapsed = 0.0
    for indices in batches:
        batch_texts = [texts[i] for i in indices]
        start = time.perf_counter()
        vectors = _encode_batch(model, batch_texts, plan)
        batch_seconds = time.perf_counter() - start
        elapsed += batch_seconds
        if output is None:
            output = np.empty((len(texts), vectors.shape[1]), dtype=np.float32)
        output[indices] = vectors
        if reporter is not None:
            padded = len(indices) * int(token_counts[indices].max())
            reporter.update(len(indices), batch_seconds, padded)
    if output is None:
        output = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return output, elapsed

