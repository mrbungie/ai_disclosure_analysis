"""
proposer_agent.py — Automated proposer for one meta-harness-opt iteration.

The harness (this agent: pydantic-ai, Gemini 3.7 Flash, with tool access to
the filesystem and to running Python) DESIGNS a program: it reads the prior
candidates' code, scores, and per-instance traces under
harnesses/<task>/<name>/, diagnoses what's wrong from the evidence, and
writes ONE new self-contained candidate program (harnesses/<task>/<name>/
harness.py, a plain classify(text) -> ... function, stdlib only). That
program — never the agent itself — is what actually runs at corpus scale
(scripts/apply_harness.py). The agent is the search operator, exactly the
role docs/distillation_map.html's proposer plays, just automated instead of
run by hand through the /meta-harness-opt skill.

Usage:
    uv run python scripts/proposer_agent.py --task detection
    uv run python scripts/proposer_agent.py --task classification
"""

import argparse
import contextlib
import io
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

load_dotenv()

HARNESSES_DIR = Path("harnesses")

CONTRACTS = {
    "detection": (
        "classify(text: str) -> bool — is this 10-K paragraph AI-related at all? "
        "Population: individual paragraphs."
    ),
    "classification": (
        "classify(text: str) -> dict with six bool keys: is_substantive, is_promotional, "
        "is_risk_related, is_governance_related, is_use_case_specific, is_quantified. "
        "Population: chunks (already known to discuss AI)."
    ),
}

SYSTEM_PROMPT = """\
You are the proposer in a meta-harness optimization loop for task="{task}".

Contract every candidate program must satisfy: {contract}
A candidate is ONE self-contained, stdlib-only Python file (harnesses/{task}/<name>/harness.py)
exposing exactly that `classify` function. It must not import third-party packages and must
not call any network service or LLM — it is the CHEAP, deterministic program that runs at
corpus scale (over a million rows). You (the agent) are the expensive search operator that
designs it; you never run at corpus scale yourself.

One iteration, in order:
1. list_candidates() and read_file() on harnesses/{task}/ACTIVE, the ACTIVE candidate's
   harness.py, notes.md, and eval_search.json to see current standings.
2. read_trace_wrong() on the ACTIVE candidate (and any other strong candidate) to see the
   ACTUAL misclassified texts — diagnose WHY, not just how much.
3. Optionally run_python() to test a regex/pattern idea against example strings before
   committing to it.
4. write_candidate(name, harness_code, notes) with a NEW name (e.g. "002_shortname"),
   never overwriting an existing candidate directory. notes should explain what evidence
   motivated the change and what you expect.
5. run_eval(name) to score it on the search split. If it does not clearly beat the
   leaderboard, you may iterate on writing a different NEW candidate (never edit one
   already evaluated) up to a couple of times, then stop.

Never invoke anything with --split test. Never fabricate scores — only report what
run_eval actually returned. End your final message with: the diagnosis, the new
candidate's name, and its search-split result compared to the previous best.
"""


def build_agent(task: str) -> Agent:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Configure it in .env.")
    provider = GoogleProvider(api_key=api_key)
    model = GoogleModel("gemini-3.7-flash", provider=provider)
    agent = Agent(model, system_prompt=SYSTEM_PROMPT.format(task=task, contract=CONTRACTS[task]))

    @agent.tool_plain
    def list_candidates() -> str:
        """List candidate directory names under harnesses/<task>/."""
        d = HARNESSES_DIR / task
        return "\n".join(sorted(p.name for p in d.iterdir() if p.is_dir())) or "(none)"

    @agent.tool_plain
    def read_file(path: str) -> str:
        """Read a text file (harness.py, notes.md, eval_search.json, ACTIVE). Truncated to 20k chars."""
        p = Path(path)
        if not p.exists():
            return f"ERROR: {path} does not exist"
        return p.read_text()[:20000]

    @agent.tool_plain
    def read_trace_wrong(candidate: str, n: int = 30) -> str:
        """Rows the candidate got WRONG on the search trace, with predicted vs true labels."""
        import pandas as pd

        path = HARNESSES_DIR / task / candidate / "trace_search.parquet"
        if not path.exists():
            return f"no trace at {path} — run_eval this candidate first."
        df = pd.read_parquet(path)
        wrong_col = df["wrong"]
        mask = wrong_col if wrong_col.dtype == bool else wrong_col.astype(str) != ""
        wrong = df[mask]
        text_col = "paragraph_text" if "paragraph_text" in df.columns else "chunk_text"
        cols = [c for c in df.columns if c.startswith("pred_") or c.startswith("label_")] + [text_col]
        if wrong.empty:
            return "no misclassified rows on the search trace."
        return wrong[cols].head(n).to_string(max_colwidth=200)

    @agent.tool_plain
    def write_candidate(name: str, harness_code: str, notes: str) -> str:
        """Create harnesses/<task>/<name>/ with harness.py and notes.md. Refuses to overwrite."""
        d = HARNESSES_DIR / task / name
        if d.exists():
            return f"ERROR: {d} already exists — pick a new name, never overwrite an evaluated candidate."
        d.mkdir(parents=True)
        (d / "harness.py").write_text(harness_code)
        (d / "notes.md").write_text(notes)
        return f"wrote {d}/harness.py and notes.md"

    @agent.tool_plain
    def run_eval(candidate: str) -> str:
        """Run scripts/eval_harness.py on the search split for this candidate; returns its output."""
        try:
            result = subprocess.run(
                ["uv", "run", "python", "scripts/eval_harness.py", "--task", task, "--candidate", candidate],
                capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: eval_harness.py timed out after 180s"
        return (result.stdout + result.stderr)[-6000:]

    @agent.tool_plain
    def run_python(code: str) -> str:
        """Execute a short Python snippet (e.g. to test a regex against example strings). stdout only."""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": __builtins__})
        except Exception as e:
            return f"ERROR: {e}\n{buf.getvalue()}"
        return buf.getvalue()[:5000] or "(no output — use print())"

    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(CONTRACTS))
    args = parser.parse_args()

    agent = build_agent(args.task)
    result = agent.run_sync(
        f"Run one meta-harness-opt iteration for task={args.task}. Start by listing candidates "
        f"and reading the ACTIVE one's code, notes, and score, then diagnose from its trace."
    )
    print(result.output)


if __name__ == "__main__":
    main()
