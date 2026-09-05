# Project-wide directives

COMMIT TO B2 AND GIT OFTEN.

## Never delete LLM-labeled or pre-classified data

Any file under `data/` that required an LLM call to produce — judge labels
(`eval_set_detection.parquet`, `eval_set_classification.parquet`),
embeddings, ConceptSeed suggestions, or any other derived artifact whose
regeneration costs money or LLM calls — must NEVER be deleted outright,
including as a side effect of a "reset," "decontaminate," "regenerate
from seed," or similar cleanup step.

**Why:** a prior session's `meta-opt/run-2` branch (containing frozen
detection candidates and a spent test-holdout look) was discarded when
the branch was deleted unmerged, and the underlying labeled data in
`data/interim/eval/` — which only ever lived in the gitignored `data/`
tree, never in git — was lost with no way to recover it. Git history
saved the *code*; nothing saved the *data*.

**Exception — an explicit instruction wins.** If I tell you to delete a
specific file, delete it. Don't archive it "just in case," don't cite this
rule back at me, don't ask again. This rule exists to stop *incidental*
deletion — a cleanup step, a branch reset, a regeneration that silently
takes data with it — not to override me when I've named the file and said
to remove it. If the deletion looks costly (an LLM-labeled artifact that
would cost money to rebuild), say so in one line, then do it. Delete it
everywhere it lives, including the B2 bucket — a stale copy in the remote
comes back on the next `pull` and is worse than not deleting at all.

**How to apply** (for deletions I did NOT explicitly ask for): before
removing or overwriting any such file:
1. Move it (don't delete) into `data/archive/<original-relative-path>/<timestamp>/`.
2. Next to it, write a `POINTER.json` with:
   - `original_path`: where it lived in `data/`
   - `deleted_at`: ISO timestamp
   - `deleted_by`: what triggered the removal (script name, git commit/branch,
     manual cleanup, agent session)
   - `reason`: why it was removed
   - `produced_by`: how to regenerate it (script + args), and whether that
     regeneration calls an LLM / costs money
3. Only then proceed with the reset/overwrite.

`data/archive/` lives under the already-gitignored `data/` tree, so this
costs disk, not git history — same tradeoff as the phase0 embedding cache
that's deliberately never deleted (see `.claude/skills/phase0-opt/SKILL.md`).
