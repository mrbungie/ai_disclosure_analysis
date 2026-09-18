# Atomic AI Disclosure Validator (`apps/validator`)

Atomic human validation tool for auditing AI model extractions across the project (prefilter, semantic frames, and operational activities).

Each verification check is **flattened into 1 fact at a time**: a single direct, objective question evaluated against the paragraph text (with evidence sentences highlighted), answered using 3 simple choices:
- **[Y] Yes**
- **[N] No**
- **[U] Uncertain**

---

## 1. Getting Started

```bash
# Serve the web interface:
make validator
# or alternatively:
cd apps/validator && python -m http.server 8765

# Open in browser:
http://localhost:8765
```

To regenerate or flatten the verification sample:
```bash
# Flatten from an existing sample (fast):
uv run --frozen --no-sync python apps/validator/build_sample.py --from-existing apps/validator/data.json

# Or sample from scratch traversing bronze and silver layers:
uv run --frozen --no-sync python apps/validator/build_sample.py --frames 300 --prefilter 300 --activities 120
```

---

## 2. Validation Scopes

You can select the **scope** in the top navigation bar:

1. **Prefilter** (AI Mention & Corporate Relevance):
   - Explicit mention of AI / Machine Learning / GenAI.
   - Substantive corporate disclosure regarding the firm itself vs. incidental / boilerplate.
2. **Frames** (LLM Judge Semantic Frames):
   - Genuine frame existence (verifying absence of hallucination).
   - Temporal dimension (`realized`, `planned`, `expected`, `hypothetical`).
   - Promotional rhetoric (corporate hype without substance vs. factual disclosure).
   - Identified specificities (product/system, business process, vendor/partner, quantified metric, date/timeline).
   - Validity of attributed textual evidence.
   - Negative paragraphs: checking whether a relevant frame was missed.
3. **Activities** (Operational AI Activities):
   - Core asserted fact (firm develops/deploys tool X).
   - Technology origin (**proprietary** only if the firm builds/sells it as its own product; **third-party** if an external vendor/tool is named; **unspecified** if not stated).
   - Action and technological object.
   - Target beneficiary and adoption/maturity stage.
   - Named entities and roles (own brand, external vendor, partner, etc.).
   - Direct textual evidence.
   - Paragraph completeness (checking whether an activity described in the text was omitted).

---

## 3. Keyboard Shortcuts (Ultra-fast)

- `Y` or `1`: Mark **Yes** (automatically advances to the next item).
- `N` or `2`: Mark **No** (automatically advances to the next item).
- `U` or `3`: Mark **Uncertain** (automatically advances).
- `←` or `A`: Previous item.
- `→` or `D`: Next item.
- `P`: Jump to the first unanswered item.

---

## 4. Persistence and Export

- Responses are continuously persisted in browser `localStorage`.
- **Export**: Generates a JSON file with all recorded validations. Save to `apps/validator/annotations/<name>.json`.
- **Import**: Load or merge annotations from another machine or session.
- **Summary**:
  ```bash
  uv run --frozen --no-sync python apps/validator/summarize.py
  ```
  Computes aggregated metrics: confusion matrices, precision/recall (raw and stratum-reweighted for prefilter), agreement rates, and category-level breakdowns.
