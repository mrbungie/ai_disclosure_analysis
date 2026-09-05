"""
scripts/common/pdf/backends/vlm_dots.py — dots.mocr backend.

Kept as an alternative to MinerU rather than the default: on the POC
(docs/analytics/pdf-backend-poc.md) it matched MinerU's table-cell recall
and produced slightly richer table markup (<br> inside cells, <strong> on
a subtotal row, real <thead>), but it has one weakness that matters for
this corpus specifically — it has no block type for gutter/margin notes,
so the CMF and GRI reference markers printed down the side of every
Memoria page come back as ordinary Text blocks and land in the prose
corpus. It's the better choice if what you need is faithful table markup;
MinerU is the better choice for a clean narrative corpus.

Its vision tower imports flash-attn unconditionally, and Dao-AILab ships
no sm_120 (Blackwell / RTX 5090) wheel. `SDPA_FLASH_ATTN_SHIM` below is an
exact stand-in — varlen packs N sequences into one tensor delimited by
cu_seqlens, so attending within each segment separately is the same math
as the block-diagonal mask flash-attn applies internally.
"""

import os

from scripts.common.pdf import blocks as B

DEFAULT_MODEL = os.environ.get("DOTS_MOCR_MODEL", "rednote-hilab/dots.mocr")

_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

_TYPE_MAP = {
    "Text": B.BODY, "Title": B.HEADING, "Section-header": B.HEADING,
    "List-item": B.LIST_ITEM, "Caption": B.CAPTION, "Footnote": B.FOOTNOTE,
    "Table": B.TABLE, "Formula": B.FORMULA, "Picture": B.FIGURE,
    "Page-header": B.PAGE_HEADER, "Page-footer": B.PAGE_FOOTER,
}


def install_flash_attn_shim() -> None:
    """Registers an SDPA-backed `flash_attn` module if the real one is
    absent. Import-time side effect confined to this function so it only
    ever runs for callers that actually chose this backend."""
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("flash_attn") is not None:
        return
    import torch
    import torch.nn.functional as F

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None,
                               max_seqlen_k=None, dropout_p=0.0, softmax_scale=None,
                               causal=False, **_):
        outs = []
        for i in range(1, len(cu_seqlens_q)):
            s, e = int(cu_seqlens_q[i - 1]), int(cu_seqlens_q[i])
            sk, ek = int(cu_seqlens_k[i - 1]), int(cu_seqlens_k[i])
            out = F.scaled_dot_product_attention(
                q[s:e].transpose(0, 1).unsqueeze(0),
                k[sk:ek].transpose(0, 1).unsqueeze(0),
                v[sk:ek].transpose(0, 1).unsqueeze(0),
                dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
            outs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outs, 0)

    shim = types.ModuleType("flash_attn")
    shim.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = shim


class DotsMocrBackend:
    name = "dots"
    reads_pixels = True
    max_workers = 1

    def __init__(self, model: str = DEFAULT_MODEL, dpi: int | None = None,
                 max_new_tokens: int = 16000, **_):
        self.model_id = model
        self.dpi = dpi
        self.max_new_tokens = max_new_tokens
        self._model = self._processor = None

    def start(self) -> None:
        if self._model is not None:
            return
        install_flash_attn_shim()
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, attn_implementation="sdpa", dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def close(self) -> None:
        self._model = self._processor = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def parse_page(self, page, page_no: int) -> list[B.Block]:
        import json

        from scripts.common.pdf.render import DEFAULT_DPI, render_page

        self.start()
        image = render_page(page, dpi=self.dpi or DEFAULT_DPI)
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": _PROMPT}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True,
                                 return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated)]
        raw = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            # A page whose generation hit max_new_tokens comes back as
            # truncated JSON. Losing the page silently is the worse
            # failure, so it's surfaced as one BODY block the caller can
            # see in the corpus rather than dropped.
            return [B.Block(type=B.BODY, text=raw.strip(), page=page_no, raw_type="unparsed")]
        return [B.Block(type=_TYPE_MAP.get(item.get("category"), B.BODY),
                        text=item.get("text") or "", page=page_no,
                        bbox=tuple(item["bbox"]) if item.get("bbox") else None,
                        raw_type=item.get("category", ""))
                for item in items]
