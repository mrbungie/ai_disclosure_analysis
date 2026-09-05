"""
scripts/common/pdf/backends/vlm_mineru.py — MinerU2.5 backend (default
VLM). Reads page PIXELS, so unlike the PyMuPDF backend it recovers pages
with no text layer at all, and it returns TYPED blocks — which is what
replaces the hand-written furniture heuristics rather than adding to them.

Chosen over dots.mocr and PaddleOCR-VL on the POC in
docs/analytics/pdf-backend-poc.md: same perfect table-cell recall as
dots.mocr, but it is the only backend that classified the CMF/GRI margin
markers as their own type instead of leaking them into the prose stream,
and it recovers infographic KPI text that dots.mocr discards as an
undifferentiated Picture.

TWO EXECUTION PATHS, same output. `transformers` needs nothing beyond the
model and runs anywhere; `vllm` needs a server but is the only path with a
tolerable wall clock at corpus scale (~300k pages — see the POC doc's
throughput section). The backend is written so the pipeline can't tell
them apart.
"""

import os

from scripts.common.pdf import blocks as B
from scripts.common.pdf.watchdog import page_deadline

DEFAULT_MODEL = os.environ.get("MINERU_MODEL", "opendatalab/MinerU2.5-Pro-2605-1.2B")

#: MinerU's own block labels -> this project's taxonomy. Anything not
#: listed falls through to BODY, which is the safe default: an unknown
#: label that is really narrative gets kept (recoverable downstream),
#: whereas defaulting to "furniture" would silently delete it.
_TYPE_MAP = {
    "text": B.BODY, "title": B.HEADING, "list": B.LIST_ITEM,
    "table": B.TABLE, "table_caption": B.CAPTION, "table_footnote": B.FOOTNOTE,
    "image": B.FIGURE, "image_caption": B.CAPTION, "image_footnote": B.CAPTION,
    "equation": B.FORMULA, "code": B.BODY,
    "header": B.PAGE_HEADER, "footer": B.PAGE_FOOTER,
    "page_number": B.PAGE_NUMBER, "page_footnote": B.FOOTNOTE,
    "aside_text": B.MARGIN_NOTE,
}


class MinerUBackend:
    name = "mineru"
    reads_pixels = True
    #: One GPU, one process. See PdfBackend.max_workers — forking a loaded
    #: VLM across os.cpu_count() workers is an OOM, not a speedup.
    max_workers = 1

    def __init__(self, model: str = DEFAULT_MODEL, dpi: int | None = None,
                 runtime: str = "transformers", server_url: str = "",
                 page_timeout: int = 240, **_):
        self.model_id = model
        #: See PaddleOcrVlBackend.page_timeout. Higher here because this
        #: backend's measured steady state is ~2x slower.
        self.page_timeout = page_timeout
        self.runtime = runtime
        self.server_url = server_url
        self.dpi = dpi
        self._client = None
        self._model = None

    def start(self) -> None:
        if self._client is not None:
            return
        from mineru_vl_utils import MinerUClient

        if self.runtime == "vllm":
            # vllm-async-engine / http client: the model is served out of
            # process, so nothing is loaded here.
            self._client = MinerUClient(backend="http-client", server_url=self.server_url,
                                        model_name=self.model_id)
            return

        import torch
        from transformers import AutoConfig, AutoProcessor
        import transformers

        config = AutoConfig.from_pretrained(self.model_id)
        # The checkpoint names its own architecture (a Qwen2-VL variant);
        # resolving it from the config rather than hardcoding the class
        # keeps this working across MinerU point releases, which have
        # changed the backing architecture before.
        model_cls = getattr(transformers, config.architectures[0])
        self._model = model_cls.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="cuda")
        processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        self._client = MinerUClient(backend="transformers", model=self._model, processor=processor)

    def close(self) -> None:
        self._client = self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def parse_page(self, page, page_no: int) -> list[B.Block]:
        from scripts.common.pdf.render import DEFAULT_DPI, render_page

        self.start()
        image = render_page(page, dpi=self.dpi or DEFAULT_DPI)
        out = []
        # two_step_extract = layout pass, then per-region recognition. The
        # regions come back already sorted into reading order, which is the
        # single biggest thing the PyMuPDF backend gets wrong on multi-column
        # Memoria pages.
        with page_deadline(self.page_timeout):
            items = self._client.two_step_extract(image)
        for item in items:
            raw_type = item.get("type", "")
            out.append(B.Block(
                type=_TYPE_MAP.get(raw_type, B.BODY),
                text=item.get("content") or "",
                page=page_no,
                bbox=tuple(item["bbox"]) if item.get("bbox") else None,
                raw_type=raw_type,
            ))
        return out
