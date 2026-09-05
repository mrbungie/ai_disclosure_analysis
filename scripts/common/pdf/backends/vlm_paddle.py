"""
scripts/common/pdf/backends/vlm_paddle.py — PaddleOCR-VL 1.6 backend.

The default VLM for this pipeline, on the measured POC in
docs/analytics/pdf-backend-poc.md: it tied MinerU2.5-Pro and dots.mocr on
table-cell recall (52/52 on a four-table liquidity page the CPU backend
scored 42/52 on), matched MinerU on keeping page furniture out of the
prose stream — including the CMF/GRI gutter markers that dots.mocr leaks —
recovered more of an infographic page's KPI text than either, and did it
in roughly half the wall clock.

It also ships its own furniture policy rather than needing one written for
it: `markdown_ignore_labels` in the pipeline's returned settings is the
model's own list of labels it drops when rendering a page, and
`_TYPE_MAP` below routes exactly those to this project's FURNITURE types.

INSTALLATION is the one place this backend is fussier than the others: it
needs `paddlepaddle-gpu` built for your CUDA arch, which for a Blackwell
card (RTX 5090, sm_120) means the cu129 wheel, not the cu126 one the model
card suggests. `opencv-python-headless` is required in place of
`opencv-python` on a machine with no libGL.
"""

import os

from scripts.common.pdf import blocks as B

DEFAULT_PIPELINE_VERSION = os.environ.get("PADDLEOCR_VL_VERSION", "v1.6")

_TYPE_MAP = {
    "text": B.BODY, "paragraph_title": B.HEADING, "doc_title": B.HEADING,
    "abstract": B.BODY, "content": B.BODY, "reference": B.BODY,
    "table": B.TABLE, "table_title": B.CAPTION, "figure_title": B.CAPTION,
    "chart_title": B.CAPTION, "vision_footnote": B.FOOTNOTE, "footnote": B.FOOTNOTE,
    "formula": B.FORMULA, "algorithm": B.BODY,
    "image": B.FIGURE, "figure": B.FIGURE, "chart": B.FIGURE, "seal": B.FIGURE,
    "header": B.PAGE_HEADER, "header_image": B.PAGE_HEADER,
    "footer": B.PAGE_FOOTER, "footer_image": B.PAGE_FOOTER,
    "number": B.PAGE_NUMBER, "aside_text": B.MARGIN_NOTE,
}


class PaddleOcrVlBackend:
    name = "paddleocr_vl"
    reads_pixels = True
    max_workers = 1

    def __init__(self, pipeline_version: str = DEFAULT_PIPELINE_VERSION, dpi: int | None = None,
                 runtime: str = "transformers", server_url: str = "", **_):
        self.pipeline_version = pipeline_version
        self.dpi = dpi
        self.runtime = runtime
        self.server_url = server_url
        self._pipeline = None

    def start(self) -> None:
        if self._pipeline is not None:
            return
        from paddleocr import PaddleOCRVL

        kwargs = {"pipeline_version": self.pipeline_version}
        if self.runtime == "vllm":
            # The 0.9B recognizer served out of process. The layout detector
            # still runs locally on Paddle; only the VLM call goes over HTTP.
            kwargs.update(vl_rec_backend="vllm-server", vl_rec_server_url=self.server_url)
        self._pipeline = PaddleOCRVL(**kwargs)

    def close(self) -> None:
        self._pipeline = None

    def parse_page(self, page, page_no: int) -> list[B.Block]:
        import numpy as np

        from scripts.common.pdf.render import DEFAULT_DPI, render_page

        self.start()
        image = render_page(page, dpi=self.dpi or DEFAULT_DPI)
        # BGR: paddle's pipeline takes OpenCV-convention arrays, and handing
        # it RGB silently degrades recognition on colour-on-colour text
        # rather than raising.
        array = np.asarray(image)[:, :, ::-1]

        out = []
        for result in self._pipeline.predict(array):
            payload = result.json
            if not isinstance(payload, dict):
                import json

                payload = json.loads(payload)
            res = payload.get("res", payload)
            for block in res.get("parsing_res_list", []):
                raw_type = block.get("block_label") or ""
                out.append(B.Block(
                    type=_TYPE_MAP.get(raw_type, B.BODY),
                    text=block.get("block_content") or "",
                    page=page_no,
                    bbox=tuple(block["block_bbox"]) if block.get("block_bbox") else None,
                    raw_type=raw_type,
                ))
        return out
