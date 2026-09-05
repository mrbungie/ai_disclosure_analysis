"""Backend registry. Imports are lazy — a machine with no GPU (or no
torch) must still be able to run the PyMuPDF backend, and importing
`scripts.common.pdf` must not drag in transformers."""

_BACKENDS = {
    "pymupdf": ("scripts.common.pdf.backends.pymupdf_layout", "PyMuPdfBackend"),
    "paddleocr_vl": ("scripts.common.pdf.backends.vlm_paddle", "PaddleOcrVlBackend"),
    "mineru": ("scripts.common.pdf.backends.vlm_mineru", "MinerUBackend"),
    "dots": ("scripts.common.pdf.backends.vlm_dots", "DotsMocrBackend"),
}


def available() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str, **kwargs):
    import importlib

    if name not in _BACKENDS:
        raise ValueError(f"unknown pdf backend {name!r}; available: {available()}")
    module_path, class_name = _BACKENDS[name]
    return getattr(importlib.import_module(module_path), class_name)(**kwargs)
