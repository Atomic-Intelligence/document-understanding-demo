"""Docling on the CPU, in the configuration of the evaluation: layout model Heron, TableFormer in accurate
mode, RapidOCR for text in images."""

from __future__ import annotations

import io
import logging
import os
import time
import warnings
from pathlib import Path

from PIL import Image

from .data import Block, Parsed

DOCLING_TYPE = {"title": "heading", "section_header": "heading", "table": "table", "document_index": "table",
                "list_item": "list", "key_value_region": "key_value", "form": "key_value",
                "picture": "figure", "chart": "figure"}


def _quiet():
    """Keep the notebook's output to what the audience should read: no model-loading logs or warnings."""
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import transformers

    transformers.utils.logging.disable_progress_bar()
    for name in ("docling", "RapidOCR", "transformers", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)


def make_converter():
    """A Docling converter with its models loaded, so that the first page does not wait for them."""
    _quiet()
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (LayoutObjectDetectionOptions, OcrMode, PdfPipelineOptions,
                                                    RapidOcrOptions, TableFormerMode, TableStructureOptions)
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=RapidOcrOptions(backend="torch", mode=OcrMode("default"),
                                    rapidocr_params={"Global.log_level": "error"}),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(mode=TableFormerMode("accurate")),
        layout_options=LayoutObjectDetectionOptions.from_preset("layout_heron_default"),
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=min(4, os.cpu_count() or 4)),
        generate_page_images=True,
    )
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
    })
    converter.initialize_pipeline(InputFormat.IMAGE)
    return converter


def run(converter, source: str | Path | Image.Image) -> Parsed:
    """Parse a PDF, an image file or a PIL image, and return the first page's regions and the Markdown."""
    from docling.datamodel.base_models import DocumentStream

    if isinstance(source, Image.Image):
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        buffer.seek(0)
        source = DocumentStream(name="page.png", stream=buffer)
    start = time.perf_counter()
    doc = converter.convert(source).document
    seconds = time.perf_counter() - start

    page = doc.pages[min(doc.pages)]
    image = page.image.pil_image.convert("RGB")
    scale = image.width / page.size.width
    out = Parsed("docling", image, markdown=doc.export_to_markdown(), seconds=seconds)
    for item, _level in doc.iterate_items():
        prov = [p for p in getattr(item, "prov", []) if p.page_no == page.page_no]
        if not prov:
            continue
        b = prov[0].bbox.to_top_left_origin(page.size.height)
        out.blocks.append(Block(DOCLING_TYPE.get(str(item.label.value), "text"),
                                (b.l * scale, b.t * scale, b.r * scale, b.b * scale)))
    out.tables = [[[cell.text for cell in row] for row in table.data.grid] for table in doc.tables]
    return out
