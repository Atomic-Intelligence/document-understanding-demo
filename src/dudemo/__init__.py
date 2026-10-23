"""Helpers for the document-parsing demo notebook: loading pages and saved outputs (`data`), and drawing
regions on a page and rendering Markdown (`show`)."""

import logging
import os
import warnings


def quiet():
    """Keep the notebook's output to what the audience reads: no model-loading logs, progress bars or
    library warnings."""
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        import rapidocr.utils.log as rapidocr_log   # RapidOCR logs through its own handler

        rapidocr_log.logger.handlers = [logging.NullHandler()]
        import transformers

        transformers.utils.logging.disable_progress_bar()
    except ImportError:
        pass
    for name in ("docling", "RapidOCR", "transformers", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)
