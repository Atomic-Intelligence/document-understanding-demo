"""Helpers for the document-parsing demo notebook: loading pages and saved outputs (`data`), and drawing
regions on a page and rendering Markdown (`show`)."""

import importlib.util
import logging
import os
import sys
import warnings


def quiet():
    """Keep the notebook's output to what the audience reads: no model-loading logs, progress bars or
    library warnings. It imports no library that is not already loaded: transformers reads the settings
    from the environment when it loads, and RapidOCR is silenced only if it is there."""
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # also transformers' own progress bars
    if "transformers" in sys.modules:
        sys.modules["transformers"].utils.logging.disable_progress_bar()
    if importlib.util.find_spec("rapidocr"):
        import rapidocr.utils.log as rapidocr_log   # RapidOCR logs through its own handler; a light import

        rapidocr_log.logger.handlers = [logging.NullHandler()]
    for name in ("docling", "RapidOCR", "transformers", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)
