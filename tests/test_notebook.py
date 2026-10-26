"""The demo notebook keeps its sections in deck order, is committed without outputs, and runs end to end."""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"
WITH_OUTPUTS = NOTEBOOK.with_name("demo_with_outputs.ipynb")

SECTIONS = [
    "## Setup",
    "## 1. The page",
    "## 2. Parse it with Docling",
    "## 3. The same page, two other parsers",
    "## 4. Fine-tuning Qwen VL",
    "## 5. Three kinds of error",
    "## 6. Checks after extraction",
    "## 7. A table across pages",
    "## 8. Try your own document",
]


def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def test_sections_in_order():
    headings = [
        line.strip()
        for cell in cells()
        if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines()
        if line.startswith("## ")
    ]
    assert headings == SECTIONS


def test_committed_without_outputs():
    for cell in cells():
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_copy_with_outputs_has_the_same_cells():
    """demo_with_outputs.ipynb is demo.ipynb run once; regenerate it when demo.ipynb changes:
    jupyter nbconvert --to notebook --execute notebooks/demo.ipynb --output demo_with_outputs.ipynb"""
    saved = json.loads(WITH_OUTPUTS.read_text(encoding="utf-8"))["cells"]
    assert [c["source"] for c in saved] == [c["source"] for c in cells()]
    assert not [o for c in saved if c["cell_type"] == "code" for o in c["outputs"] if o["output_type"] == "error"]


def test_runs_end_to_end():
    """Runs every cell on the CPU, Docling included; about a minute once Docling's models are downloaded."""
    pytest.importorskip("docling")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=900, resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors
