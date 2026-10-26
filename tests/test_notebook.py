"""The demo notebook keeps its sections in deck order, is committed without outputs, and runs end to end."""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"
WITH_OUTPUTS = NOTEBOOK.with_name("demo_with_outputs.ipynb")
LIVE = NOTEBOOK.with_name("demo_live.ipynb")

SECTIONS = [
    "## Setup",
    "## 1. The page",
    "## 2. Parse it with Docling",
    "## 3. The same page, two other parsers",
    "## 4. Fine-tuning Qwen VL",
    "## 5. Three kinds of error",
    "## 6. Checks after extraction",
    "## 7. Two parsers, rules and a person",
    "## 8. Try your own document",
]


def cells(notebook=NOTEBOOK):
    return json.loads(notebook.read_text(encoding="utf-8"))["cells"]


@pytest.mark.parametrize("notebook", [NOTEBOOK, LIVE], ids=["demo", "live"])
def test_sections_in_order(notebook):
    headings = [
        line.strip()
        for cell in cells(notebook)
        if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines()
        if line.startswith("## ")
    ]
    assert headings == SECTIONS


@pytest.mark.parametrize("notebook", [NOTEBOOK, LIVE], ids=["demo", "live"])
def test_committed_without_outputs(notebook):
    for cell in cells(notebook):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_live_copy_differs_only_in_its_setup_and_the_live_switch():
    """demo_live.ipynb is demo.ipynb with every model run live: the same code, except that its setup also
    installs PaddleOCR-VL and one cell more sets data.LIVE."""
    demo = ["".join(c["source"]) for c in cells() if c["cell_type"] == "code"]
    live = ["".join(c["source"]) for c in cells(LIVE) if c["cell_type"] == "code"]
    extra = [c for c in live if c not in demo]
    assert len(live) == len(demo) + 1 and len(extra) == 2
    assert "paddlepaddle-gpu" in extra[0] and "data.LIVE" in extra[1]


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
