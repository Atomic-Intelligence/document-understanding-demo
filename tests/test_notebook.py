"""The demo notebook keeps its sections in deck order and is committed without outputs."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"

SECTIONS = [
    "## Setup",
    "## 1. The page",
    "## 2. Parse it with Docling",
    "## 3. The same page, three other parsers",
    "## 4. Three kinds of error",
    "## 5. General and fine-tuned Qwen VL",
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
