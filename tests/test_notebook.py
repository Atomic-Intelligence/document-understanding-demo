"""The demo notebook keeps its sections in deck order, is committed without outputs, and runs end to end."""

import json
import subprocess
import sys
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
    "## 7. Two parsers, rules and a person",
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


def test_every_engine_is_cached_by_default():
    """The first code cell chooses, per engine, live or cached; as committed, all are cached."""
    first = "".join(next(c for c in cells() if c["cell_type"] == "code")["source"])
    namespace = {}
    exec(first, namespace)
    assert set(namespace["LIVE"]) == {"docling", "paddleocr_vl", "qwen_base", "qwen_finetuned"}
    assert not any(namespace["LIVE"].values())


def test_the_choice_says_how_much_gpu_memory_each_server_needs():
    from dudemo import live
    first = "".join(next(c for c in cells() if c["cell_type"] == "code")["source"])
    for engine in ("paddleocr_vl", "qwen_base"):
        line = next(line for line in first.splitlines() if f'"{engine}"' in line)
        assert f"{live.GPU_GB[engine]} GB free" in line


def test_copy_with_outputs_has_the_same_cells():
    """demo_with_outputs.ipynb is demo.ipynb run once; regenerate it when demo.ipynb changes:
    jupyter nbconvert --to notebook --execute notebooks/demo.ipynb --output demo_with_outputs.ipynb"""
    saved = json.loads(WITH_OUTPUTS.read_text(encoding="utf-8"))["cells"]
    assert [c["source"] for c in saved] == [c["source"] for c in cells()]
    assert not [o for c in saved if c["cell_type"] == "code" for o in c["outputs"] if o["output_type"] == "error"]


def test_cached_loads_no_engine():
    """With every engine cached, the notebook loads none of their libraries: it runs every code cell after
    the setup in a fresh Python and lists what got imported. (docling-core imports OpenCV by itself when
    it is installed; it is not an engine.)"""
    script = f"""
import json, sys
sys.path.insert(0, {str(NOTEBOOK.parents[1] / "src")!r})
code = ["".join(c["source"]) for c in json.load(open({str(NOTEBOOK)!r}))["cells"] if c["cell_type"] == "code"]
namespace = {{}}
exec(code[0], namespace)
for cell in code[2:]:
    exec(cell, namespace)
print(json.dumps([m for m in ("torch", "transformers", "docling", "paddle", "vllm") if m in sys.modules]))
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=NOTEBOOK.parent)
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads(run.stdout.strip().splitlines()[-1]) == []


def test_runs_end_to_end():
    """Runs every cell as committed, with every engine cached: a few seconds."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=900, resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors
