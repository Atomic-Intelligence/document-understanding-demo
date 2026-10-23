"""Pages, ground truth and saved model outputs, all read into one shape: `Parsed`."""

from __future__ import annotations

import ast
import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

DATA = Path(__file__).resolve().parents[2] / "data"

NAMES = {
    "ground_truth": "Ground truth",
    "docling": "Docling",
    "paddleocr_vl": "PaddleOCR-VL",
    "qwen_base": "Qwen VL, general",
    "qwen_finetuned": "Qwen VL, fine-tuned",
}
CACHED_MODELS = ["paddleocr_vl", "qwen_base", "qwen_finetuned"]


@dataclass
class Block:
    type: str                       # heading, text, table, key_value, list, figure
    box: tuple | None               # (x0, y0, x1, y1) in page pixels, top-left origin


@dataclass
class Parsed:
    """What one parser returned for one page."""
    model: str
    page: Image.Image
    blocks: list[Block] = field(default_factory=list)
    markdown: str = ""
    tables: list[list[list[str]]] = field(default_factory=list)   # table -> row -> cell text
    pairs: list[tuple[str, str]] = field(default_factory=list)    # key-value fields
    seconds: float | None = None

    @property
    def name(self) -> str:
        return NAMES.get(self.model, self.model)


def page_path(page: str) -> Path:
    return DATA / "pages" / f"{page}.png"


def page_image(page: str) -> Image.Image:
    return Image.open(page_path(page)).convert("RGB")


def key(text: str) -> str:
    """A cell or label folded for matching: lower case, letters and digits only."""
    return re.sub(r"[^0-9a-zà-ž]", "", text.casefold())


def examples() -> list[dict]:
    return json.loads((DATA / "examples.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- tables in HTML

class _Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_rows(source: str) -> list[list[str]]:
    parser = _Rows()
    parser.feed(source)
    return [r for r in parser.rows if r]


def pairs_from_tables(tables: list[list[list[str]]]) -> list[tuple[str, str]]:
    """Key-value fields written as a table: a cell ending in ':' followed by its value."""
    pairs = []
    for rows in tables:
        for row in rows:
            for key, value in zip(row, row[1:]):
                if key.endswith(":") and not value.endswith(":"):
                    pairs.append((key, value))
    return pairs


def _pairs_markdown(pairs) -> str:
    return "\n".join(f"**{html.escape(k)}** {html.escape(v)}  " for k, v in pairs)


def _table_html(source: str) -> str:
    source = source.strip()
    return source if source.startswith("<table") else f"<table>{source}</table>"


# --------------------------------------------------------------------------- ground truth

GT_TYPE = {"heading": "heading", "paragraph": "text", "table": "table", "key_value_section": "key_value",
           "logo": "figure"}


def ground_truth(page: str) -> Parsed:
    gt = json.loads((DATA / "ground_truth" / f"{page}.json").read_text(encoding="utf-8"))
    out = Parsed("ground_truth", page_image(page))
    md = []
    for r in gt["regions"]:
        out.blocks.append(Block(GT_TYPE.get(r["type"], "text"), tuple(r["box"])))
        if "html" in r:
            out.tables.append(html_rows(r["html"]))
            md.append(_table_html(r["html"]))
        elif "pairs" in r:
            out.pairs += [tuple(p) for p in r["pairs"]]
            md.append(_pairs_markdown(r["pairs"]))
        elif r["type"] == "heading":
            md.append(f"### {r['text']}")
        else:
            md.append(r["text"])
    out.markdown = "\n\n".join(md)
    return out


def table_cells(page: str) -> list[dict]:
    """The ground truth's table cells with their boxes, for cropping one row of a table."""
    return json.loads((DATA / "ground_truth" / f"{page}.json").read_text(encoding="utf-8"))["cells"]


def table_regions(page: str) -> list[tuple]:
    gt = json.loads((DATA / "ground_truth" / f"{page}.json").read_text(encoding="utf-8"))
    return [tuple(r["box"]) for r in gt["regions"] if r["type"] == "table"]


# --------------------------------------------------------------------------- saved outputs

PADDLE_TYPE = {"doc_title": "heading", "paragraph_title": "heading", "table": "table", "image": "figure",
               "figure": "figure", "chart": "figure"}


def _paddle(page: str) -> Parsed:
    folder = DATA / "cached" / "paddleocr_vl"
    res = json.loads((folder / f"{page}.json").read_text(encoding="utf-8"))
    out = Parsed("paddleocr_vl", page_image(page), markdown=(folder / f"{page}.md").read_text(encoding="utf-8"))
    for b in res["blocks"]:
        out.blocks.append(Block(PADDLE_TYPE.get(b["label"], "text"), tuple(b["box"])))
        if b["label"] == "table":
            out.tables.append(html_rows(b["content"]))
    out.pairs = pairs_from_tables(out.tables)
    return out


# A DocTags block: its type, a <location> whose four numbers are the box, and its <text>. The general
# model often writes the closing tags of the location wrongly, so only the numbers are read.
_DOCTAGS = re.compile(r"<(heading|paragraph|list|table|key_value_section|omr_question)>\s*"
                      r"<location>(.*?)</location>\s*<text>(.*?)</text>\s*</\1>", re.S)
_NUMBER = re.compile(r">\s*(-?\d+(?:\.\d+)?)\s*<")
QWEN_TYPE = {"heading": "heading", "paragraph": "text", "list": "list", "table": "table",
             "key_value_section": "key_value", "omr_question": "key_value"}


def _kv_lines(body: str) -> list[tuple[str, str]]:
    pairs = []
    for line in re.split(r"<br\s*/?>|\n", body):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                d = ast.literal_eval(line)
                pairs.append((str(d.get("key", "")), str(d.get("value", ""))))
                continue
            except (ValueError, SyntaxError):
                pass
        key, _, value = line.partition(":")
        pairs.append((key + ":", value.strip()) if value else (line, ""))
    return pairs


def _qwen(page: str, model: str) -> Parsed:
    raw = (DATA / "cached" / model / f"{page}.doctags.txt").read_text(encoding="utf-8")
    out = Parsed(model, page_image(page))
    md = []
    for m in _DOCTAGS.finditer(raw):
        tag, location, body = m.group(1), m.group(2), m.group(3).strip()
        numbers = [float(n) for n in _NUMBER.findall(location)]
        out.blocks.append(Block(QWEN_TYPE[tag], tuple(numbers) if len(numbers) == 4 else None))
        if tag == "table":
            out.tables.append(html_rows(body))
            md.append(_table_html(body))
        elif tag in ("key_value_section", "omr_question"):
            pairs = _kv_lines(body)
            out.pairs += pairs
            md.append(_pairs_markdown(pairs))
        elif tag == "heading":
            md.append(f"### {body}")
        else:
            md.append(body)
    out.markdown = "\n\n".join(md)
    return out


def cached(page: str, model: str) -> Parsed:
    """A saved output of a model that needs a GPU."""
    if model == "paddleocr_vl":
        return _paddle(page)
    if model in ("qwen_base", "qwen_finetuned"):
        return _qwen(page, model)
    raise ValueError(f"no saved outputs for {model!r}; saved: {', '.join(CACHED_MODELS)}")
