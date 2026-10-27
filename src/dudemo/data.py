"""Pages, ground truth and saved model outputs, all read into one shape: `Parsed`."""

from __future__ import annotations

import ast
import dataclasses
import html
import json
import re
import time
from dataclasses import dataclass
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
ENGINES = ["docling", *CACHED_MODELS]

# The engines to run in the notebook instead of loading their saved outputs (the notebook sets it in its
# setup; see live.py). Each one reads a page the first time the notebook needs it, into a folder of its own
# for this session, so a new session runs them again.
LIVE: set[str] = set()
LIVE_DIR = DATA.parent / "outputs" / "live" / time.strftime("%Y%m%d-%H%M%S")


@dataclass
class Block:
    type: str                       # heading, text, table, key_value, list, figure, header_footer
    box: tuple | None               # (x0, y0, x1, y1) in page pixels, top-left origin


@dataclass
class Parsed:
    """What one parser returned for one page."""
    model: str
    page: Image.Image
    blocks: list[Block] = dataclasses.field(default_factory=list)
    markdown: str = ""
    tables: list[list[list[str]]] = dataclasses.field(default_factory=list)   # table -> row -> cell text
    pairs: list[tuple[str, str]] = dataclasses.field(default_factory=list)    # key-value fields

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


def field(page: str, label: str) -> dict:
    """A key-value field of the ground truth: its key, its value, and one box around both."""
    fields = json.loads((DATA / "ground_truth" / f"{page}.json").read_text(encoding="utf-8"))["fields"]
    return next(f for f in fields if key(f["key"]) == key(label))


_SUPERSCRIPT = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _caret(text: str) -> str:
    """Superscript digits written with a caret, 10⁹ as 10^9, and spacing made even."""
    return " ".join(_SUPERSCRIPT.sub(lambda m: "^" + m.group().translate(_DIGITS), text).split())


def same(a: str | None, b: str | None) -> bool:
    """Two values read the same: equal apart from spacing, with 10⁹ and 10^9 taken as one notation. A lost
    superscript is a difference: 109 is not 10⁹. Nothing is the same only as nothing."""
    if a is None or b is None:
        return a is b
    return _caret(a) == _caret(b)


def values(parsed: Parsed) -> dict[str, str]:
    """Every value a parser wrote, by where it is: a key-value field by its key, a table cell as
    'row · column', with the row named by its first cell and the column by the table's first row."""
    out = dict(parsed.pairs)
    for table in parsed.tables:
        for row in table[1:]:
            for column, cell in zip(table[0][1:], row[1:]):
                out[f"{row[0]} · {column}"] = cell
    return out


def _plain(text: str) -> str:
    """A label's key without accents, so LIJĖCNIK and LIJEČNIK are one label: only for finding a place, never
    for deciding which reading is right."""
    import unicodedata
    text = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    return key("".join(c for c in text if not unicodedata.combining(c)))


def _closest(name: str, names) -> str | None:
    """The name in `names` that folds to the same key as `name`, then the same without accents, or else the
    closest one without accents, if close enough. An exact match comes first: HDL and LDL KOLESTEROL differ
    by one letter."""
    import difflib
    folded = {key(n): n for n in names}
    if key(name) in folded:
        return folded[key(name)]
    plain = {_plain(n): n for n in names}
    if _plain(name) in plain:
        return plain[_plain(name)]
    match = difflib.get_close_matches(_plain(name), list(plain), n=1, cutoff=0.8)
    return plain[match[0]] if match else None


def read(parsed: Parsed, where: str) -> str | None:
    """What a parser wrote at a place of the page: a field by its key, or a table value as 'row · column'.
    Labels, row names and headers are found by the closest name, since parsers misspell them too."""
    if " · " not in where:
        fields = dict(parsed.pairs)
        found = _closest(where, fields)
        return fields[found] if found else None
    rows = {row[0]: dict(zip(table[0][1:], row[1:])) for table in parsed.tables for row in table[1:] if row}
    name, column = where.split(" · ")
    row = _closest(name, rows)
    found = _closest(column, rows[row]) if row else None
    return rows[row][found] if found else None


def _text_place(text: str) -> str:
    """The place of a title or paragraph: 'Text:' and its first words."""
    words = text.split()
    return "Text: " + " ".join(words[:4]) + (" …" if len(words) > 4 else "")


def _text_regions(page: str) -> list[dict]:
    gt = json.loads((DATA / "ground_truth" / f"{page}.json").read_text(encoding="utf-8"))
    return [r for r in gt["regions"] if "text" in r]


def text_box(page: str, where: str) -> tuple:
    """The box of the title or paragraph at a 'Text: …' place."""
    return tuple(next(r["box"] for r in _text_regions(page) if _text_place(r["text"]) == where))


def page_values(page: str, text: bool = False) -> dict[str, str]:
    """Every value of the page by its place, from the page's record: each field, each table value and,
    with `text`, each title and paragraph."""
    out = values(ground_truth(page))
    if text:
        out |= {_text_place(r["text"]): " ".join(r["text"].split()) for r in _text_regions(page)}
    return out


def read_text(parsed: Parsed, text: str) -> str | None:
    """What a parser wrote for a title or paragraph of the page: the part of its text that lines up with
    it, or None if it left it out. A parser can put two lines of the page in one block, so the block is
    cut to the part that matches; what it wrote differently inside that part stays."""
    import difflib
    chunks = [" ".join(c.lstrip("# ").split()) for c in parsed.markdown.split("\n\n")]
    chunks = [c for c in chunks if c and not c.startswith(("<table", "**", "|"))]

    def lined_up(chunk):
        blocks = [b for b in difflib.SequenceMatcher(None, chunk, text, autojunk=False).get_matching_blocks()
                  if b.size]
        return sum(b.size for b in blocks), blocks

    best = max(chunks, key=lambda c: lined_up(c)[0], default=None)
    if best is None or lined_up(best)[0] < len(text) / 2:
        return None
    blocks = lined_up(best)[1]
    start, end = blocks[0].a - blocks[0].b, blocks[-1].a + blocks[-1].size + (len(text) - blocks[-1].b - blocks[-1].size)
    found = best[max(start, 0):end].strip()
    # Letters of a short line match here and there in any long text: what lines up must read like the
    # line, or the parser left it out.
    return found if difflib.SequenceMatcher(None, found, text).ratio() >= 0.8 else None


def readings(page: str, *parsed: Parsed, text: bool = False) -> list[tuple[str, list[str | None]]]:
    """Every value of the page, by its place, with what each parser wrote there (None for nothing); with
    `text`, every title and paragraph too. The places come from the page's record, so a value that no
    parser wrote is counted too."""
    truth = page_values(page, text)
    return [(where, [read_text(p, truth[where]) if where.startswith("Text: ") else read(p, where)
                     for p in parsed]) for where in truth]


def with_answers(decided: list[tuple[str, str | None, str]], readings: list[tuple[str, list]],
                 answers: dict) -> list[tuple[str, str | None]]:
    """Every place as it ends: what was decided, or, for a place left for a person, the person's answer. For
    a field or table value the answer is the value. For a title or paragraph it is a list with the words
    the page has at each spot where the two readings differ, in order; the rest of the text is where they
    agree."""
    import difflib
    written = dict(readings)
    final = []
    for where, value, status in decided:
        answer = answers.get(where)
        if status == "person" and where.startswith("Text: ") and answer is not None:
            words, other = written[where][0].split(), written[where][1].split()
            spots = [op for op in difflib.SequenceMatcher(None, words, other).get_opcodes() if op[0] != "equal"]
            if len(spots) != len(answer):
                answer = None
            else:
                for (_, i1, i2, _, _), said in reversed(list(zip(spots, answer))):
                    words[i1:i2] = said.split()
                answer = " ".join(words)
        final.append((where, answer if status == "person" else value))
    return final


def wrong_values(page: str, *parsed: Parsed) -> list[tuple[str, str, list[str | None]]]:
    """The values of the page that at least one parser wrote differently or left out: where the value is,
    the value on the page, and what each parser wrote there (None when it has nothing there)."""
    truth = page_values(page)
    return [(where, truth[where], got) for where, got in readings(page, *parsed)
            if not all(g is not None and same(g, truth[where]) for g in got)]


def row_box(page: str, label: str) -> tuple:
    """The box (x0, y0, x1, y1) of the table row whose first cell is `label`, as wide as its table."""
    cell = next(c for c in table_cells(page) if key(c["text"]) == key(label))
    table = table_box(page, label)
    return (table[0], cell["box"][1], table[2], cell["box"][3])


def table_box(page: str, label: str, pad: float = 0) -> tuple:
    """The box of the table that holds the row whose first cell is `label`, widened by `pad` pixels. Two
    tables can stand side by side, so the cell must be inside the table both ways."""
    cell = next(c for c in table_cells(page) if key(c["text"]) == key(label))
    x0, y0, x1, y1 = next(t for t in table_regions(page)
                          if t[0] <= cell["box"][0] <= t[2] and t[1] <= cell["box"][1] <= t[3])
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


# --------------------------------------------------------------------------- saved outputs

PADDLE_TYPE = {"doc_title": "heading", "paragraph_title": "heading", "table": "table", "image": "figure",
               "figure": "figure", "chart": "figure", "header": "header_footer", "footer": "header_footer",
               "number": "header_footer"}


def saved_note(engine: str, page: str) -> str:
    """Where a saved output comes from: when it was written, how long the engine took for the page on each
    device it was timed on, and, if it ran on a vLLM server, how long that took to start
    (data/cached/runs.json)."""
    run = json.loads((DATA / "cached" / "runs.json").read_text(encoding="utf-8"))[engine]
    times = [f"{seconds:.0f} s on {device}" for device, seconds in run["pages"][page].items()]
    server = f", through a vLLM server ({run['server_start_s']:.0f} s to start)" if "server_start_s" in run else ""
    return (f"{NAMES[engine]}, {page}: loaded from data/cached/, written on {run['date']}. It took "
            + (", and ".join(times)) + server + ".")


def output(model: str, page: str, say: bool = False) -> Path:
    """The file with a model's output on a page: its saved copy in data/cached/, or, for a model in LIVE, the
    one it writes when it reads the page here, the first time it is asked for. With `say`, it prints which
    (a new live reading prints its own times), so a cell run again says it again."""
    name = f"{page}.json" if model == "paddleocr_vl" else f"{page}.doctags.txt"
    if model not in LIVE:
        if say:
            print(saved_note(model, page))
        return DATA / "cached" / model / name
    path = LIVE_DIR / model / name
    if not path.exists():
        from . import live
        live.run(model, page, path.parent)
    elif say:
        print(f"{NAMES[model]}, {page}: read live earlier in this session.")
    return path


def _paddle(page: str) -> Parsed:
    """PaddleOCR-VL's blocks, with the Markdown written from all of them in reading order. Its own Markdown
    export (the .md file) leaves out the blocks it labels as header and footer."""
    res = json.loads(output("paddleocr_vl", page, say=True).read_text(encoding="utf-8"))
    out = Parsed("paddleocr_vl", page_image(page))
    md = []
    for b in res["blocks"]:
        kind = PADDLE_TYPE.get(b["label"], "text")
        out.blocks.append(Block(kind, tuple(b["box"])))
        if b["label"] == "table":
            out.tables.append(html_rows(b["content"]))
        md.append(f"### {b['content']}" if kind == "heading" else b["content"])
    out.markdown = "\n\n".join(md)
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
    raw = output(model, page, say=True).read_text(encoding="utf-8")
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
    out.pairs += pairs_from_tables(out.tables)    # fields it wrote as a table, as PaddleOCR-VL does
    return out


def first_tag(page: str, model: str) -> tuple[str, str, str]:
    """The first block of a Qwen VL answer as the model wrote it: its tag, what is inside its <location>,
    and its text."""
    raw = output(model, page).read_text(encoding="utf-8")
    m = _DOCTAGS.search(raw)
    if m is None:
        raise ValueError(f"no DocTags block in the output of {model} for {page}")
    return m.group(1), m.group(2), m.group(3).strip()


def cached(page: str, model: str) -> Parsed:
    """The output of a model that needs a GPU: its saved copy, or, for a model in LIVE, what it reads now."""
    if model == "paddleocr_vl":
        return _paddle(page)
    if model in ("qwen_base", "qwen_finetuned"):
        return _qwen(page, model)
    raise ValueError(f"no saved outputs for {model!r}; saved: {', '.join(CACHED_MODELS)}")


# --------------------------------------------------------------------------- Docling

def docling_document(page: str):
    """Docling's saved DoclingDocument for a page, with its page image, as section 2.2 would have it."""
    from docling_core.types.doc import DoclingDocument
    print(saved_note("docling", page))
    return DoclingDocument.load_from_json(DATA / "cached" / "docling" / f"{page}.json")


DOCLING_TYPE = {"title": "heading", "section_header": "heading", "table": "table", "document_index": "table",
                "list_item": "list", "key_value_region": "key_value", "form": "key_value",
                "picture": "figure", "chart": "figure", "page_header": "header_footer",
                "page_footer": "header_footer"}


def from_docling(doc) -> Parsed:
    """A DoclingDocument's first page in the notebook's shape: regions with boxes in the page image's
    pixels, top-left origin; the tables' cells; and the Markdown."""
    page = doc.pages[min(doc.pages)]
    image = page.image.pil_image.convert("RGB")
    scale = image.width / page.size.width
    out = Parsed("docling", image, markdown=doc.export_to_markdown())
    for item, _level in doc.iterate_items():
        prov = [p for p in getattr(item, "prov", []) if p.page_no == page.page_no]
        if prov:
            b = prov[0].bbox.to_top_left_origin(page.size.height)
            out.blocks.append(Block(DOCLING_TYPE.get(str(item.label.value), "text"),
                                    (b.l * scale, b.t * scale, b.r * scale, b.b * scale)))
    out.tables = [[[cell.text for cell in row] for row in table.data.grid] for table in doc.tables]
    return out
