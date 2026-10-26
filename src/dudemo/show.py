"""Pictures for the notebook: regions drawn on the page, the Markdown a parser wrote, rendered, and one
table row as the page has it and as a parser wrote it."""

from __future__ import annotations

import base64
import html
import io
import re

import marko
from IPython.display import HTML, display
from PIL import Image, ImageDraw, ImageFont

from . import data
from .data import Parsed

COLORS = {"heading": "#E07B39", "text": "#7F95A6", "table": "#058ACC", "key_value": "#5E9E1F",
          "list": "#8E6BBF", "figure": "#C2185B", "header_footer": "#0F9D8F"}
LABELS = {"heading": "heading", "text": "text", "table": "table", "key_value": "key-value", "list": "list",
          "figure": "figure", "header_footer": "header / footer"}
MISSED = "#FF3B30"
LINE = 4
PAGE_WIDTH = 440        # px on screen

_markdown = marko.Markdown(extensions=["gfm"])

STYLE = """<style>
.dd-md {height: 720px; overflow: auto; border: 1px solid #d0d4d8; padding: 4px 14px; background: #fff;
        color: #111; font-size: 13px; line-height: 1.35; flex: 1; min-width: 320px}
.dd-md table {border-collapse: collapse; margin: 6px 0}
.dd-md td, .dd-md th {border: 1px solid #c4c9ce; padding: 2px 6px; text-align: left}
.dd-md h1, .dd-md h2, .dd-md h3 {font-size: 15px; margin: 10px 0 4px}
.dd-row td, .dd-row th {border: 1px solid #c4c9ce; padding: 3px 8px; font-size: 14px; background: #fff; color: #111;
                        text-align: left}
.dd-row th {background: #f1f3f5}
.dd-row .diff {background: #fde2e1; color: #b00020}
.dd-title {font-weight: 600; font-size: 16px; margin: 14px 0 6px}
</style>"""


# --------------------------------------------------------------------------- regions on the page

def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _dashed(draw, box, color, width=LINE, dash=14):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = max(abs(bx - ax), abs(by - ay))
        for start in range(0, int(length), 2 * dash):
            t0, t1 = start / length, min(start + dash, length) / length
            draw.line((ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1),
                      fill=color, width=width)


def regions_image(parsed: Parsed, missed_from: Parsed | None = None) -> Image.Image:
    """The page with every region the parser returned, outlined in its type's colour. With `missed_from`
    (the ground truth), the regions the parser missed are outlined in dashed red."""
    image = parsed.page.copy()
    draw = ImageDraw.Draw(image)
    boxes = [b.box for b in parsed.blocks if b.box and b.box[2] > b.box[0] and b.box[3] > b.box[1]]
    for block in parsed.blocks:
        if block.box in boxes:
            draw.rectangle(block.box, outline=COLORS.get(block.type, COLORS["text"]), width=LINE)
    if missed_from is not None:
        for region in missed_from.blocks:
            if not any(_iou(region.box, box) >= 0.5 for box in boxes):
                _dashed(draw, region.box, MISSED)
    return image


def _png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _legend(types, missed=False) -> str:
    items = [f'<span style="display:inline-block;width:12px;height:12px;border:3px solid {COLORS[t]};'
             f'margin:0 4px 0 12px;vertical-align:middle"></span>{LABELS[t]}' for t in COLORS if t in types]
    if missed:
        items.append(f'<span style="display:inline-block;width:12px;height:12px;border:3px dashed {MISSED};'
                     f'margin:0 4px 0 12px;vertical-align:middle"></span>missed')
    return f'<div style="font-size:13px;margin-top:6px">{"".join(items)}</div>'


def _rendered(markdown_text: str) -> str:
    return _markdown.convert(markdown_text)


def parse(parsed: Parsed, missed_from: Parsed | None = None):
    """The page with the parser's regions on the left, the Markdown it wrote, rendered, on the right."""
    image = regions_image(parsed, missed_from)
    source = html.escape(parsed.markdown)
    display(HTML(f"""{STYLE}
<div class="dd-title">{parsed.name}</div>
<div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
  <div><img src="{_png(image)}" width="{PAGE_WIDTH}" style="border:1px solid #d0d4d8">
       {_legend({b.type for b in parsed.blocks}, missed_from is not None)}</div>
  <div class="dd-md">{_rendered(parsed.markdown)}</div>
</div>
<details style="margin-top:6px"><summary style="cursor:pointer">Markdown as written</summary>
<pre style="white-space:pre-wrap;font-size:12px;max-height:400px;overflow:auto">{source}</pre></details>"""))


def side_by_side(*parsed: Parsed, missed_from: Parsed | None = None):
    """Two or more parsers' regions on the same page, drawn into one picture with each parser's name
    above its page, scaled to the width of the output so the pages stay in one row on any screen."""
    pages = [regions_image(p, missed_from) for p in parsed]
    gap, band = 40, 70
    font = ImageFont.load_default(size=40)
    width = sum(page.width for page in pages) + gap * (len(pages) - 1)
    height = band + max(page.height for page in pages)
    picture = Image.new("RGB", (width, height), "white")
    pen = ImageDraw.Draw(picture)
    x = 0
    for page, p in zip(pages, parsed):
        pen.text((x, 12), p.name, fill="#111111", font=font)
        picture.paste(page, (x, band))
        pen.rectangle((x, band, x + page.width - 1, band + page.height - 1), outline="#d0d4d8", width=2)
        x += page.width + gap
    types = {b.type for p in parsed for b in p.blocks}
    display(HTML(f'{STYLE}<img src="{_png(picture)}" style="width:100%;max-width:{width // 2}px">'
                 f'{_legend(types, missed_from is not None)}'))


def regions(parsed: Parsed, title: str | None = None):
    """One parser's regions on the page, at the page's own size, with the legend."""
    display(HTML(f'{STYLE}<div class="dd-title">{title or parsed.name}</div>'
                 f'<img src="{_png(regions_image(parsed))}" style="border:1px solid #d0d4d8">'
                 f'{_legend({b.type for b in parsed.blocks})}'))


def _coordinates(location: str) -> str:
    """The numbers inside <location>, one per line, as HTML. A closing tag that does not close the tag
    opened before it is red."""
    lines = []
    for piece in re.findall(r".*?</\w+>", "".join(location.split())):
        m = re.fullmatch(r"(?:<(\w+)>)?([^<]*)</(\w+)>", piece)
        if not m:
            lines.append(html.escape(piece))
            continue
        opened, number, closed = m.groups()
        end = html.escape(f"</{closed}>")
        if opened != closed:
            end = f'<span style="color:{MISSED};font-weight:700">{end}</span>'
        lines.append((html.escape(f"<{opened}>") if opened else "") + html.escape(number) + end)
    return "\n".join("  " + line for line in lines)


def first_tag(page: str, *parsed: Parsed):
    """Each Qwen VL model's first block as it wrote it, with the numbers of its <location> one per line.
    Below it, the box those numbers give on the top of the page, drawn just outside so its text stays visible."""
    bottom = min(p.page.height for p in parsed)
    bottom = min(bottom, max(p.blocks[0].box[3] for p in parsed) + 40)
    columns = []
    for p in parsed:
        tag, location, text = data.first_tag(page, p.model)
        text = " ".join(text.split())
        color = COLORS[p.blocks[0].type]
        top = p.page.crop((0, 0, p.page.width, bottom))
        x0, y0, x1, y1 = p.blocks[0].box
        ImageDraw.Draw(top).rectangle((x0 - LINE, y0 - LINE, x1 + LINE, y1 + LINE), outline=color, width=LINE)
        written = (f'{html.escape(f"<{tag}>")}\n'
                   f'<span style="display:block;border-left:4px solid {color};padding-left:6px">'
                   f'{html.escape("<location>")}\n{_coordinates(location)}\n{html.escape("</location>")}</span>'
                   f'{html.escape(f"<text>{text}</text>")}\n{html.escape(f"</{tag}>")}')
        columns.append(
            f'<div><div class="dd-title">{p.name}</div>'
            f'<pre style="white-space:pre-wrap;font-size:13px;background:#f6f8fa;color:#111;padding:8px 10px;'
            f'border:1px solid #d0d4d8;margin:0 0 10px">{written}</pre>'
            f'<img src="{_png(top)}" style="width:100%;border:1px solid #d0d4d8"></div>')
    display(HTML(f'{STYLE}<div style="display:grid;grid-template-columns:repeat({len(parsed)},minmax(0,1fr));'
                 f'gap:24px;max-width:{2 * PAGE_WIDTH + 24}px">{"".join(columns)}</div>'))


def field_error(page: str, label: str, *parsed: Parsed):
    """One key-value field: the field on the page, enlarged, what the ground truth holds for it, and what
    each parser wrote. A value that differs from the page is marked red."""
    f = data.field(page, label)
    x0, y0, x1, y1 = f["box"]
    crop = data.page_image(page).crop((x0 - 8, y0 - 3, x1 + 8, y1 + 3))
    rows = []
    for p in parsed:
        written = data.value_of(p, label)
        differs = written is None or data.key(written) != data.key(f["value"])
        shown = "(the field is missing)" if written is None else written
        rows.append(f'<tr><td>{p.name} wrote</td>'
                    f'<td class="{"diff" if differs else ""}"><code>{html.escape(shown)}</code></td></tr>')
    display(HTML(f"""{STYLE}
<img src="{_png(crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS))}" style="max-width:100%;border:1px solid #d0d4d8">
<table class="dd-row" style="margin-top:10px">
<tr><th></th><th><code>{html.escape(f["key"])}</code></th></tr>
<tr><td>On the page</td><td><code>{html.escape(f["value"])}</code></td></tr>
{"".join(rows)}</table>"""))


def wrong_values(page: str, *parsed: Parsed):
    """A table of the values on the page that at least one parser wrote differently or left out: the value
    on the page, and what each parser wrote, red where it differs from the page."""
    def cell(value, differs=False):
        shown = "(missing)" if value is None else value or "(empty)"
        return f'<td class="{"diff" if differs else ""}"><code>{html.escape(shown)}</code></td>'
    rows = "".join(
        f'<tr><td>{html.escape(where)}</td>{cell(value)}'
        + "".join(cell(g, g is None or not data.same(g, value)) for g in got) + "</tr>"
        for where, value, got in data.wrong_values(page, *parsed))
    header = "".join(f"<th>{p.name}</th>" for p in parsed)
    display(HTML(f'{STYLE}<table class="dd-row"><tr><th></th><th>On the page</th>{header}</tr>{rows}</table>'))


def image(picture: Image.Image, title: str):
    """One image at its own size."""
    display(HTML(f'{STYLE}<div class="dd-title">{title}</div>'
                 f'<img src="{_png(picture)}" style="border:1px solid #d0d4d8">'))


STATUS = {"passed": ("#2E9E5B", "passed the checks"), "fixed": ("#1A73E8", "fixed by a rule"),
          "person": ("#E89B00", "left for a person"), "right": ("#2E9E5B", "right"), "wrong": ("#D93025", "wrong")}


def _box(page_1: str, page_2: str, item: dict) -> tuple[int, tuple]:
    """Which page a checked row or field is on, and its box there: a row of the first report's tables, or a
    field of the second report, found by the closest label."""
    if "Parameter" in item:
        return 0, data.row_box(page_1, item["Parameter"])
    label, _ = data.pair_of(data.ground_truth(page_2), item["Field"])
    return 1, tuple(data.field(page_2, label)["box"])


def _two_pages(page_1: str, page_2: str, boxes: list[tuple[int, tuple, str]], titles: list[str], legend: list[str]):
    """Both reports in one picture, each box filled lightly in its status's colour, scaled to the output."""
    pages = [data.page_image(page_1).convert("RGBA"), data.page_image(page_2).convert("RGBA")]
    for n, box, status in boxes:
        color = STATUS[status][0]
        layer = Image.new("RGBA", pages[n].size, (0, 0, 0, 0))
        pen = ImageDraw.Draw(layer)
        x0, y0, x1, y1 = box
        pen.rectangle((x0 - LINE, y0 - LINE, x1 + LINE, y1 + LINE), fill=color + "40", outline=color, width=LINE)
        pages[n] = Image.alpha_composite(pages[n], layer)
    gap, band = 40, 70
    font = ImageFont.load_default(size=36)
    picture = Image.new("RGB", (sum(p.width for p in pages) + gap, band + max(p.height for p in pages)), "white")
    pen = ImageDraw.Draw(picture)
    x = 0
    for page, title in zip(pages, titles):
        pen.text((x, 14), title, fill="#111111", font=font)
        picture.paste(page.convert("RGB"), (x, band))
        pen.rectangle((x, band, x + page.width - 1, band + page.height - 1), outline="#d0d4d8", width=2)
        x += page.width + gap
    keys = "".join(f'<span style="display:inline-block;width:12px;height:12px;border:3px solid {STATUS[s][0]};'
                   f'background:{STATUS[s][0]}40;margin:0 4px 0 12px;vertical-align:middle"></span>{STATUS[s][1]}'
                   for s in legend)
    display(HTML(f'{STYLE}<img src="{_png(picture)}" style="width:100%;max-width:{picture.width // 2}px">'
                 f'<div style="font-size:13px;margin-top:6px">{keys}</div>'))


TITLES = ["Report 1: Docling's tables", "Report 2: patient fields"]


def statuses(page_1: str, rows: list[dict], page_2: str, fields: list[dict]):
    """Every checked row and field as a box on its report, in the colour of its status: passed the checks,
    fixed by a rule, or left for a person."""
    boxes = [(*_box(page_1, page_2, item), item["status"]) for item in rows + fields]
    _two_pages(page_1, page_2, boxes, TITLES, ["passed", "fixed", "person"])


def for_a_person(page_1: str, rows: list[dict], page_2: str, fields: list[dict]):
    """What is left for a person: that part of the page, enlarged, and what was written for it. The page is
    the only answer shown; the person reads it."""
    for item in [i for i in rows + fields if i["status"] == "person"]:
        n, (x0, y0, x1, y1) = _box(page_1, page_2, item)
        crop = data.page_image([page_1, page_2][n]).crop((x0 - 8, y0 - 3, x1 + 8, y1 + 3))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        if n == 0:
            written = _row_table(["Parametar", "Rezultat", "Jedinica", "Referentni interval", "Flag"],
                                 [item[c] for c in ("Parameter", "Result", "Unit", "Range", "Flag")])
        else:
            written = '<table class="dd-row">' + "".join(
                f'<tr><td>{name} wrote</td><td><code>{html.escape(item[name])}</code></td></tr>'
                for name in ("Qwen VL, fine-tuned", "PaddleOCR-VL")) + "</table>"
        display(HTML(f'{STYLE}<div class="dd-title">{html.escape(item.get("Parameter") or item["Field"])}</div>'
                     f'<img src="{_png(crop)}" style="max-width:100%;border:1px solid #d0d4d8">{written}'))


def score(page_1: str, rows: list[dict], page_2: str, fields: list[dict]):
    """Every row and field, as it ends, held against the ground truth: right or wrong on its report, and how
    many are right in the end."""
    truth_1, truth_2 = data.ground_truth(page_1), data.ground_truth(page_2)
    boxes, right = [], 0
    for item in rows + fields:
        if "Parameter" in item:
            _, on_page = find_row(truth_1.tables, item["Parameter"])
            written = [item[c] for c in ("Parameter", "Result", "Unit", "Range", "Flag")]
            ok = all(data.same(w, p) for w, p in zip(written, on_page))
        else:
            _, on_page = data.pair_of(truth_2, item["Field"])
            ok = item["Value"] is not None and data.same(item["Value"], on_page)
        right += ok
        boxes.append((*_box(page_1, page_2, item), "right" if ok else "wrong"))
    _two_pages(page_1, page_2, boxes, TITLES, ["right", "wrong"])
    total = len(rows) + len(fields)
    display(HTML(f'<div style="font-size:20px;font-weight:700;margin:10px 0">Right in the end: {right} of {total} '
                 f'({100 * right / total:.0f} %)</div>'))


def markdown(parsed: Parsed | str, title: str | None = None, closed: bool = False):
    """Rendered Markdown, of a parser's output or of a Markdown string; with `closed`, folded under its
    title until clicked."""
    text = parsed if isinstance(parsed, str) else parsed.markdown
    title = title or (parsed.name if isinstance(parsed, Parsed) else "Markdown")
    body = f'<div class="dd-md" style="height:auto">{_rendered(text)}</div>'
    if closed:
        display(HTML(f'{STYLE}<details><summary class="dd-title" style="cursor:pointer">{title}'
                     f'</summary>{body}</details>'))
    else:
        display(HTML(f'{STYLE}<div class="dd-title">{title}</div>{body}'))


# --------------------------------------------------------------------------- one table row

def find_row(tables, label: str):
    """The header row and the row whose first cell is `label`, from the first table that has it."""
    for rows in tables:
        for row in rows[1:]:
            if row and data.key(row[0]) == data.key(label):
                return rows[0], row
    return None, None


def _row_table(header, row, against=None) -> str:
    """One row with its header. With `against` (the ground truth's row), every written cell that is not
    in that row is marked red, whatever the row's shape."""
    if row is None:
        return '<p style="color:#b00020;font-size:14px">(the row is missing)</p>'
    def differs(cell):
        return against is not None and cell.strip() and not any(data.same(cell, c) for c in against)
    cells = "".join(f'<td class="{"diff" if differs(c) else ""}"><code>{html.escape(c)}</code></td>' for c in row)
    head = "".join(f"<th><code>{html.escape(c)}</code></th>" for c in header)
    return f'<table class="dd-row"><tr>{head}</tr><tr>{cells}</tr></table>'


def row_crop(page: str, label: str, pad: int = 8) -> Image.Image:
    """The band of the page that holds the table row whose first cell is `label`, enlarged."""
    x0, y0, x1, y1 = data.row_box(page, label)
    crop = data.page_image(page).crop((x0 - pad, y0 - 2, x1 + pad, y1 + 3))
    return crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)


def row_error(page: str, label: str, parsed: Parsed):
    """One table row: the row on the page, enlarged; the row as the page has it; and the row as the parser
    wrote it, with every cell that is not in the page's row marked red."""
    header, row = find_row(data.ground_truth(page).tables, label)
    out_header, out_row = find_row(parsed.tables, label)
    display(HTML(f"""{STYLE}
<img src="{_png(row_crop(page, label))}" style="max-width:100%;border:1px solid #d0d4d8">
<p style="margin:10px 0 2px;font-size:13px;color:#555">On the page</p>{_row_table(header, row)}
<p style="margin:10px 0 2px;font-size:13px;color:#555">{parsed.name} wrote</p>
{_row_table(out_header or [], out_row, against=row)}"""))


def table(frame):
    """A table without its index column, with every failed check (a cell starting with ✗) marked red. A
    parser's table can have two columns of the same name; it is shown as it is, without the marking pandas
    cannot do there."""
    styled = frame.style.hide(axis="index").set_table_styles(
        [{"selector": "td, th", "props": "font-size:14px;padding:3px 10px;text-align:left"}])
    if frame.columns.is_unique:
        styled = styled.map(lambda v: "background:#fde2e1;color:#b00020;font-weight:600" if str(v).startswith("✗") else "")
    display(styled)
