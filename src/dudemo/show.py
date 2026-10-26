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


def compare(*parsed: Parsed, missed_from: Parsed | None = None):
    """Two or more parsers next to each other: each one's regions, and its rendered Markdown below."""
    columns = "".join(
        f'<div style="width:{PAGE_WIDTH}px"><div class="dd-title">{p.name}</div>'
        f'<img src="{_png(regions_image(p, missed_from))}" width="{PAGE_WIDTH}" style="border:1px solid #d0d4d8">'
        f'<div class="dd-md" style="height:420px;margin-top:10px">{_rendered(p.markdown)}</div></div>'
        for p in parsed)
    types = {b.type for p in parsed for b in p.blocks}
    display(HTML(f'{STYLE}<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">{columns}</div>'
                 f'{_legend(types, missed_from is not None)}'))


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


def image(picture: Image.Image, title: str):
    """One image at its own size."""
    display(HTML(f'{STYLE}<div class="dd-title">{title}</div>'
                 f'<img src="{_png(picture)}" style="border:1px solid #d0d4d8">'))


def page_break(end_of_page: Image.Image, top_of_next: Image.Image):
    """Two parts of a table, one above the other, with the page break between them."""
    display(HTML(f"""{STYLE}
<div class="dd-title">End of page 1</div><img src="{_png(end_of_page)}" style="border:1px solid #d0d4d8">
<div style="border-top:3px dashed #8a9199;margin:14px 0 4px;width:{end_of_page.width}px"></div>
<div style="font-size:13px;color:#555">page break</div>
<div class="dd-title">Top of page 2</div><img src="{_png(top_of_next)}" style="border:1px solid #d0d4d8">"""))


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
    present = {data.key(c) for c in against} if against is not None else None
    cells = "".join(
        f'<td class="{"diff" if present is not None and c.strip() and data.key(c) not in present else ""}">'
        f'<code>{html.escape(c)}</code></td>' for c in row)
    head = "".join(f"<th><code>{html.escape(c)}</code></th>" for c in header)
    return f'<table class="dd-row"><tr>{head}</tr><tr>{cells}</tr></table>'


def row_crop(page: str, label: str, pad: int = 8) -> Image.Image:
    """The band of the page that holds the table row whose first cell is `label`, enlarged."""
    x0, y0, x1, y1 = data.row_box(page, label)
    crop = data.page_image(page).crop((x0 - pad, y0 - 2, x1 + pad, y1 + 3))
    return crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)


def error(example: dict, parsed: Parsed, truth: Parsed):
    """One error: the row on the page, the row as the ground truth has it, and as the parser wrote it.
    Cells that differ from the ground truth are marked red."""
    header, row = find_row(truth.tables, example["row"])
    out_header, out_row = find_row(parsed.tables, example["row"])
    display(HTML(f"""{STYLE}
<div class="dd-title">{example["kind"]}: {parsed.name}</div>
<p style="margin:2px 0 8px">{example["note"]}</p>
<img src="{_png(row_crop(example["page"], example["row"]))}" style="max-width:100%;border:1px solid #d0d4d8">
<p style="margin:10px 0 2px;font-size:13px;color:#555">On the page</p>{_row_table(header, row)}
<p style="margin:10px 0 2px;font-size:13px;color:#555">{parsed.name} wrote</p>
{_row_table(out_header or [], out_row, against=row)}"""))


def table(frame):
    """A table without its index column, with every failed check (✗) marked red. A parser's table can
    have two columns of the same name; it is shown as it is, without the marking pandas cannot do there."""
    styled = frame.style.hide(axis="index").set_table_styles(
        [{"selector": "td, th", "props": "font-size:14px;padding:3px 10px;text-align:left"}])
    if frame.columns.is_unique:
        styled = styled.map(lambda v: "background:#fde2e1;color:#b00020;font-weight:600" if v == "✗" else "")
    display(styled)
