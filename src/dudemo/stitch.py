"""A page break through a table, made by cutting the page, and the repair: both parts stacked into one
image so that the header travels with the rows that continue the table."""

from __future__ import annotations

from PIL import Image

from . import data


def split_table(page: str, after_row: str, pad: int = 8) -> tuple[Image.Image, Image.Image]:
    """The table that holds `after_row`, cut below that row: the part with the header ("end of page 1")
    and the rest ("top of page 2")."""
    cells = data.table_cells(page)
    row = next(c for c in cells if data.key(c["text"]) == data.key(after_row))
    table = next(t for t in data.table_regions(page) if t[1] <= row["box"][1] <= t[3])
    below = [c["box"][1] for c in cells if c["box"][1] > row["box"][3] - 2 and c["box"][1] <= table[3]]
    cut = (row["box"][3] + min(below)) / 2 if below else row["box"][3]
    image = data.page_image(page)
    left, right = table[0] - pad, table[2] + pad
    return (image.crop((left, table[1] - pad, right, cut)),
            image.crop((left, cut, right, table[3] + pad)))


def stack(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """Both parts in one image, one above the other."""
    out = Image.new("RGB", (max(top.width, bottom.width), top.height + bottom.height), "white")
    out.paste(top, (0, 0))
    out.paste(bottom, (0, top.height))
    return out


def with_margin(image: Image.Image, margin: int = 40) -> Image.Image:
    """White space around a crop, so that a layout model sees it as a page."""
    out = Image.new("RGB", (image.width + 2 * margin, image.height + 2 * margin), "white")
    out.paste(image, (margin, margin))
    return out
