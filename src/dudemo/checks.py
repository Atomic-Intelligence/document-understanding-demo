"""Simple checks on a parser's output that need no ground truth: three rules on each lab result, and a
comparison of the patient fields with a second parser."""

from __future__ import annotations

import difflib
import re

import pandas as pd

from .data import Parsed

UNITS = {"10⁹/L", "10¹²/L", "g/L", "mmol/L", "µmol/L", "μmol/L", "U/L", "%", "fL", "pg", "L/L", "mg/L",
         "µg/L", "ng/mL", "nmol/L", "pmol/L", "mU/L", "mIU/L", "IU/L", "kU/L", "mmol/mol", "s"}
_NUM = r"\d+(?:[.,]\d+)?"
_RANGE = re.compile(rf"(?P<lo>{_NUM})\s*-\s*(?P<hi>{_NUM})|(?P<op>[<>≤≥])\s*(?P<lim>{_NUM})")
PASS, FAIL, NOT_CHECKED = "✓", "✗", "–"


def _number(s: str) -> float | None:
    m = re.fullmatch(_NUM, s.strip())
    return float(m.group().replace(",", ".")) if m else None


def unit_known(unit: str) -> str:
    return PASS if unit.replace(" ", "") in UNITS else FAIL


def range_shaped(ref: str) -> str:
    return PASS if _RANGE.fullmatch(ref.strip()) else FAIL


def flag_fits(value: str, ref: str, flag: str) -> str:
    """A printed H or L must agree with the result and the range. Rows without a flag are not checked."""
    flag = flag.strip().upper()
    v, m = _number(value), _RANGE.fullmatch(ref.strip())
    if flag not in ("H", "L") or v is None or not m:
        return NOT_CHECKED
    num = lambda s: float(s.replace(",", "."))  # noqa: E731
    if m["lo"]:
        expected = "L" if v < num(m["lo"]) else "H" if v > num(m["hi"]) else ""
    elif m["op"] in "<≤":
        expected = "H" if v > num(m["lim"]) else ""
    else:
        expected = "L" if v < num(m["lim"]) else ""
    return PASS if expected == flag else FAIL


def lab_results(parsed: Parsed) -> list[dict]:
    """Every result row of the parser's lab tables. The range is everything between the unit and the
    flag column, so a range split over two columns is read as written."""
    rows = []
    for table in parsed.tables:
        header = [h.casefold() for h in table[0]] if table else []
        if not any("rezultat" in h for h in header) or not any("jedinica" in h for h in header):
            continue
        value_col = next(i for i, h in enumerate(header) if "rezultat" in h)
        unit_col = next(i for i, h in enumerate(header) if "jedinica" in h)
        for row in table[1:]:
            if len(row) <= unit_col + 1:
                continue
            rows.append({"Parameter": row[0], "Result": row[value_col], "Unit": row[unit_col],
                         "Range": " ".join(c for c in row[unit_col + 1:-1] if c), "Flag": row[-1]})
    return rows


def check_results(parsed: Parsed) -> pd.DataFrame:
    """One line per lab result, with a mark per rule."""
    lines = []
    for r in lab_results(parsed):
        lines.append({**r,
                      "unit is a known unit": unit_known(r["Unit"]),
                      "range looks like a range": range_shaped(r["Range"]),
                      "flag fits result and range": flag_fits(r["Result"], r["Range"], r["Flag"])})
    return pd.DataFrame(lines)


def _norm(s: str) -> str:
    return " ".join(s.replace(":", " ").split()).casefold()


def compare_fields(first: Parsed, second: Parsed) -> pd.DataFrame:
    """Each patient field the second parser found, as both parsers wrote it, and whether they agree. Keys
    are matched loosely, so a small misspelling of a label still finds its field."""
    fields = [key for key, _ in second.pairs]

    def lookup(parsed, field):
        keys = {_norm(k): v for k, v in parsed.pairs}
        match = difflib.get_close_matches(_norm(field), list(keys), n=1, cutoff=0.75)
        return keys[match[0]] if match else "(missing)"

    lines = []
    for field in fields:
        a, b = lookup(first, field), lookup(second, field)
        lines.append({"Field": field.rstrip(":"), first.name: a, second.name: b,
                      "same in both": PASS if _norm(a) == _norm(b) else FAIL})
    return pd.DataFrame(lines)
