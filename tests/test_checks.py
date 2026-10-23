"""The rules of section 6 and the comparison with a second parser."""

from dudemo import checks, data
from dudemo.checks import FAIL, NOT_CHECKED, PASS


def test_unit_known():
    assert checks.unit_known("10⁹/L") == PASS
    assert checks.unit_known("109/L") == FAIL
    assert checks.unit_known("µmol/L") == PASS


def test_range_shaped():
    assert checks.range_shaped("4,4 - 6,1") == PASS
    assert checks.range_shaped("< 45") == PASS
    assert checks.range_shaped("4,4 6,1") == FAIL
    assert checks.range_shaped("3,9") == FAIL


def test_flag_fits():
    assert checks.flag_fits("55", "< 45", "H") == PASS
    assert checks.flag_fits("55", "> 45", "H") == FAIL      # the reversed sign
    assert checks.flag_fits("6,9", "4,4 - 6,1", "H") == PASS
    assert checks.flag_fits("3,0", "4,4 - 6,1", "H") == FAIL
    assert checks.flag_fits("22", "< 35", "") == NOT_CHECKED


def test_second_parser_disagrees_on_the_registration_number():
    table = checks.compare_fields(data.cached("lab_report_1", "qwen_base"),
                                  data.cached("lab_report_1", "paddleocr_vl"))
    row = table[table["Field"] == "Matični broj"].iloc[0]
    assert row["same in both"] == FAIL
    assert (table["same in both"] == PASS).sum() > len(table) / 2
