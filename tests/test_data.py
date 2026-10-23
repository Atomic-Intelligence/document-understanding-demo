"""Reading the pages, the ground truth and the saved outputs."""

from dudemo import data
from dudemo.show import find_row


def test_every_qwen_block_has_a_box_on_the_page():
    for model in ("qwen_base", "qwen_finetuned"):
        for page in ("lab_report_1", "lab_report_2"):
            parsed = data.cached(page, model)
            width, height = parsed.page.size
            assert parsed.blocks
            for block in parsed.blocks:
                x0, y0, x1, y1 = block.box
                assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height


def test_ground_truth_row_and_fields():
    truth = data.ground_truth("lab_report_1")
    header, row = find_row(truth.tables, "ALT (ALAT)")
    assert header[0] == "Parametar"
    assert row == ["ALT (ALAT)", "55", "U/L", "< 45", "H"]
    assert ("Matični broj:", "145098-68") in truth.pairs


def test_saved_outputs_have_tables_and_fields():
    for model in data.CACHED_MODELS:
        parsed = data.cached("lab_report_1", model)
        assert len(parsed.tables) >= 2
        assert parsed.pairs
        assert parsed.markdown


def test_html_rows():
    rows = data.html_rows("<table><tr><th>a</th><th>b</th></tr><tr><td>1<br>2</td><td> x </td></tr></table>")
    assert rows == [["a", "b"], ["1 2", "x"]]


def test_every_example_points_at_a_row_on_its_page():
    for example in data.examples():
        _, row = find_row(data.ground_truth(example["page"]).tables, example["row"])
        assert row is not None
