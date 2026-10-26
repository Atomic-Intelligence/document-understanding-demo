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


def test_row_lies_inside_its_table():
    table = data.table_box("lab_report_1", "Hematokrit (Hct)")
    row = data.row_box("lab_report_1", "Hematokrit (Hct)")
    assert table[0] == row[0] and table[2] == row[2]
    assert table[1] < row[1] < row[3] < table[3]


def test_a_row_is_in_its_own_table_when_two_stand_side_by_side():
    left, right = data.table_box("lab_report_2", "LEUKOCITI (WBC)"), data.table_box("lab_report_2", "GLUKOZA (natašte)")
    assert left[2] < right[0]


def test_first_tag_is_the_first_block():
    for model in ("qwen_base", "qwen_finetuned"):
        tag, location, text = data.first_tag("lab_report_1", model)
        first = data.cached("lab_report_1", model).blocks[0]
        assert data.QWEN_TYPE[tag] == first.type
        assert [float(n) for n in data._NUMBER.findall(location)] == list(first.box)
        assert text == "KLINIČKI BOLNIČKI CENTAR ZAGREB"


def test_fine_tuned_writes_every_wrong_value_right():
    """Step 4.3 says so: on the first page, each value one of the Qwen models gets wrong, the fine-tuned one
    has right."""
    general, fine_tuned = (data.cached("lab_report_1", m) for m in ("qwen_base", "qwen_finetuned"))
    wrong = data.wrong_values("lab_report_1", general, fine_tuned)
    assert wrong
    for _, value, (_, written) in wrong:
        assert written is not None and data.same(written, value)


def test_read_finds_misspelled_labels_and_headers():
    paddle = data.cached("lab_report_2", "paddleocr_vl")       # writes LIJEĆNIK, Referentini, nataste
    assert data.read(paddle, "LIJEČNIK:") == "Dr. Ivić"
    assert data.read(paddle, "LEUKOCITI (WBC) · Referentni interval") == "3,4 - 9,7"
    assert data.read(paddle, "GLUKOZA (natašte) · Rezultat") == "5,25"
    qwen = data.cached("lab_report_2", "qwen_finetuned")
    assert data.read(qwen, "HDL KOLESTEROL · Rezultat") == "1,35"      # not LDL's, one letter away


def test_read_text_cuts_a_block_to_the_part_that_lines_up():
    paddle = data.cached("lab_report_2", "paddleocr_vl")      # puts the hospital's name and address in one block
    name = 'KLINIČKI BOLNIČKI CENTAR "DR. FRAN MIHALJEVIĆ"'
    assert data.read_text(paddle, name) == name
    assert data.read_text(paddle, "(Ivan Perić)") == "(Ivan Peric)"


def test_read_text_finds_nothing_for_a_line_the_parser_left_out():
    qwen = data.cached("lab_report_1", "qwen_finetuned")         # leaves out the signature block
    assert data.read_text(qwen, "Dr. Petrović N.") is None
    assert data.read_text(qwen, "Digitalni potpis validiran") is None


def test_two_parsers_never_agree_on_a_wrong_value_on_the_second_report():
    """Step 6.4 keeps every value, title and paragraph the two parsers agree on."""
    truth = data.page_values("lab_report_2", text=True)
    parsers = [data.cached("lab_report_2", m) for m in ("qwen_finetuned", "paddleocr_vl")]
    for where, (qwen, paddle) in data.readings("lab_report_2", *parsers, text=True):
        if qwen is not None and data.same(qwen, paddle):
            assert data.same(qwen, truth[where]), where


def test_answers_fill_in_the_words_where_two_readings_differ():
    parsers = [data.cached("lab_report_2", m) for m in ("qwen_finetuned", "paddleocr_vl")]
    readings = data.readings("lab_report_2", *parsers, text=True)
    where = "Text: Pronađene su povišene koncentracije …"
    decided = [(where, None, "person")]
    [(_, filled)] = data.with_answers(decided, readings, {where: ["Pronađene", "daljnjeg", "praćenja"]})
    assert filled == data.page_values("lab_report_2", text=True)[where]
    [(_, too_few)] = data.with_answers(decided, readings, {where: ["Pronađene"]})
    assert too_few is None


def test_same_takes_superscripts_as_carets():
    assert data.same("10⁹/L", "10^9/L") and data.same("10¹²/L", " 10^12/L")
    assert not data.same("< 35", "> 35")
    assert not data.same("10⁹/L", "109/L") and not data.same("10¹²/L", "1012/L")


def test_html_rows():
    rows = data.html_rows("<table><tr><th>a</th><th>b</th></tr><tr><td>1<br>2</td><td> x </td></tr></table>")
    assert rows == [["a", "b"], ["1 2", "x"]]
