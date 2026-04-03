from stella_ml.harness import OpenClawStyleHarness
from stella_ml.unstructured import autoeda_text, ngrams, parse_html_text, parse_xml_text


def test_parse_html_text_removes_tags() -> None:
    text = parse_html_text("<html><body><h1>Title</h1><p>Hello world</p></body></html>")
    assert "Title" in text
    assert "Hello world" in text


def test_parse_xml_text_extracts_text() -> None:
    text = parse_xml_text("<root><item>A</item><item>B</item></root>")
    assert text == "A B"


def test_autoeda_and_ngrams() -> None:
    report = autoeda_text("hello world hello")
    assert report.num_tokens == 3
    assert ngrams("hello world hello", n=2)


def test_harness_run_unstructured_data_flow_html() -> None:
    harness = OpenClawStyleHarness()
    output = harness.run_unstructured_data_flow("<html><body>alpha beta alpha</body></html>", source_type="html")
    assert output["text_eda"].num_tokens >= 3
    assert output["ngrams"]
