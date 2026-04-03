from stella_ml.providers.huggingface import HuggingFaceProvider


def test_hf_extract_generated_text_list() -> None:
    body = [{"generated_text": "hello"}]
    assert HuggingFaceProvider._extract_content(body) == "hello"


def test_hf_extract_error() -> None:
    body = {"error": "model loading"}
    assert "HF_ERROR" in HuggingFaceProvider._extract_content(body)
