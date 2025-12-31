"""Test KeyphraseTool."""

import pytest

from src.tools.keyphrase_tool import KeyphraseTool


@pytest.fixture
def keyphrase_tool():
    """Create KeyphraseTool instance."""
    return KeyphraseTool(model_name="ml6team/keyphrase-extraction-distilbert-inspec", device="cpu")


def test_tool_initialization(keyphrase_tool):
    """Test tool can be initialized."""
    assert keyphrase_tool is not None
    assert keyphrase_tool.model is not None


def test_keyphrase_extraction(keyphrase_tool):
    """Test keyphrase extraction."""
    text = "Machine learning and artificial intelligence are transforming technology."
    result = keyphrase_tool.analyze(text)

    assert "keyphrases" in result
    assert "scores" in result
    assert len(result["keyphrases"]) > 0


def test_technical_text(keyphrase_tool):
    """Test with technical text."""
    text = "Deep learning neural networks use backpropagation for training."
    result = keyphrase_tool.analyze(text)

    keyphrases = result["keyphrases"]
    assert len(keyphrases) > 0

    # Should extract technical terms
    all_phrases = " ".join(keyphrases).lower()
    assert any(term in all_phrases for term in ["learning", "neural", "network"])


def test_product_review(keyphrase_tool):
    """Test with product review."""
    text = "Great battery life and excellent camera quality. Fast performance."
    result = keyphrase_tool.analyze(text)

    keyphrases = result["keyphrases"]
    assert len(keyphrases) > 0


def test_scores_present(keyphrase_tool):
    """Test that scores are returned for keyphrases."""
    text = "Cloud computing enables scalable infrastructure deployment."
    result = keyphrase_tool.analyze(text)

    keyphrases = result["keyphrases"]
    scores = result["scores"]

    # All keyphrases should have scores
    for phrase in keyphrases:
        assert phrase in scores
        assert 0 <= scores[phrase] <= 1


def test_batch_analysis(keyphrase_tool):
    """Test batch processing."""
    texts = [
        "Artificial intelligence and machine learning.",
        "Software development and programming.",
        "Data science and analytics.",
    ]

    results = keyphrase_tool.analyze_batch(texts)

    assert len(results) == 3
    assert all("keyphrases" in r for r in results)
    assert all("scores" in r for r in results)
