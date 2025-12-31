"""Test EntityTool."""

import pytest

from src.tools.entity_tool import EntityTool


@pytest.fixture
def entity_tool():
    """Create EntityTool instance."""
    return EntityTool(model_name="dslim/bert-base-NER", device="cpu")


def test_tool_initialization(entity_tool):
    """Test tool can be initialized."""
    assert entity_tool is not None
    assert entity_tool.model is not None


def test_person_entity(entity_tool):
    """Test person entity detection."""
    text = "Steve Jobs founded Apple in California."
    result = entity_tool.analyze(text)

    entities = result["entities"]
    assert len(entities) > 0

    # Check if Steve Jobs detected
    entity_texts = [e["text"] for e in entities]
    assert any("Steve" in text or "Jobs" in text for text in entity_texts)


def test_organization_entity(entity_tool):
    """Test organization entity detection."""
    text = "I work at Microsoft and use their products daily."
    result = entity_tool.analyze(text)

    entities = result["entities"]
    entity_texts = [e["text"] for e in entities]

    assert any("Microsoft" in text for text in entity_texts)


def test_location_entity(entity_tool):
    """Test location entity detection."""
    text = "We traveled to Paris and London last summer."
    result = entity_tool.analyze(text)

    entities = result["entities"]
    entity_texts = [e["text"] for e in entities]

    assert any("Paris" in text or "London" in text for text in entity_texts)


def test_no_entities(entity_tool):
    """Test text with no entities."""
    text = "This is a simple sentence without any named entities."
    result = entity_tool.analyze(text)

    # Might have 0 entities or very low confidence ones
    assert "entities" in result
    assert isinstance(result["entities"], list)


def test_batch_analysis(entity_tool):
    """Test batch processing."""
    texts = [
        "Apple Inc. is based in California.",
        "Barack Obama was president.",
        "No entities here.",
    ]

    results = entity_tool.analyze_batch(texts)

    assert len(results) == 3
    assert all("entities" in r for r in results)
