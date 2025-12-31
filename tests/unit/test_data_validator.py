"""Test DataValidatorAgent."""

import pandas as pd
import pytest

from src.agents.data_validator import DataValidatorAgent
from src.models.schemas import DataValidatorInput
from src.utils.exceptions import ValidationError


@pytest.fixture
def agent():
    """Create DataValidatorAgent instance."""
    return DataValidatorAgent(min_rows=10, max_rows=10000)


@pytest.fixture
def clean_data():
    """Clean DataFrame without issues."""
    return pd.DataFrame(
        {
            "id": range(1, 51),
            "comment": [
                f"This is a good quality comment number {i} with enough text." for i in range(1, 51)
            ],
            "rating": [5] * 50,
        }
    )


@pytest.fixture
def messy_data():
    """Messy DataFrame with various issues."""
    comments = [
        "Good comment with enough text here",  # OK
        "Another valid comment with content",  # OK
        None,  # Missing
        "",  # Empty
        "   ",  # Whitespace only
        "ab",  # Too short
        "Good comment with enough text here",  # Duplicate
        "Check this link http://example.com out",  # URL
        "<p>HTML tag here</p> with content",  # HTML
        "  Multiple   spaces   here  ",  # Excessive whitespace
    ] * 5  # 50 rows total

    return pd.DataFrame({"id": range(1, 51), "comment": comments, "rating": [3] * 50})


def test_agent_initialization(agent):
    """Test agent can be initialized."""
    assert agent is not None
    assert agent.config.name == "DataValidatorAgent"
    assert agent.min_rows == 10
    assert agent.max_rows == 10000


def test_validate_clean_data(agent, clean_data):
    """Test validation of clean data."""
    input_data = DataValidatorInput(dataframe=clean_data, text_column="comment")

    output = agent.execute(input_data)

    assert output.status == "pass"
    assert output.stats.total_rows == 50
    assert output.stats.rows_after_cleaning == 50
    assert output.stats.removed_rows == 0
    assert len(output.issues) == 0


def test_clean_messy_data(agent, messy_data):
    """Test cleaning of messy data."""
    input_data = DataValidatorInput(dataframe=messy_data, text_column="comment")

    output = agent.execute(input_data)

    # Should remove: missing, empty, whitespace, short, duplicates
    assert output.stats.removed_rows > 0
    assert output.stats.rows_after_cleaning < 50

    # Check that cleaned data has no issues
    assert all(len(str(text).strip()) >= 3 for text in output.cleaned_data["comment"])


def test_remove_html_tags(agent):
    """Test HTML tag removal."""
    df = pd.DataFrame(
        {
            "comment": [
                "<p>Text with HTML</p>",
                "<div>More <strong>HTML</strong></div>",
                "Normal text",
            ]
            * 5  # 15 rows
        }
    )

    input_data = DataValidatorInput(dataframe=df, text_column="comment")
    output = agent.execute(input_data)

    # Check HTML tags removed
    for text in output.cleaned_data["comment"]:
        assert "<" not in text
        assert ">" not in text


def test_remove_urls(agent):
    """Test URL removal."""
    df = pd.DataFrame(
        {
            "comment": [
                "Check http://example.com for more info",
                "Visit www.example.com today",
                "No URL here",
            ]
            * 5  # 15 rows
        }
    )

    input_data = DataValidatorInput(dataframe=df, text_column="comment")
    output = agent.execute(input_data)

    # Check URLs removed
    for text in output.cleaned_data["comment"]:
        assert "http" not in text
        assert "www." not in text


def test_remove_duplicates(agent):
    """Test duplicate removal."""
    df = pd.DataFrame(
        {
            "comment": [
                "Same comment repeated",
                "Same comment repeated",
                "Same comment repeated",
                "Unique comment here",
                "Another unique one",
            ]
            * 5  # 25 rows
        }
    )

    input_data = DataValidatorInput(dataframe=df, text_column="comment")
    output = agent.execute(input_data)

    assert output.stats.duplicates > 0
    # Should only have unique comments
    assert len(output.cleaned_data["comment"].unique()) == len(output.cleaned_data)


def test_column_not_found(agent, clean_data):
    """Test error when column doesn't exist."""
    input_data = DataValidatorInput(dataframe=clean_data, text_column="nonexistent")

    with pytest.raises(ValidationError, match="Column 'nonexistent' not found"):
        agent.execute(input_data)


def test_too_few_rows(agent):
    """Test error when too few rows."""
    df = pd.DataFrame({"comment": ["Short comment"] * 5})  # Only 5 rows

    input_data = DataValidatorInput(dataframe=df, text_column="comment")

    with pytest.raises(ValidationError, match="only 5 rows"):
        agent.execute(input_data)


def test_too_many_rows(agent):
    """Test error when too many rows."""
    df = pd.DataFrame({"comment": ["Comment"] * 15000})  # Too many

    input_data = DataValidatorInput(dataframe=df, text_column="comment")

    with pytest.raises(ValidationError, match="maximum"):
        agent.execute(input_data)


def test_stats_calculation(agent, messy_data):
    """Test that statistics are calculated correctly."""
    input_data = DataValidatorInput(dataframe=messy_data, text_column="comment")

    output = agent.execute(input_data)
    stats = output.stats

    assert stats.total_rows == 50
    assert stats.rows_after_cleaning > 0
    assert stats.removed_rows == stats.total_rows - stats.rows_after_cleaning
    assert stats.avg_text_length > 0
    assert stats.min_text_length >= 3  # Minimum threshold
    assert stats.max_text_length >= stats.min_text_length


def test_warning_status(agent):
    """Test warning status when many rows removed."""
    df = pd.DataFrame(
        {
            "comment": [f"Good comment with text number {i}" for i in range(20)]
            + [""] * 35  # 63% empty
        }
    )

    input_data = DataValidatorInput(dataframe=df, text_column="comment")
    output = agent.execute(input_data)

    # Should warn about high removal rate (but still have enough rows)
    assert output.status == "warning"
    assert len(output.warnings) > 0
    assert output.stats.rows_after_cleaning >= 10  # Still above minimum
    assert output.stats.removed_rows > output.stats.total_rows * 0.5  # >50% removed


def test_excessive_whitespace_removal(agent):
    """Test that excessive whitespace is normalized."""
    df = pd.DataFrame(
        {
            "comment": ["Multiple   spaces   here", "  Leading and trailing  ", "Tab\t\tcharacters"]
            * 5
        }
    )

    input_data = DataValidatorInput(dataframe=df, text_column="comment")
    output = agent.execute(input_data)

    # Check single spaces
    for text in output.cleaned_data["comment"]:
        assert "  " not in text  # No double spaces
        assert text == text.strip()  # No leading/trailing


def test_spam_detection(agent):
    """Test spam comment removal."""
    df = pd.DataFrame(
        {
            "comment": [
                "Great product, very satisfied!",
                "BUY NOW!!! CLICK HERE!!!",
                "Visit http://spam.com now for FREE prizes!!!",
                "Decent quality for the price",
                "WINNER WINNER GET YOUR PRIZE NOW",
                "$$$ Make money fast $$$",
                "Normal comment here",
            ]
            * 3  # 21 rows
        }
    )

    input_data = DataValidatorInput(dataframe=df, text_column="comment")
    output = agent.execute(input_data)

    # Check spam removed
    assert output.stats.rows_after_cleaning < 21

    # Check no spam in cleaned data
    for text in output.cleaned_data["comment"]:
        assert not agent._is_spam(text)


def test_excessive_caps_as_spam(agent):
    """Test that excessive caps is detected as spam."""
    df = pd.DataFrame(
        {
            "comment": [
                "THIS IS ALL CAPS SPAM MESSAGE",
                "Normal comment with some CAPS",
                "ANOTHER SPAM WITH ALL CAPS HERE",
            ]
            * 5
        }
    )

    input_data = DataValidatorInput(dataframe=df, text_column="comment")
    output = agent.execute(input_data)

    # Should remove excessive caps
    cleaned_texts = output.cleaned_data["comment"].tolist()
    all_caps_count = sum(1 for t in cleaned_texts if t.isupper() and len(t) > 10)
    assert all_caps_count == 0
