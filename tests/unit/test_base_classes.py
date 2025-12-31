"""Test base classes."""

from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.models.schemas import AgentConfig
from src.tools.base import BaseTool


class DummyInput(BaseModel):
    """Dummy input for testing."""

    text: str


class DummyOutput(BaseModel):
    """Dummy output for testing."""

    result: str


class DummyAgent(BaseAgent):
    """Dummy agent for testing."""

    def execute(self, input_data: DummyInput) -> DummyOutput:
        """Execute dummy logic."""
        return DummyOutput(result=f"processed: {input_data.text}")


class DummyTool(BaseTool):
    """Dummy tool for testing."""

    def _load_model(self):
        """Load dummy model."""
        return "dummy_model"

    def analyze(self, text: str) -> dict:
        """Analyze text."""
        return {"text": text, "result": "analyzed"}


def test_base_agent_initialization():
    """Test BaseAgent can be initialized."""
    config = AgentConfig(name="TestAgent")
    agent = DummyAgent(config)

    assert agent.config.name == "TestAgent"
    assert agent.logger is not None


def test_base_agent_execute():
    """Test BaseAgent execute method."""
    config = AgentConfig(name="TestAgent")
    agent = DummyAgent(config)

    input_data = DummyInput(text="hello")
    output = agent.execute(input_data)

    assert output.result == "processed: hello"


def test_base_tool_initialization():
    """Test BaseTool can be initialized."""
    tool = DummyTool(model_name="test-model", device="cpu")

    assert tool.model_name == "test-model"
    assert tool.device == "cpu"
    assert tool.model == "dummy_model"


def test_base_tool_analyze():
    """Test BaseTool analyze method."""
    tool = DummyTool(model_name="test-model")

    result = tool.analyze("test text")

    assert result["text"] == "test text"
    assert result["result"] == "analyzed"


def test_base_tool_analyze_batch():
    """Test BaseTool analyze_batch method."""
    tool = DummyTool(model_name="test-model")

    texts = ["text1", "text2", "text3"]
    results = tool.analyze_batch(texts)

    assert len(results) == 3
    assert results[0]["text"] == "text1"
    assert results[1]["text"] == "text2"
