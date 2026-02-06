# TaskFlow Developer Guide

> **Complete guide for developers contributing to TaskFlow**

This guide covers everything you need to set up, develop, test, and contribute to TaskFlow.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Architecture Overview](#architecture-overview)
4. [Development Workflow](#development-workflow)
5. [Code Standards](#code-standards)
6. [Testing](#testing)
7. [Adding Features](#adding-features)
8. [Contributing](#contributing)

---

## Development Setup

### Prerequisites

**Required:**
- Python 3.10 or higher
- pip (Python package manager)
- Git

**Optional:**
- Docker & Docker Compose
- Virtual environment manager (venv, conda)

**Check Versions:**

```bash
python --version  # Should be 3.10+
pip --version
git --version
```

---

### Local Setup

**1. Clone Repository**

```bash
git clone https://github.com/yourusername/taskflow-sentiment-analysis.git
cd taskflow-sentiment-analysis
```

**2. Create Virtual Environment**

```bash
# Using venv
python -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

**3. Install Dependencies**

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies (includes testing, linting)
pip install -r requirements-dev.txt
```

**4. Configure Environment**

```bash
# Copy example env file
cp .env.example .env

# Edit .env file
nano .env
```

**Required Environment Variables:**

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here  # Optional, for GPT-4

# Optional
HF_TOKEN=hf_your_token_here  # For private HuggingFace models
LOG_LEVEL=INFO
MAX_WORKERS=4
```

**5. Verify Setup**

```bash
# Run tests
pytest tests/

# Start server
uvicorn src.api.main:app --reload --port 8000

# Visit http://localhost:8000
```

---

### Development Tools

**Code Quality:**

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Linter
pip install ruff

# Formatter
pip install black

# Type checker
pip install mypy
```

**IDE Setup:**

**VS Code (Recommended):**

```json
// .vscode/settings.json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true
}
```

**PyCharm:**
- Enable Black formatter: Preferences → Tools → Black
- Enable pytest: Preferences → Tools → Python Integrated Tools → Testing → pytest

---

## Project Structure

```
taskflow-sentiment-analysis/
├── src/                          # Source code
│   ├── agents/                   # 7 AI agents
│   │   ├── base.py              # BaseAgent class
│   │   ├── column_detector.py  # Auto-detect text column
│   │   ├── data_validator.py   # Clean & validate data
│   │   ├── orchestrator.py     # Coordinate tools
│   │   ├── pre_evaluator.py    # Quality check (pre-LLM)
│   │   ├── report_planner.py   # Plan report structure (LLM)
│   │   ├── report_writer.py    # Generate report text (LLM)
│   │   ├── report_evaluator.py # Validate report quality (LLM)
│   │   └── report_generator.py # Orchestrate report generation
│   │
│   ├── tools/                    # 5 ML analysis tools
│   │   ├── base.py              # BaseTool class
│   │   ├── sentiment_tool.py   # Sentiment analysis
│   │   ├── emotion_tool.py     # Emotion detection
│   │   ├── topic_tool.py       # Topic extraction
│   │   ├── entity_tool.py      # Named entity recognition
│   │   └── keyphrase_tool.py   # Keyphrase extraction
│   │
│   ├── llm/                      # LLM integration
│   │   ├── base.py              # BaseLLM interface
│   │   ├── anthropic_llm.py    # Claude integration
│   │   ├── openai_llm.py       # GPT-4 integration (optional)
│   │   └── mock_llm.py         # Mock for testing
│   │
│   ├── api/                      # FastAPI server
│   │   ├── main.py              # App initialization
│   │   ├── routes.py            # API endpoints
│   │   ├── models.py            # Pydantic request/response models
│   │   ├── storage.py           # In-memory job storage
│   │   └── jobs.py              # Background job processor
│   │
│   ├── export/                   # Report export
│   │   └── pdf_generator.py    # Markdown → PDF conversion
│   │
│   ├── models/                   # Data models
│   │   └── schemas.py           # Pydantic schemas for agents
│   │
│   └── utils/                    # Utilities
│       ├── logger.py            # Logging setup
│       └── exceptions.py        # Custom exceptions
│
├── static/                       # Web UI
│   ├── css/
│   │   └── style.css            # Styling
│   ├── js/
│   │   └── app.js               # Frontend logic
│   └── index.html               # Main page
│
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   │   ├── test_agents/
│   │   ├── test_tools/
│   │   └── test_llm/
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test data & fixtures
│
├── docs/                         # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── USER_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── TROUBLESHOOTING.md
│
├── scripts/                      # Utility scripts
│   ├── test_e2e.py              # End-to-end testing
│   └── test_real_api.py         # Real API testing
│
├── configs/                      # Configuration files
│   └── models.yaml              # Model configurations
│
├── .env.example                  # Environment template
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── Dockerfile                   # Docker image
├── docker-compose.yml           # Docker Compose config
├── pytest.ini                   # Pytest configuration
└── README.md                    # Main documentation
```

---

## Architecture Overview

### Design Patterns

**1. Agent Pattern**

All agents inherit from `BaseAgent`:

```python
class BaseAgent(ABC):
    @abstractmethod
    def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute agent logic."""
        pass
```

**Benefits:**
- Uniform interface
- Easy testing (mock agents)
- Clear contracts

**2. Strategy Pattern (LLM)**

```python
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> Dict:
        pass

# Implementations: ClaudeLLM, OpenAILLM, MockLLM
```

**3. Factory Pattern (Tools)**

```python
class ToolFactory:
    @staticmethod
    def create_tool(tool_name: str, config: ToolConfig):
        return TOOLS[tool_name](config)
```

---

### Data Flow

```
CSV Upload
    ↓
ColumnDetectorAgent → Detect text column
    ↓
DataValidatorAgent → Clean & validate
    ↓
AnalysisOrchestratorAgent → Run 5 tools in parallel
    ├─ SentimentTool
    ├─ EmotionTool
    ├─ TopicTool
    ├─ EntityTool
    └─ KeyphraseTool
    ↓
PrePromptEvaluatorAgent → Quality check
    ↓
ReportGenerator → Orchestrate report generation
    ├─ ReportPlannerAgent (LLM)
    ├─ ReportWriterAgent (LLM)
    └─ ReportEvaluatorAgent (LLM)
    ↓
PDF/Markdown Export
```

---

### Agent Lifecycle

```python
# 1. Initialize
agent = SomeAgent(config)

# 2. Validate input
input_data = SomeInput(**data)

# 3. Execute
output = agent.execute(input_data)

# 4. Log
agent.log_execution(input_data, output)

# 5. Return
return output
```

---

## Development Workflow

### Creating a New Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

```bash
# 1. Write code
# 2. Write tests
# 3. Run tests
pytest tests/

# 4. Format code
black src/

# 5. Check linting
ruff check src/

# 6. Type check (optional)
mypy src/
```

### Committing Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add new feature

- Implement feature X
- Add tests for feature X
- Update documentation"

# Push to remote
git push origin feature/your-feature-name
```

### Creating Pull Request

1. Push branch to GitHub
2. Open Pull Request
3. Describe changes
4. Link related issues
5. Request review
6. Address feedback
7. Merge when approved

---

## Code Standards

### Python Style Guide

**Follow PEP 8 with these specifics:**

- **Line Length:** 100 characters
- **Indentation:** 4 spaces
- **Quotes:** Double quotes for strings
- **Imports:** Organized (stdlib, third-party, local)

**Example:**

```python
"""Module docstring.

Detailed description of what this module does.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel

from src.utils.logger import setup_logger


class MyClass:
    """Class docstring.
    
    Detailed description of what this class does.
    """
    
    def __init__(self, config: Dict):
        """Initialize MyClass.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
    
    def process(self, data: List[str]) -> Optional[Dict]:
        """Process data.
        
        Args:
            data: List of strings to process
            
        Returns:
            Processed data dictionary or None if failed
            
        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        result = self._internal_process(data)
        return result
```

---

### Docstrings

**Use Google style:**

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description.
    
    Longer description if needed. Can span multiple lines
    and include details about the function's behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string
        
    Example:
        >>> function_name("test", 5)
        True
    """
    pass
```

---

### Type Hints

**Always use type hints:**

```python
# Good
def process_data(data: List[str], threshold: float = 0.5) -> Dict[str, int]:
    pass

# Bad
def process_data(data, threshold=0.5):
    pass
```

---

### Error Handling

```python
from src.utils.exceptions import TaskFlowError

class SomeAgent(BaseAgent):
    def execute(self, input_data: SomeInput) -> SomeOutput:
        try:
            # Validate
            if not self.validate_input(input_data):
                raise TaskFlowError("Invalid input")
            
            # Process
            result = self._process(input_data)
            
            # Log
            self.log_execution(input_data, result)
            
            return result
            
        except TaskFlowError:
            # Re-raise TaskFlow errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            self.logger.error(f"Unexpected error: {e}")
            raise TaskFlowError(f"Agent execution failed: {e}") from e
```

---

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| **Module** | lowercase_underscore | `sentiment_tool.py` |
| **Class** | PascalCase | `SentimentTool` |
| **Function** | lowercase_underscore | `process_data()` |
| **Variable** | lowercase_underscore | `input_data` |
| **Constant** | UPPERCASE | `MAX_RETRIES` |
| **Private** | _leading_underscore | `_internal_method()` |

---

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests (isolated)
│   ├── test_agents/
│   │   ├── test_column_detector.py
│   │   └── test_data_validator.py
│   ├── test_tools/
│   │   ├── test_sentiment_tool.py
│   │   └── test_emotion_tool.py
│   └── test_llm/
│       └── test_mock_llm.py
│
├── integration/             # Integration tests
│   ├── test_pipeline.py
│   └── test_report_generation.py
│
└── fixtures/                # Test data
    ├── sample_data.csv
    └── test_config.yaml
```

---

### Writing Unit Tests

```python
"""Unit tests for SentimentTool."""

import pytest

from src.tools.sentiment_tool import SentimentTool


class TestSentimentTool:
    """Test suite for SentimentTool."""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return SentimentTool(device="cpu")
    
    def test_analyze_positive(self, tool):
        """Test positive sentiment detection."""
        result = tool.analyze("This is great!")
        
        assert result["label"] == "positive"
        assert result["score"] > 0.5
    
    def test_analyze_negative(self, tool):
        """Test negative sentiment detection."""
        result = tool.analyze("This is terrible!")
        
        assert result["label"] == "negative"
        assert result["score"] > 0.5
    
    def test_analyze_batch(self, tool):
        """Test batch processing."""
        texts = ["Good", "Bad", "Okay"]
        results = tool.analyze_batch(texts)
        
        assert len(results) == 3
        assert all("label" in r for r in results)
    
    def test_empty_input(self, tool):
        """Test empty input handling."""
        with pytest.raises(ValueError):
            tool.analyze("")
```

---

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_sentiment_tool.py

# Run specific test
pytest tests/unit/test_sentiment_tool.py::TestSentimentTool::test_analyze_positive

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/

# Run only failed tests
pytest --lf

# Run in parallel (faster)
pytest -n auto tests/
```

---

### Test Coverage

**Target:** >70% coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# View report
open htmlcov/index.html
```

---

### Mocking

```python
from unittest.mock import Mock, patch

def test_with_mock_llm():
    """Test with mocked LLM."""
    mock_llm = Mock()
    mock_llm.generate.return_value = {
        "content": "Test report",
        "tokens": 100
    }
    
    agent = ReportWriterAgent(llm=mock_llm)
    result = agent.execute(test_input)
    
    assert result.report_text == "Test report"
    mock_llm.generate.assert_called_once()
```

---

## Adding Features

### Adding a New Agent

**1. Create Agent Class**

```python
# src/agents/my_new_agent.py

from src.agents.base import BaseAgent
from src.models.schemas import MyInput, MyOutput

class MyNewAgent(BaseAgent):
    """New agent description."""
    
    def execute(self, input_data: MyInput) -> MyOutput:
        """Execute agent logic.
        
        Args:
            input_data: Input data
            
        Returns:
            Output data
        """
        # Implementation
        result = self._process(input_data)
        return MyOutput(**result)
```

**2. Define Schemas**

```python
# src/models/schemas.py

class MyInput(BaseModel):
    """Input schema for MyNewAgent."""
    field1: str
    field2: int

class MyOutput(BaseModel):
    """Output schema for MyNewAgent."""
    result: str
    score: float
```

**3. Write Tests**

```python
# tests/unit/test_my_new_agent.py

class TestMyNewAgent:
    def test_execute_success(self):
        agent = MyNewAgent()
        input_data = MyInput(field1="test", field2=5)
        result = agent.execute(input_data)
        assert isinstance(result, MyOutput)
```

**4. Update Documentation**

- Add to `docs/AGENT_INTERFACES.md`
- Update architecture diagram
- Add usage example

---

### Adding a New Tool

**1. Create Tool Class**

```python
# src/tools/my_new_tool.py

from src.tools.base import BaseTool
from transformers import pipeline

class MyNewTool(BaseTool):
    """New tool description."""
    
    def _load_model(self):
        """Load model."""
        self.pipeline = pipeline(
            "task-name",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1
        )
    
    def analyze(self, text: str) -> Dict:
        """Analyze text.
        
        Args:
            text: Input text
            
        Returns:
            Analysis results
        """
        result = self.pipeline(text)
        return {
            "label": result[0]["label"],
            "score": result[0]["score"]
        }
```

**2. Add to Orchestrator**

```python
# src/agents/orchestrator.py

class AnalysisOrchestratorAgent:
    def __init__(self, config: ToolConfig, device: str = "cpu"):
        # ... existing tools ...
        self.my_new_tool = MyNewTool(
            model_name=config.my_new_model,
            device=device
        )
```

**3. Update Config**

```python
# src/models/schemas.py

class ToolConfig(BaseModel):
    # ... existing configs ...
    my_new_model: str = "default/model-name"
```

---

### Adding a New Endpoint

**1. Define Models**

```python
# src/api/models.py

class MyRequest(BaseModel):
    """Request model."""
    param1: str
    param2: int

class MyResponse(BaseModel):
    """Response model."""
    result: str
    status: str
```

**2. Add Route**

```python
# src/api/routes.py

@router.post("/my-endpoint", response_model=MyResponse)
async def my_endpoint(request: MyRequest):
    """New endpoint description.
    
    Args:
        request: Request data
        
    Returns:
        Response data
    """
    # Implementation
    result = process_request(request)
    return MyResponse(result=result, status="success")
```

**3. Update API Docs**

- Add to `docs/API_DOCUMENTATION.md`
- Include example request/response
- Document error cases

---

## Contributing

### Contribution Workflow

1. **Fork** repository
2. **Clone** your fork
3. **Create** feature branch
4. **Make** changes
5. **Test** thoroughly
6. **Commit** with clear messages
7. **Push** to your fork
8. **Create** Pull Request

---

### Pull Request Guidelines

**Title:**
```
feat: Add sentiment analysis caching
fix: Correct column detection for edge case
docs: Update API documentation
test: Add integration tests for pipeline
```

**Description Template:**

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass locally
```

---

### Code Review Process

**As Author:**
1. Ensure all tests pass
2. Address linting issues
3. Update documentation
4. Respond to feedback promptly
5. Make requested changes

**As Reviewer:**
1. Check code quality
2. Verify tests exist
3. Test functionality
4. Review documentation
5. Provide constructive feedback

---

### Commit Message Convention

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Example:**

```
feat(tools): Add caching for sentiment analysis

- Implement LRU cache for sentiment results
- Add cache configuration options
- Update tests to cover caching behavior

Closes #123
```

---

## Development Tips

### Debugging

**1. Use Logger**

```python
self.logger.debug(f"Processing {len(data)} items")
self.logger.info("Agent execution started")
self.logger.warning("Low confidence score detected")
self.logger.error("Failed to process data", exc_info=True)
```

**2. Interactive Debugging**

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in
breakpoint()
```

**3. Print Debugging**

```python
from pprint import pprint
pprint(complex_data_structure)
```

---

### Performance Profiling

```python
import cProfile
import pstats

# Profile code
profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

---

### Hot Reload

```bash
# Server auto-reloads on file changes
uvicorn src.api.main:app --reload --port 8000
```

---

## Resources

**Documentation:**
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Pytest Docs](https://docs.pytest.org/)

**Style Guides:**
- [PEP 8](https://pep8.org/)
- [Google Python Style](https://google.github.io/styleguide/pyguide.html)

**Tools:**
- [Black Formatter](https://black.readthedocs.io/)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [MyPy Type Checker](https://mypy.readthedocs.io/)

---

## Getting Help

**Questions?**

💬 **Discussions:** [GitHub Discussions](https://github.com/yourusername/taskflow/discussions)  
🐛 **Issues:** [GitHub Issues](https://github.com/yourusername/taskflow/issues)  
📧 **Email:** dev@taskflow.example.com

---

**Happy coding! 🚀**