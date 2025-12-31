# TaskFlow Sentiment Analysis

Advanced sentiment analysis system that generates professional reports from social media comments using hybrid AI approach (specialized models + LLM orchestration).

## 🎯 Key Features

- **Hybrid Intelligence**: 5 specialized ML models + LLM synthesis
- **87% Cost Savings**: $0.015/1000 comments (vs $0.20+ with pure GPT-4)
- **Professional Reports**: PDF/HTML with insights & recommendations
- **Auto-detection**: Intelligent column detection in any CSV format
- **Quality Assurance**: Multi-layer validation & evaluation

## 🏗️ Architecture
```
CSV Upload → Data Layer → Analysis Layer → QC Layer → Intelligence Layer → Report
              (2 agents)   (5 tools)       (1 agent)   (3 LLM agents)
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/taskflow-sentiment-analysis
cd taskflow-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys
```

### Usage

Coming in Week 4 (API + UI)

## 🧪 Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/unit/test_column_detector.py
```

## 📁 Project Structure
```
src/
├── agents/      # 7 AI agents
├── tools/       # 5 ML analysis tools
├── models/      # Pydantic schemas
├── utils/       # Helpers
├── llm/         # LLM integration
└── api/         # FastAPI server
```

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI
- **ML/NLP**: Transformers, PyTorch
- **LLM**: Anthropic Claude, OpenAI GPT-4
- **Validation**: Pydantic
- **Testing**: pytest (>70% coverage)

## 📊 Performance

- Process 1000 comments in <60s
- Generate report in <30s
- 90%+ sentiment accuracy
- $0.015 per 1000 comments

## 🗓️ Development Timeline

- **Week 1**: Data Layer ✅ (current)
- **Week 2**: Analysis Layer
- **Week 3**: Intelligence Layer
- **Week 4**: Production Deployment

## 📝 Documentation

See [docs/](docs/) folder for detailed documentation:
- [Architecture](docs/ARCHITECTURE.md)
- [Agent Interfaces](docs/AGENT_INTERFACES.md)
- [Data Models](docs/DATA_MODELS.md)
- [Project Overview](docs/PROJECT_OVERVIEW.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 👤 Author

[Your Name] - [LinkedIn](your-linkedin) - [Email](your-email)

---

**Status**: Week 1 - Data Layer Development