# MARIE Agent System 🤖

**Multi-Agent Research Intelligence Engine** for ImpactU CRIS

Evidence-based scientometric analysis powered by LangGraph multi-agent orchestration.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.5-green.svg)](https://langchain-ai.github.io/langgraph/)
[![vLLM](https://img.shields.io/badge/vLLM-0.6+-orange.svg)](https://vllm.ai/)

## 🎯 Features

- 🤝 **Multi-Agent Architecture**: Specialized agents for entity resolution, retrieval, validation, metrics, citations, and reporting
- 🏛️ **Hexagonal Architecture**: Clean separation with Ports & Adapters for maximum flexibility
- 🚀 **Local LLM Support**: vLLM integration optimized for 4GB VRAM GPUs
- 🔍 **Hybrid Search**: MongoDB (structured) + OpenSearch (RAG) for comprehensive evidence retrieval
- 📊 **Confidence Scoring**: Every answer includes confidence level and reasoning
- 🔗 **Evidence-Based**: All claims mapped to explicit evidence sources
- 📝 **Complete Audit Trail**: Full tracking of agent decisions and data sources

## 🏗️ Architecture

MARIE implements **Magentic principles** + **Hexagonal Architecture**:

```
┌──────────────────────────────────────────────────────────┐
│                   Presentation Layer                      │
│               CLI, API, Web Interface                     │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│              LangGraph Orchestrator                       │
│         (Plan → Route → Execute → Evaluate)              │
└──┬────────┬────────┬────────┬────────┬────────┬─────────┘
   │        │        │        │        │        │
   ▼        ▼        ▼        ▼        ▼        ▼
┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐
│ER  │  │Ret │  │Val │  │Met │  │Cit │  │Rep │  Agents
└─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘
  │       │       │       │       │       │
  └───────┴───────┴───────┴───────┴───────┘
                    │
          ┌─────────▼──────────┐
          │  Ports (Interfaces) │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │  Adapters (Impl)    │
          │  - vLLM            │
          │  - MongoDB         │
          │  - OpenSearch      │
          └────────────────────┘
```

### Specialized Agents

1. **Entity Resolution**: Disambiguates institutions, authors, research groups
2. **Retrieval**: Hybrid search across MongoDB and OpenSearch
3. **Validation**: Ensures data consistency and quality
4. **Metrics**: Computes scientometric indicators
5. **Citations**: Maps claims to evidence
6. **Reporting**: Generates structured, evidence-backed answers

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- MongoDB running on `localhost:27017`
- OpenSearch running on `localhost:9200`
- (Optional) NVIDIA GPU with 4GB+ VRAM for local LLM

### 1. Installation

```bash
git clone https://github.com/colav-playground/impactu_marie_agent.git
cd impactu_marie_agent

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### 2. Install Ollama & Download Models (Recommended)

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Download recommended models
ollama pull qwen2:1.5b      # General queries (934 MB)
ollama pull phi3:mini        # Advanced reasoning (2.2 GB)
ollama pull llama3.2:1b      # Fast inference (1.3 GB)
```

**Model Selection Guide:**
- **qwen2:1.5b** ⭐ Best balance quality/size - Default
- **phi3:mini** - Strong reasoning for complex queries
- **llama3.2:1b** - Ultra-fast for simple queries

### 3. Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Use Ollama (recommended)
MARIE_LLM_PROVIDER=ollama
MARIE_LLM_MODEL=qwen2:1.5b

# Or use vLLM (requires more setup)
# MARIE_LLM_PROVIDER=vllm
# MARIE_LLM_MODEL=Qwen/Qwen2-1.5B-Instruct

# Or use Anthropic API
# MARIE_LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your_key_here
```

### 4. Run MARIE

```bash
python -m marie_agent.cli "What are the top 5 most cited papers from Universidad de Antioquia?"
```

## 📖 Documentation

- [Hexagonal Architecture](docs/hexagonal_architecture.md) - Ports & Adapters design
- [Magentic Multi-Agent](docs/cris_magentic_multiagent_architecture.md) - Agent orchestration
- [RAG Implementation](docs/kahi_rag_implementation.md) - Retrieval system

## 💻 Usage Examples

### CLI

```bash
# Simple query
python -m marie_agent.cli "Top cited papers from UdeA"

# Complex query with filters
python -m marie_agent.cli "Author productivity in machine learning from 2020-2024"

# With custom request ID
python -m marie_agent.cli "Collaboration network analysis" --request-id custom-123
```

### Python API

```python
from marie_agent.graph import run_marie_query

result = run_marie_query(
    query="What are the top papers about AI from UdeA?",
    request_id="req_001"
)

print(result["final_answer"])
print(f"Confidence: {result['confidence_score']:.2%}")
```

### Example Output

```
================================================================================
MARIE Multi-Agent System
================================================================================
Query: What are the top 5 most cited papers from Universidad de Antioquia?

## Top 5 Most Cited Papers

**1. Paper Title Here**
   - Year: 2020
   - Citations: 245
   - DOI: https://doi.org/...

**2. Another Great Paper**
   - Year: 2019
   - Citations: 198
   - DOI: https://doi.org/...

[...]

### Summary Statistics
- Total documents analyzed: 10
- Total citations: 892
- Average citations: 89.2

🟢 Confidence: 85.32% (high)
   Reasoning: Strong evidence base with 5 resolved entities, 10 documents, 15 citations
   Limitations: Limited to indexed data, Citation counts from database snapshot
```

## 🔧 Advanced Configuration

### Ollama Model Switching

Switch models for different use cases:

```bash
# Fast queries (llama3.2:1b)
MARIE_LLM_MODEL=llama3.2:1b

# Best quality (qwen2:1.5b) ⭐ Recommended
MARIE_LLM_MODEL=qwen2:1.5b

# Strong reasoning (phi3:mini)
MARIE_LLM_MODEL=phi3:mini
```

### Multi-Model Strategy

Use different models for different agents:

```python
# In orchestrator: phi3:mini for planning
# In retrieval: llama3.2:1b for fast queries
# In reporting: qwen2:1.5b for quality answers
```

### vLLM Settings (Alternative - 4GB VRAM)

```bash
# Maximum GPU usage
VLLM_GPU_MEMORY_UTILIZATION=0.9

# Reduce sequence length for more VRAM
VLLM_MAX_MODEL_LEN=2048

# Use quantization for larger models
VLLM_QUANTIZATION=awq  # or gptq
```

### Model Selection

```python
# Fast queries (SmolLM)
MARIE_LLM_MODEL=HuggingFaceTB/SmolLM-1.7B-Instruct

# Best quality (Qwen2) ⭐ Recommended
MARIE_LLM_MODEL=Qwen/Qwen2-1.5B-Instruct

# Strong reasoning (Phi-2, needs quantization)
MARIE_LLM_MODEL=microsoft/phi-2
```

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=marie_agent

# Integration tests (requires services running)
pytest -m integration
```

## 📊 Performance

### Model Comparison (Ollama)

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| qwen2:1.5b | 934 MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | General use ✅ |
| phi3:mini | 2.2 GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Complex reasoning |
| llama3.2:1b | 1.3 GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | Fast queries |

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **COLAV** - University of Antioquia
- **LangGraph** - Agent orchestration framework
- **vLLM** - High-performance LLM inference
- **Magentic** - Multi-agent design principles

## 📧 Contact

- GitHub: [@colav-playground](https://github.com/colav-playground)
- Email: colav@udea.edu.co
