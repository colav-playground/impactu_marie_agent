# MARIE Agent - Hexagonal Architecture

## 🏗️ Architecture Overview

MARIE implements **Hexagonal Architecture** (Ports & Adapters) for clean separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                      Presentation                        │
│                  (CLI, API, UI)                         │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                   Application Layer                      │
│              (Orchestrator, Workflow)                   │
└─────────┬────────────────────────────────┬──────────────┘
          │                                │
┌─────────▼─────────┐          ┌──────────▼──────────────┐
│    Domain Core    │          │     Ports (Interfaces)   │
│  - Entities       │          │  - LLMPort              │
│  - Use Cases      │◄─────────┤  - DatabasePort         │
│  - Value Objects  │          │  - SearchPort           │
└───────────────────┘          └─────────┬───────────────┘
                                         │
                               ┌─────────▼───────────────┐
                               │   Adapters (Impl)       │
                               │  - VLLMAdapter          │
                               │  - AnthropicAdapter     │
                               │  - MongoDBAdapter       │
                               │  - OpenSearchAdapter    │
                               └─────────────────────────┘
```

## 📁 Directory Structure

```
marie_agent/
├── core/                  # 🎯 Domain Layer (Pure business logic)
│   ├── entities/          # Domain entities
│   ├── use_cases/         # Business use cases
│   └── value_objects/     # Value objects
│
├── ports/                 # 🔌 Ports (Interfaces/Contracts)
│   ├── llm_port.py       # LLM service interface
│   ├── database_port.py   # Database interface
│   └── search_port.py     # Search service interface
│
├── adapters/              # 🔧 Adapters (Implementations)
│   ├── vllm_adapter.py   # vLLM local inference
│   ├── anthropic_adapter.py  # Anthropic API
│   ├── mongodb_adapter.py    # MongoDB implementation
│   ├── opensearch_adapter.py # OpenSearch implementation
│   └── llm_factory.py        # Factory for LLM adapters
│
├── agents/                # 🤖 Specialized Agents
│   ├── entity_resolution.py
│   ├── retrieval.py
│   ├── metrics.py
│   ├── citations.py
│   └── reporting.py
│
├── orchestrator.py        # 🎭 Workflow orchestration
├── graph.py              # LangGraph workflow definition
└── cli.py                # Command-line interface
```

## 🔌 Ports (Interfaces)

### LLMPort
Defines the contract for all LLM services:

```python
from marie_agent.ports.llm_port import LLMPort

class MyCustomLLM(LLMPort):
    def is_available(self) -> bool: ...
    def parse_query(self, query: str) -> Dict: ...
    def think(self, prompt: str) -> str: ...
    def generate(self, prompt: str) -> str: ...
```

## 🔧 Adapters (Implementations)

### Available Adapters

#### 1. VLLMAdapter (Local Inference)
```python
from marie_agent.adapters.vllm_adapter import VLLMAdapter

llm = VLLMAdapter(
    model_name="Qwen/Qwen2-1.5B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=2048
)
```

**Optimized for 4GB VRAM:**
- ✅ Qwen2-1.5B-Instruct (3GB) - Best quality/size balance
- ✅ Phi-2 (5.5GB) - Strong reasoning (requires quantization)
- ✅ SmolLM-1.7B (3.4GB) - Fast inference

#### 2. AnthropicAdapter (API-based)
```python
from marie_agent.llm import MarieLLM

llm = MarieLLM()  # Uses ANTHROPIC_API_KEY
```

### LLM Factory
Automatically selects the appropriate adapter based on configuration:

```python
from marie_agent.adapters.llm_factory import get_llm_adapter

llm = get_llm_adapter()  # Returns VLLMAdapter or AnthropicAdapter
```

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Provider Selection
MARIE_LLM_PROVIDER=vllm  # vllm (local) or anthropic (API)

# vLLM Configuration (for local models)
MARIE_LLM_MODEL=Qwen/Qwen2-1.5B-Instruct
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_MODEL_LEN=2048
VLLM_TENSOR_PARALLEL_SIZE=1

# Anthropic Configuration (if using API)
MARIE_LLM_PROVIDER=anthropic
MARIE_LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_key_here
```

## 🚀 Quick Start

### 1. Download Models (for vLLM)

```bash
python download_models.py
```

This will download optimized models for 4GB VRAM GPU.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your preferences
```

### 3. Run MARIE

```bash
# With vLLM (local)
export MARIE_LLM_PROVIDER=vllm
python -m marie_agent.cli "What are the top 5 most cited papers from UdeA?"

# With Anthropic API
export MARIE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_key
python -m marie_agent.cli "What are the top 5 most cited papers from UdeA?"
```

## 🎯 Benefits of Hexagonal Architecture

### 1. **Testability**
Easy to mock dependencies and test business logic in isolation:

```python
class MockLLM(LLMPort):
    def parse_query(self, query: str):
        return {"intent": "test", "entities": {}}

# Test with mock
orchestrator.llm = MockLLM()
```

### 2. **Flexibility**
Switch between implementations without changing business logic:

```python
# Development: Use fast local model
llm = VLLMAdapter("SmolLM-1.7B")

# Production: Use high-quality API
llm = AnthropicAdapter()
```

### 3. **Maintainability**
Clear separation of concerns:
- **Core**: Business rules (doesn't know about LLMs, databases)
- **Ports**: Contracts (interfaces)
- **Adapters**: Technical implementation details

### 4. **Extensibility**
Easy to add new implementations:

```python
# Add OpenAI adapter
class OpenAIAdapter(LLMPort):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def parse_query(self, query: str):
        # OpenAI-specific implementation
        pass
```

## 🔄 Dependency Flow

```
CLI/API → Orchestrator → Use Cases → Ports → Adapters → External Services
                                       ↑
                                       └─── Domain Core (pure logic)
```

**Key principle**: Dependencies point inward. Core domain has no dependencies on external systems.

## 🧪 Testing Strategy

```python
# Unit Tests: Mock all ports
def test_orchestrator_planning():
    mock_llm = MockLLMPort()
    orchestrator = Orchestrator(llm=mock_llm)
    result = orchestrator.plan(query)
    assert result.status == "success"

# Integration Tests: Use real adapters
def test_vllm_integration():
    llm = VLLMAdapter("Qwen/Qwen2-1.5B-Instruct")
    result = llm.parse_query("test query")
    assert "intent" in result

# E2E Tests: Full system
def test_marie_end_to_end():
    result = run_marie_query("test query")
    assert result["status"] == "completed"
```

## 📊 Model Performance (4GB VRAM)

| Model | Size | Speed | Quality | VRAM | Recommended |
|-------|------|-------|---------|------|-------------|
| Qwen2-1.5B | 3GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | 3.2GB | ✅ Best |
| SmolLM-1.7B | 3.4GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | 3.5GB | Fast |
| Phi-2 | 5.5GB | ⚡⚡ | ⭐⭐⭐⭐ | 4GB+ | Reasoning* |

*Requires 4-bit quantization for 4GB VRAM

## 🛠️ Advanced Configuration

### Multiple Models

```python
# Specialized models for different tasks
fast_llm = VLLMAdapter("SmolLM-1.7B")  # Quick queries
reasoning_llm = VLLMAdapter("Phi-2")   # Complex reasoning

orchestrator.llm = reasoning_llm if complex_query else fast_llm
```

### Quantization (for Phi-2 on 4GB)

```python
llm = VLLMAdapter(
    model_name="microsoft/phi-2",
    quantization="awq",  # or "gptq"
    gpu_memory_utilization=0.95
)
```

## 📚 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [HuggingFace Models](https://huggingface.co/models)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
