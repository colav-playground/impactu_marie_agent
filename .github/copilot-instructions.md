# MARIE - Multi-Agent Research Intelligence Engine

## Project Overview
Python 3.10+ AI multi-agent system for research intelligence and scientific literature analysis. Built with LangChain, LangGraph, OpenSearch, and MongoDB. Uses Magentic for orchestration and supports Ollama/vLLM for LLM inference.

**Key Technologies:** LangChain 1.2+, LangGraph 1.0+, OpenSearch 3.1+, MongoDB 4.16+, Ollama, Python 3.10-3.12

## Build & Development

### Environment Setup
```bash
# ALWAYS use uv for package management, NEVER use pip directly
uv pip install -e .

# Required environment variables in .env:
OPENSEARCH_URL=http://localhost:9200
MONGODB_URI=mongodb://localhost:27017
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2:1.5b
```

### Commands
```bash
# Run interactive chat
marie_chat

# Run indexer  
marie_index --help

# Docker services
docker compose up -d opensearch mongodb ollama
```

## Project Structure
```
marie_agent/          # Main agent package
├── agents/          # Agent implementations (opensearch_expert, prompt_engineer, entity_resolution, retrieval, metrics, citations, reporting)
├── adapters/        # LLM adapters (Ollama, vLLM, fallback)
├── ports/           # Interface definitions
├── services/        # Business logic
├── cli.py           # Entry point (marie_chat command)
├── config.py        # Configuration
├── graph.py         # LangGraph workflow
├── orchestrator.py  # Magentic orchestration
└── state.py         # Shared state management

rag_indexer/         # Indexing system
├── cli.py           # marie_index command
├── indexer.py       # Main indexing logic
└── opensearch_manager.py

docs/                # Documentation (only when requested)
```

## Coding Standards

### CRITICAL RULES
1. **ALL code, comments, and docstrings MUST be in English**
2. **NEVER make automatic commits** - only when user explicitly says "commit" or "make a commit"
3. **NEVER create documentation/tests** unless explicitly requested
4. **ALWAYS use `uv` for package management** - never pip
5. **Keep responses concise** - no long explanations unless asked

### Code Style
- Use type hints: `typing, Optional, List, Dict, Any`
- Follow PEP 8
- Write comprehensive docstrings in English
- Use structured logging with context

### Agent Features
- OpenSearch Expert: Dynamic query generation with reflexion (up to 3 iterations), NO hardcoded queries
- Prompt Engineer: Self-improving prompts with reflexion (up to 2 iterations)
- Both agents log to OpenSearch for analytics (`impactu_marie_agent_*` indices)
- Reflexion pattern: evaluate → reflect → improve → retry

## Index Naming Standard
All indices MUST follow: `impactu_marie_agent_*`
- `impactu_*` - Main data indices
- `impactu_marie_agent_query_logs` - Query analytics
- `impactu_marie_agent_prompt_logs` - Prompt analytics  
- `impactu_marie_agent_plan_memory` - Planning memory
- `impactu_marie_agent_episodic_memory` - Episodic memory

## Key Files
- **CLI Entry Points:** `marie_agent/cli.py` (marie_chat), `rag_indexer/cli.py` (marie_index)
- **Core Agents:** `marie_agent/agents/*.py`
- **Config:** `marie_agent/config.py`, `.env`
- **Orchestration:** `marie_agent/orchestrator.py`, `marie_agent/graph.py`

## Common Pitfalls
- Don't use pip - use uv
- Don't create docs/tests proactively
- Don't make automatic commits
- Don't use non-English code
- Docker compose uses space, not hyphen: `docker compose` not `docker-compose`
