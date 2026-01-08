# IMPACTU MARIE Agent - GitHub Copilot Instructions

## Project Overview

MARIE (Multi-Agent Research Intelligence Engine) is a sophisticated multi-agent AI system for research intelligence and scientific literature analysis. Built with LangChain, LangGraph, and OpenSearch, it provides intelligent querying, entity resolution, metrics computation, and report generation for academic research.

**Key Technologies:** Python 3.10+, LangChain, LangGraph, OpenSearch, MongoDB, Ollama, Docker

**Project Type:** AI/ML Multi-Agent System with RAG capabilities

## Build & Development

### Environment Setup
```bash
# Install dependencies with uv (ALWAYS use uv, not pip)
uv pip install -e .

# Environment variables required
# Create .env file with:
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

# Run tests (when available)
pytest

# Docker services
docker compose up -d opensearch mongodb ollama
```

## Architecture

### Core Components
- ✅ Write docstrings for functions and classes

### 2. Version Control - ⚠️ CRITICAL RULES ⚠️
- 🚫 **STRICTLY PROHIBITED: NEVER, EVER make automatic commits without explicit permission**
- 🚫 **STRICTLY PROHIBITED: NEVER, EVER run `git commit` automatically**
- 🚫 **STRICTLY PROHIBITED: NEVER, EVER run `git push` automatically**
### Core Components
- **Orchestrator:** Magentic-based planning and routing
- **OpenSearch Expert:** Dynamic query generation with reflexion (NO hardcoded queries)
- **Prompt Engineer:** Self-improving prompt generation with reflexion
- **Entity Resolution:** Colombian research entities (institutions, authors, groups)
- **Retrieval Agent:** Fetch documents from OpenSearch
- **Metrics Agent:** Compute research metrics
- **Citations Agent:** Format references
- **Reporting Agent:** Generate final responses

### Key Features
1. **Reflexion Pattern:** Both OpenSearch Expert and Prompt Engineer use iterative self-improvement
2. **No Hardcoded Queries:** All OpenSearch queries are dynamically generated based on schema inspection
3. **Query Logging:** All queries logged to `impactu_marie_agent_query_logs` for analytics
4. **Prompt Logging:** All prompts logged to `impactu_marie_agent_prompt_logs` for analytics
5. **Memory Systems:** Learn from past successful attempts

## Development Rules

### 1. Code Style - ENGLISH ONLY
- ✅ **ALL code, comments, and docstrings MUST be in English**
- ✅ Use type hints: `typing, Optional, List, Dict, Any`
- ✅ Write comprehensive docstrings
- ✅ Follow PEP 8 style guide
- ✅ Use structured logging with context

### 2. Version Control - ⚠️ CRITICAL RULES ⚠️
- 🚫 **STRICTLY PROHIBITED: NEVER, EVER make automatic commits**
- 🚫 **STRICTLY PROHIBITED: NEVER run `git commit` automatically**
- 🚫 **STRICTLY PROHIBITED: NEVER run `git push` automatically**
- 🚫 **STRICTLY PROHIBITED: NEVER suggest making commits**
- ✅ **ONLY commit when user explicitly says "make a commit" or "commit this"**
- ✅ **ONLY push when user explicitly says "push" or "push to github"**
- ✅ **ALWAYS ask "Do you want me to commit these changes?" before committing**
- ✅ Let the user handle ALL git operations unless explicitly delegated

### 3. Documentation & Testing - NO UNSOLICITED CREATION
- ❌ **DO NOT create documentation files unless EXPLICITLY requested**
- ❌ **DO NOT create tests unless EXPLICITLY requested**
- ❌ **DO NOT create markdown files without explicit request**
- ❌ **DO NOT give long explanations unless asked**
- ✅ Place docs in `docs/` folder ONLY when explicitly asked
- ✅ Only create what is specifically requested
- ✅ Keep responses concise

### 4. Package Management
- ✅ **ALWAYS use `uv` for package installation**
- ✅ Example: `uv pip install package-name`
- ✅ Never use pip directly

### 5. Problem Solving
- ✅ Search the internet when unclear
- ✅ Research best practices before implementing
- ✅ Validate assumptions with resources
- ✅ Ask for clarification only after research

## Project Structure

```
impactu_marie_agent/
├── marie_agent/              # Main agent package
│   ├── agents/              # Agent implementations
│   │   ├── opensearch_expert.py   # Dynamic query generation with reflexion
│   │   ├── prompt_engineer.py     # Self-improving prompts with reflexion
│   │   ├── entity_resolution.py   # Colombian entities
│   │   ├── retrieval.py           # Document retrieval
│   │   ├── metrics.py             # Research metrics
│   │   ├── citations.py           # Reference formatting
│   │   └── reporting.py           # Final answer generation
│   ├── adapters/            # LLM adapters (Ollama, vLLM)
│   ├── ports/               # Interface definitions
│   ├── services/            # Business logic
│   ├── cli.py               # CLI entry point (marie_chat)
│   ├── config.py            # Configuration
│   ├── graph.py             # LangGraph workflow
│   ├── orchestrator.py      # Magentic orchestration
│   └── state.py             # Shared state management
├── rag_indexer/             # Indexing system
│   ├── cli.py               # Indexer CLI (marie_index)
│   ├── indexer.py           # Main indexing logic
│   └── opensearch_manager.py # OpenSearch operations
├── docs/                    # Documentation (only when requested)
├── .github/
│   └── copilot-instructions.md  # This file
├── docker-compose.yml       # Services: OpenSearch, MongoDB, Ollama
├── Dockerfile               # Container definition
├── pyproject.toml           # Python project config
└── README.md                # Project overview
```

## Key Files to Know

### Entry Points
- **marie_chat:** `marie_agent/cli.py:main()` - Interactive chat interface
- **marie_index:** `rag_indexer/cli.py:main()` - Indexing CLI

### Core Agents
- **OpenSearch Expert:** `marie_agent/agents/opensearch_expert.py`
  - Dynamic query generation (NO hardcoded queries)
  - Reflexion pattern (up to 3 iterations)
  - Query logging to `impactu_marie_agent_query_logs`
- **Prompt Engineer:** `marie_agent/agents/prompt_engineer.py`
  - Self-improving prompts
  - Reflexion pattern (up to 2 iterations)
  - Prompt logging to `impactu_marie_agent_prompt_logs`

### Configuration
- **Config:** `marie_agent/config.py` - All settings
- **Env Variables:** `.env` file (OpenSearch, MongoDB, Ollama URLs)

## Critical Implementation Details

### OpenSearch Indices (Follow Naming Standard)
- `impactu_*` - Main data indices
- `impactu_marie_agent_query_logs` - Query analytics
- `impactu_marie_agent_prompt_logs` - Prompt analytics
- `impactu_marie_agent_plan_memory` - Planning memory
- `impactu_marie_agent_episodic_memory` - Episodic memory

### Agent Features
1. **Reflexion Pattern:** Both OpenSearch Expert and Prompt Engineer iterate to improve
2. **Dynamic Queries:** Schema inspection → LLM generation → No hardcoded queries
3. **Memory Systems:** Learn from successful attempts
4. **Query Logging:** Full observability in OpenSearch
5. **Natural Language:** Conversational responses, no canned phrases

## Commands

### Development
```bash
# Install (ALWAYS use uv)
uv pip install -e .

# Run chat
marie_chat

# Run indexer
marie_index --help

# Docker (note: docker compose with space)
docker compose up -d
docker compose logs -f agent
docker compose down
```

### Testing (when tests exist)
```bash
pytest
```

## Remember - Critical Rules

### 🚫 NEVER DO THESE:
1. Make automatic commits or pushes
2. Create documentation without explicit request
3. Create tests without explicit request
4. Use pip (always use uv)
5. Create markdown files proactively

### ✅ ALWAYS DO THESE:
1. Write ALL code in English
2. Use uv for package management
3. Search internet before asking
4. Keep responses concise
5. Follow the reflexion pattern for agents
