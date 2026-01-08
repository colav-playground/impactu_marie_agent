# MARIE Agent - Refactoring Analysis

**Date:** 2026-01-08
**Branch:** `refactor/system-optimization`
**Total LOC:** 9,865 lines
**Files:** 33 Python files

## 🔍 Issues Identified

### 1. HARDCODED VALUES (CRITICAL)

#### Index Names (Found: 7 instances)
- `opensearch_expert.py:129`: `"impactu_marie_agent_query_logs"`
- `prompt_engineer.py:94`: `"impactu_marie_agent_prompt_logs"`
- `memory_opensearch.py:32`: `"impactu_marie_agent_memory"`
- `memory_opensearch.py:199`: `"impactu_marie_agent_plan_memory"`
- `memory_opensearch.py:331`: `"impactu_marie_agent_episodic_memory"`

**Solution:** Move ALL to `config.py` as constants

#### URLs & Hosts (Found: 20+ instances)
- `http://localhost:9200` - OpenSearch (multiple files)
- `http://localhost:11434` - Ollama (ollama_adapter.py)
- `mongodb://localhost:27017` - MongoDB (config.py, memory files)

**Solution:** Already in config.py, but default values duplicated in function signatures

#### Magic Numbers (Found: 15+ instances)
- `timeout=30` - OpenSearch clients
- `max_tokens=512` - Various LLM calls
- `max_tokens=4096` - Config default
- `MAX_ITERATIONS = 3` - OpenSearch Expert
- `MAX_ITERATIONS = 2` - Prompt Engineer
- `MIN_RESULTS_THRESHOLD = 3` - OpenSearch Expert
- `max_size=100` - Memory systems

**Solution:** Create constants section in config.py

### 2. CODE DUPLICATION

#### OpenSearch Client Initialization
Found in 4+ places:
- `opensearch_expert.py`
- `prompt_engineer.py`  
- `memory_opensearch.py`
- Services

**Solution:** Create singleton OpenSearch connection manager

#### LLM Adapter Creation
Pattern repeated:
```python
llm = create_llm_adapter()
# or
llm = get_llm_adapter()
```
Used in 10+ files

**Solution:** Already using factory, but could be improved with dependency injection

#### Memory Initialization Pattern
Similar pattern in multiple agents for memory/logging

**Solution:** Create base agent class with common functionality

### 3. CODE QUALITY ISSUES

#### TODOs (Found: 4)
1. `entity_resolution.py:175`: Calculate actual confidence (currently hardcoded 0.8)
2. `retrieval.py:199`: Parse query and build MongoDB query
3. `retrieval.py:241`: Get citations from source
4. `tools.py:186`: Calculate actual h-index

**Solution:** Implement or document why they're deferred

#### Inconsistent Error Handling
- Some functions use try/except with logging
- Others let exceptions propagate
- No consistent error types

**Solution:** Define custom exceptions, consistent patterns

#### Type Hints
- Some functions missing return types
- Optional types not always explicit
- Dict/List without generics

**Solution:** Add complete type annotations

### 4. ARCHITECTURE ISSUES

#### Circular Dependencies Risk
- Agents import from services
- Services might import from agents
- Config imported everywhere

**Solution:** Review dependency graph, ensure one-way dependencies

#### Missing Abstractions
- No base Agent class (all agents implement similar patterns independently)
- No common interface for memory systems
- Logging setup duplicated

**Solution:** Create base classes and protocols

#### State Management
- `state.py` uses simple dicts
- No validation of state transitions
- No type safety

**Solution:** Use Pydantic models or dataclasses for state

### 5. LangGraph Integration

#### Current Implementation
- Simple graph in `graph.py`
- Basic routing in orchestrator
- No conditional edges based on state

**Potential Improvements:**
- Use LangGraph's conditional routing
- Implement proper checkpointing
- Add retry logic at graph level
- Use sub-graphs for complex agents

### 6. OPTIMIZATION OPPORTUNITIES

#### Caching
- No caching for:
  - OpenSearch schema inspection (called repeatedly)
  - Entity resolution results
  - LLM responses for identical queries

**Solution:** Implement caching layer (Redis or in-memory with TTL)

#### Async Operations
- Most operations are synchronous
- Could parallelize:
  - Multiple agent calls
  - OpenSearch queries
  - LLM calls (when appropriate)

**Solution:** Use asyncio where beneficial

#### Connection Pooling
- OpenSearch connections created per request
- No connection reuse

**Solution:** Implement connection pooling

### 7. CONFIGURATION MANAGEMENT

#### Current Issues
- Defaults duplicated in multiple places
- Some configs in code, some in .env
- No validation of config values
- No config documentation

**Solution:** 
- Single source of truth in config.py
- Pydantic validation
- Environment-specific configs
- Document all settings

### 8. LOGGING & OBSERVABILITY

#### Current State
- Good: Structured logging in most places
- Missing: 
  - Trace IDs across agents
  - Performance metrics
  - Request/response logging
  - Debug mode flag

**Solution:** Add trace context, metrics collection

### 9. TESTING

#### Current State
- No tests found in codebase
- No test structure

**Solution:** 
- Add pytest structure
- Unit tests for agents
- Integration tests for graph
- Mock external dependencies

### 10. DOCUMENTATION

#### Current State
- Good: Docstrings in most functions
- Missing:
  - Architecture diagrams
  - API documentation
  - Setup guides
  - Troubleshooting guide

**Solution:** Create comprehensive docs (but only when requested per rules)

## 🎯 REFACTORING PRIORITIES

### Phase 1: Critical Issues (Do Now)
1. ✅ Move ALL hardcoded index names to config
2. ✅ Move ALL magic numbers to config constants
3. ✅ Remove duplicate OpenSearch client initialization
4. ✅ Implement TODOs or document why deferred
5. ✅ Add missing type hints

### Phase 2: Architecture Improvements
6. Create base Agent class
7. Implement proper state management with types
8. Add connection pooling
9. Improve error handling consistency
10. Review and fix circular dependencies

### Phase 3: Performance
11. Add caching layer
12. Implement async where beneficial
13. Optimize LangGraph workflow

### Phase 4: Quality (If Time Permits)
14. Add trace IDs and metrics
15. Improve configuration management
16. Add unit tests

## 📝 NOTES

- Keep changes backwards compatible
- Test after each major change
- Document breaking changes
- Follow existing code style
- English only (per project rules)
