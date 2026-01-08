# MARIE Planning System - Improvements Needed

**Date:** 2026-01-08  
**Based on:** LangChain Plan-and-Execute patterns, ReAct, ReWOO, LLMCompiler

## 🎯 Current Issues

### Problem 1: Simple Planning
- Orchestrator generates 1-2 step plans only
- No decomposition of complex queries
- Missing sub-task identification

### Problem 2: No Task Dependencies
- Cannot reference previous task outputs
- No variable assignment (like ReWOO's #E1, #E2)
- Sequential execution only

### Problem 3: Limited Reasoning
- No explicit "thinking" steps
- No intermediate result verification
- No replanning capability

## 📚 Best Practices from Research

### 1. Plan-and-Execute Pattern
**Source:** LangChain Blog, Wang et al. (Plan-and-Solve Prompting)

**Key Components:**
1. **Planner:** Generates multi-step plan upfront
2. **Executor:** Executes each step with tools
3. **Re-planner:** Adjusts plan based on results

**Benefits:**
- ⚡ Faster execution (no LLM per action)
- 💸 Cost savings (smaller models for execution)
- 🏆 Better quality (explicit reasoning)

### 2. ReWOO (Reasoning WithOut Observations)
**Source:** Xu et al. (arXiv:2305.18323)

**Key Innovation:** Variable assignment in plans

**Example Plan:**
```
Plan: Find superbowl teams
E1: Search[Who is competing in the superbowl?]

Plan: Get quarterback for first team  
E2: LLM[Quarterback for first team of #E1]

Plan: Get quarterback for second team
E3: LLM[Quarterback for second team of #E1]

Plan: Get stats for first QB
E4: Search[Stats for #E2]

Plan: Get stats for second QB  
E5: Search[Stats for #E3]
```

**Benefits:**
- 📊 Task dependencies explicit
- 🔄 Can reference previous outputs
- ⚡ No replanning between steps

### 3. LLMCompiler
**Source:** Kim et al. (arXiv:2312.04511)

**Key Innovation:** DAG-based parallel execution

**Features:**
- Streams tasks as DAG
- Schedules tasks when dependencies met
- 3.6x speedup over sequential

### 4. ReAct (Reasoning + Acting)
**Source:** Yao et al. (arXiv:2210.03629)

**Pattern:**
```
Thought: I should search for X
Action: Search("X")
Observation: Result Y
Thought: Based on Y, I need Z
Action: Query("Z")
Observation: Result W
... (repeat)
```

**Benefits:**
- 🧠 Explicit reasoning traces
- 🔍 Handles exceptions
- ✅ Reduces hallucinations

## 🛠️ Proposed Improvements for MARIE

### Phase 1: Enhanced Plan Generation ⚡ HIGH PRIORITY

**Current:**
```python
plan = {
    "steps": ["Search for data", "Generate response"]
}
```

**Improved:**
```python
plan = {
    "query_analysis": {
        "intent": "comparison",
        "entities": ["researchers", "AI", "Colombia"],
        "metrics_needed": ["h-index", "publication_count"],
        "time_range": "last 3 years"
    },
    "steps": [
        {
            "id": "E1",
            "type": "search",
            "description": "Find Colombian researchers in AI",
            "agent": "retrieval",
            "filters": {"field": "AI", "country": "Colombia"},
            "output": "$researchers"
        },
        {
            "id": "E2", 
            "type": "compute",
            "description": "Calculate h-index for each researcher",
            "agent": "metrics",
            "input": "$researchers",
            "output": "$metrics"
        },
        {
            "id": "E3",
            "type": "rank",
            "description": "Rank by h-index, get top 5",
            "agent": "processing",
            "input": "$metrics",
            "output": "$top_researchers"
        },
        {
            "id": "E4",
            "type": "aggregate",
            "description": "Count papers (2021-2024)",
            "agent": "retrieval",
            "input": "$top_researchers",
            "filters": {"year_min": 2021},
            "output": "$paper_counts"
        },
        {
            "id": "E5",
            "type": "report",
            "description": "Generate final answer",
            "agent": "reporting",
            "inputs": ["$top_researchers", "$paper_counts"],
            "output": "final_answer"
        }
    ],
    "dependencies": {
        "E2": ["E1"],
        "E3": ["E2"],
        "E4": ["E3"],
        "E5": ["E3", "E4"]
    }
}
```

### Phase 2: ReAct-Style Reasoning

**Add explicit reasoning steps:**
```python
state = {
    "reasoning_trace": [
        {
            "thought": "Need to find researchers first",
            "action": "retrieval",
            "observation": "Found 50 researchers"
        },
        {
            "thought": "Must calculate metrics to rank them",
            "action": "metrics",
            "observation": "Computed h-index for 50"
        }
    ]
}
```

### Phase 3: Variable Assignment System

**Implement variable store:**
```python
class VariableStore:
    def __init__(self):
        self.vars = {}
    
    def set(self, var_name: str, value: Any):
        self.vars[var_name] = value
        logger.info(f"Variable set: ${var_name}")
    
    def get(self, var_name: str) -> Any:
        return self.vars.get(var_name)
    
    def resolve(self, text: str) -> str:
        """Replace $var references with actual values."""
        for var_name, value in self.vars.items():
            text = text.replace(f"${var_name}", str(value))
        return text
```

### Phase 4: Query Decomposition

**Enhance orchestrator planning prompt:**
```python
DECOMPOSITION_PROMPT = """
You are an expert query planner for a research intelligence system.

Given a complex query, decompose it into atomic sub-tasks that can be executed independently.

Query: {query}

Analysis:
1. What is the main intent? (search, comparison, ranking, aggregation, etc.)
2. What entities are mentioned?
3. What metrics/calculations are needed?
4. What filters/constraints apply?
5. What is the expected output format?

Generate a detailed plan with:
- Task ID (E1, E2, E3...)
- Task type (search, compute, filter, rank, aggregate, report)
- Description
- Agent to execute
- Input variables (reference previous tasks with $E1, $E2)
- Output variable name
- Dependencies

Example for "Top 5 Colombian AI researchers by h-index with 2020-2024 papers":

E1: Search researchers (field=AI, country=Colombia) → $researchers
E2: Compute h-index for $researchers → $h_indices  
E3: Rank $h_indices, take top 5 → $top_5
E4: Count papers for $top_5 (year>=2020) → $paper_counts
E5: Generate report from $top_5 and $paper_counts → final_answer

Dependencies: E2→E1, E3→E2, E4→E3, E5→[E3,E4]

Now generate a plan for: {query}
"""
```

### Phase 5: Parallel Execution (Future)

**Use async_ops for independent tasks:**
```python
from marie_agent.core.async_ops import run_parallel_dict

# Execute independent tasks in parallel
results = await run_parallel_dict({
    'entity_resolution': lambda: resolve_entities(query),
    'schema_inspection': lambda: get_schema(index),
    'query_parsing': lambda: parse_query(query)
})
```

## 🎯 Implementation Priority

### ✅ Phase 1: Enhanced Plan Generation (CRITICAL)
**Effort:** 2-3 hours  
**Impact:** High  
**Files:**
- `marie_agent/orchestrator.py` - Update planning prompts
- `marie_agent/core/plan_generator.py` - Add detailed plan structure

### 🔄 Phase 2: Variable Assignment System
**Effort:** 1-2 hours  
**Impact:** High  
**Files:**
- `marie_agent/core/variable_store.py` - NEW
- `marie_agent/state.py` - Add variable_store field
- `marie_agent/agents/*.py` - Use variable store

### 📊 Phase 3: ReAct Reasoning Traces
**Effort:** 1 hour  
**Impact:** Medium  
**Files:**
- `marie_agent/state.py` - Add reasoning_trace
- `marie_agent/agents/*.py` - Log thoughts/observations

### ⚡ Phase 4: Parallel Execution (Optional)
**Effort:** 2-3 hours  
**Impact:** Medium (performance)  
**Files:**
- `marie_agent/graph.py` - Add parallel node execution
- Use existing `async_ops.py`

## 📈 Expected Improvements

### Before (Current System):
```
Query: "Top 5 Colombian AI researchers by h-index, papers 2020-2024"

Plan:
1. Search for data
2. Generate response

Result: Generic answer, no ranking, no metrics
```

### After (With Improvements):
```
Query: "Top 5 Colombian AI researchers by h-index, papers 2020-2024"

Decomposed Plan:
E1: Search(field=AI, country=Colombia) → $researchers (50 found)
E2: ComputeMetrics($researchers) → $metrics (h-index calculated)
E3: Rank($metrics, limit=5) → $top_5 (top researchers)
E4: CountPapers($top_5, year>=2020) → $counts (paper counts)
E5: Report($top_5, $counts) → answer

Result: "The top 5 Colombian AI researchers by h-index are:
1. Dr. X (h-index: 45, papers 2020-2024: 23)
2. Dr. Y (h-index: 38, papers 2020-2024: 19)
..."
```

### Metrics:
- **Plan quality:** +300% (1-2 steps → 5-7 steps)
- **Answer accuracy:** +150% (detailed vs generic)
- **Query handling:** Complex queries now supported
- **Traceability:** Full reasoning visible

## 🚀 Quick Start Implementation

### Step 1: Update Planning Prompt (30 min)
Edit `orchestrator.py` line ~200:
```python
# Add detailed decomposition instructions
# Use examples with E1, E2, $var syntax
```

### Step 2: Add Variable Store (1 hour)
Create `core/variable_store.py`:
```python
class VariableStore:
    # Implementation from Phase 3 above
```

### Step 3: Update Agents (1 hour)
Each agent should:
- Accept input variables
- Store output in variable_store
- Reference previous outputs

### Step 4: Test Complex Queries (30 min)
```bash
marie_chat
# Test: "Compare top 3 researchers in ML vs AI by citations"
# Should generate 8-10 step plan with dependencies
```

## 📚 References

1. **LangChain Plan-and-Execute:**  
   https://blog.langchain.dev/planning-agents/

2. **ReAct Paper (Yao et al.):**  
   https://arxiv.org/abs/2210.03629

3. **ReWOO Paper (Xu et al.):**  
   https://arxiv.org/abs/2305.18323

4. **LLMCompiler (Kim et al.):**  
   https://arxiv.org/abs/2312.04511

5. **Plan-and-Solve Prompting (Wang et al.):**  
   https://arxiv.org/abs/2305.04091

## ✅ Next Steps

1. Review this document
2. Choose implementation phases (recommend 1-3)
3. Update orchestrator planning logic
4. Add variable store system
5. Test with complex queries
6. Monitor improvements

---

**Status:** Ready for implementation  
**Estimated Total Time:** 5-8 hours for Phases 1-3  
**Impact:** Transforms MARIE from simple Q&A to complex reasoning system
