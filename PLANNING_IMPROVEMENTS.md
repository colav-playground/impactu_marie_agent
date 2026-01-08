# MARIE Planning System - Improvements Needed

**Date:** 2026-01-08  
**Based on:** LangChain Plan-and-Execute patterns, ReAct, ReWOO, LLMCompiler  
**Compatible with:** Microsoft Magentic-One Architecture (already implemented in MARIE)

## ✅ Current Magentic Architecture (PRESERVED)

MARIE already implements core Magentic-One components:

### Existing Components:
1. **OrchestratorAgent** - Central orchestrator (Magentic Lead Agent equivalent)
2. **ContextWindow** - Conversation history management
3. **ProgressTracker** - Progress ledger for execution tracking
4. **QualityEvaluator** - Response quality assessment
5. **DynamicPlanGenerator** - Adaptive plan generation with memory
6. **ActionGuard** - Safety checks for agent actions
7. **Episodic Memory** - Long-term memory (OpenSearch-backed)

### Key Magentic Principles Already Implemented:
- ✅ **Orchestrator-driven workflow** - OrchestratorAgent coordinates all agents
- ✅ **Progress ledger** - ProgressTracker maintains execution state
- ✅ **Dynamic replanning** - Can adjust plans based on results
- ✅ **Context management** - ContextWindow prevents overflow
- ✅ **Quality evaluation** - QualityEvaluator checks outputs
- ✅ **Human-in-loop** - ActionGuard + human_manager integration

## 🎯 Current Issues

### Problem 1: Simple Planning
- Orchestrator generates 1-2 step plans only
- No decomposition of complex queries
- Missing sub-task identification
- **DynamicPlanGenerator exists but needs better prompts**

### Problem 2: No Task Dependencies
- Cannot reference previous task outputs
- No variable assignment (like ReWOO's #E1, #E2)
- Sequential execution only
- **ProgressTracker tracks progress but doesn't pass variables**

### Problem 3: Limited Reasoning
- No explicit "thinking" steps
- No intermediate result verification
- No replanning capability (exists but not triggered)
- **QualityEvaluator exists but needs integration with replanning**

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

## 🛠️ Proposed Improvements for MARIE (Magentic-Compatible)

### Phase 1: Enhanced Plan Generation ⚡ HIGH PRIORITY

**IMPORTANT:** We keep Magentic's ProgressTracker and DynamicPlanGenerator, just enhance them.

**Current (DynamicPlanGenerator output):**
```python
plan = {
    "steps": ["Search for data", "Generate response"]
}
```

**Improved (Enhanced DynamicPlanGenerator):**
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

### Phase 2: Variable Store System (Magentic-Compatible)

**Integration with existing ProgressTracker:**

The ProgressTracker already tracks execution state. We extend it with variable storage:

```python
# In core/progress_ledger.py - EXTEND existing class
class ProgressTracker:
    def __init__(self, llm):
        # ... existing code ...
        self.variable_store = {}  # NEW: Add variable storage
    
    def record_completion(self, step_id: str, result: Any):
        """Record step completion and store output variable."""
        # ... existing code ...
        
        # NEW: Store result as variable if step defines output
        if 'output_var' in step:
            var_name = step['output_var']
            self.variable_store[var_name] = result
            logger.info(f"Variable stored: ${var_name}")
    
    def resolve_variables(self, text: str) -> str:
        """Replace $var references with actual values."""
        for var_name, value in self.variable_store.items():
            text = text.replace(f"${var_name}", str(value))
        return text
    
    def get_variable(self, var_name: str) -> Any:
        """Get variable value."""
        return self.variable_store.get(var_name)
```

**NO new files needed - just extend existing ProgressTracker!**

### Phase 3: ReAct Reasoning Traces (Magentic-Compatible)

**Integration with existing ContextWindow:**

The ContextWindow already stores messages. We add structured reasoning:

```python
# In core/context_window.py - EXTEND existing class
class ContextWindow:
    def __init__(self, max_tokens: int = 8000):
        # ... existing code ...
        self.reasoning_trace = []  # NEW: Add reasoning trace
    
    def add_reasoning_step(
        self,
        thought: str,
        action: str,
        observation: str,
        agent_name: str
    ):
        """Add a ReAct-style reasoning step."""
        step = {
            "thought": thought,
            "action": action,
            "observation": observation,
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.reasoning_trace.append(step)
        
        # Also add to regular messages
        self.add_message(
            role="agent",
            content=f"Thought: {thought}\nAction: {action}\nObservation: {observation}",
            agent_name=agent_name,
            metadata={"type": "reasoning_trace"}
        )
```

**NO new files - just extend ContextWindow!**
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

## 🎯 Implementation Priority (Magentic-Compatible)

### ✅ Phase 1: Enhanced Plan Generation (CRITICAL)
**Effort:** 2-3 hours  
**Impact:** High  
**Magentic Integration:** Update DynamicPlanGenerator prompts  
**Files:**
- `marie_agent/core/plan_generator.py` - Update decomposition prompt
- `marie_agent/orchestrator.py` - Better query parsing

**Changes:**
- Enhance existing planning prompts (no new components)
- Add structured plan format with task types
- Keep using existing DynamicPlanGenerator class

### 🔄 Phase 2: Variable Store in ProgressTracker (HIGH)
**Effort:** 1-2 hours  
**Impact:** High  
**Magentic Integration:** Extend existing ProgressTracker  
**Files:**
- `marie_agent/core/progress_ledger.py` - Add variable_store dict
- `marie_agent/agents/*.py` - Store/retrieve variables

**Changes:**
- Add variable_store dict to ProgressTracker.__init__
- Add record_completion() to store output variables
- Add resolve_variables() to replace $var references
- NO new classes needed!

### 📊 Phase 3: ReAct Traces in ContextWindow (MEDIUM)
**Effort:** 1 hour  
**Impact:** Medium  
**Magentic Integration:** Extend existing ContextWindow  
**Files:**
- `marie_agent/core/context_window.py` - Add reasoning_trace list
- `marie_agent/agents/*.py` - Log thoughts/observations

**Changes:**
- Add reasoning_trace list to ContextWindow
- Add add_reasoning_step() method
- Agents call this before/after actions
- NO new classes needed!

### ⚡ Phase 4: Parallel Execution (Optional)
**Effort:** 2-3 hours  
**Impact:** Medium (performance)  
**Magentic Integration:** Compatible with orchestrator  
**Files:**
- `marie_agent/graph.py` - Use async_ops for parallel edges
- Use existing `async_ops.py`

**Changes:**
- Identify independent tasks in plan
- Use run_parallel_dict() from existing async_ops
- Keep orchestrator coordination intact
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

## ✅ Next Steps (Magentic-Compatible Implementation)

### Priority 1: Enhance DynamicPlanGenerator (2-3 hours)
1. Update `core/plan_generator.py` prompt for detailed decomposition
2. Add task type classification (search, compute, rank, etc.)
3. Add variable assignment ($E1, $E2) in plan output
4. Keep all existing Magentic components intact

### Priority 2: Extend ProgressTracker (1-2 hours)
1. Add `variable_store` dict to ProgressTracker class
2. Extend `record_completion()` to store output variables
3. Add `resolve_variables()` method for $var replacement
4. Update agents to use variables

### Priority 3: Extend ContextWindow (1 hour)
1. Add `reasoning_trace` list to ContextWindow
2. Add `add_reasoning_step()` method
3. Agents log thought/action/observation

### Testing Complex Queries
```bash
marie_chat
# Test: "Compare top 3 researchers in ML vs AI by citations"
# Expected: 8-10 step plan with dependencies
# Expected: Variable references like $ml_researchers, $ai_researchers
```

---

## 🔑 Key Magentic Principles Preserved

✅ **Orchestrator Coordination** - OrchestratorAgent remains central coordinator  
✅ **Progress Tracking** - ProgressTracker extended, not replaced  
✅ **Context Management** - ContextWindow extended, not replaced  
✅ **Quality Evaluation** - QualityEvaluator integration maintained  
✅ **Human-in-Loop** - ActionGuard and human_manager intact  
✅ **Dynamic Replanning** - Orchestrator can still adjust plans  
✅ **Memory Systems** - Episodic memory (OpenSearch) preserved

**All improvements are EXTENSIONS of existing Magentic components, not replacements!**

---

**Status:** Ready for Magentic-compatible implementation  
**Estimated Total Time:** 4-6 hours for Phases 1-3  
**Impact:** Complex query handling without breaking Magentic architecture  
**Risk:** LOW - Only extending existing components
