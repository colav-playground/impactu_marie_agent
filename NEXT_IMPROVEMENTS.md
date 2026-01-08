# MARIE - Next Critical Improvements

**Date:** 2026-01-08  
**Current Status:** Planning implemented but agents not executing searches  
**Priority:** HIGH

## 🐛 **PROBLEMA ACTUAL IDENTIFICADO:**

### Issue: Agents Generate Plans But Don't Execute Real Searches

**Síntomas:**
```
Query: "Top 3 Colombian AI researchers"
Plan: 4 steps generado ✅
Execution: Agents ejecutados ✅  
Result: "Lo siento, no tengo información..." ❌
```

**Root Causes:**

1. **Routing Mismatch:** 
   - Plan dice: "Find products for universities"
   - Router no entiende "Find products" → defaulting to reporting
   - Retrieval agent NUNCA se ejecuta

2. **No Tool Invocation:**
   - Retrieval agent tiene código para OpenSearch
   - Pero el prompt del LLM genera texto, no ejecuta búsquedas
   - Agents necesitan usar TOOLS no solo generar texto

3. **No Capability Assessment:**
   - Sistema no verifica QUÉ puede buscar antes de planear
   - No sabe qué índices existen en OpenSearch
   - No sabe qué datos tiene disponibles

## 🎯 **SOLUCIÓN PROPUESTA:**

### Phase 1: Fix Routing (30 min) ✅ URGENT

**Problema:** Routing no mapea correctamente títulos de plan a agentes

**Fix:**
```python
# En core/routing.py - should_continue()

# Get step metadata (added in Phase 1 planning)
step_metadata = steps[current_step].metadata if hasattr(steps[current_step], 'metadata') else {}
agent_name = step_metadata.get('agent_name') or step.get('agent_name')

# Use explicit agent_name from metadata FIRST
if agent_name:
    logger.info(f"Routing to agent from metadata: {agent_name}")
    return agent_name

# Fallback to title parsing (current behavior)
...
```

**Benefit:** Plans con agent_name explícito se ejecutan correctamente

### Phase 2: Agent Tools Integration (2-3h) 🔧 HIGH PRIORITY

**Problema:** Agents no invocan herramientas reales (OpenSearch Expert, MongoDB)

**Current Flow:**
```
Retrieval Agent:
1. Get optimized search terms from LLM (texto)
2. Call _search_opensearch() internamente
3. Return results
```

**Needed Flow:**
```
Retrieval Agent:
1. Invoke OpenSearch Expert tool (returns dynamic query)
2. Execute query on OpenSearch
3. Store results in variable_store ($results)
4. Return structured data
```

**Implementation:**

```python
# In retrieval.py
def retrieve(self, state: AgentState) -> AgentState:
    query = state["user_query"]
    
    # 1. Get ProgressTracker for variable resolution
    from marie_agent.core.progress_ledger import ProgressTracker
    tracker = state.get('progress_tracker')  # Passed from orchestrator
    
    # 2. Resolve any input variables from previous steps
    if tracker:
        query = tracker.resolve_variables(query)
    
    # 3. INVOKE OpenSearch Expert as TOOL
    from marie_agent.agents.opensearch_expert import opensearch_expert_agent
    
    expert_state = opensearch_expert_agent.build_query(
        query=query,
        filters=state.get('filters', {}),
        entities=state.get('entities_resolved', {})
    )
    
    opensearch_query = expert_state.get('opensearch_query')
    
    # 4. Execute query
    results = self._execute_opensearch_query(opensearch_query)
    
    # 5. Store in variable_store
    if tracker:
        output_var = state.get('current_step_metadata', {}).get('output_var')
        if output_var:
            tracker.store_variable(output_var, results, task_id)
    
    # 6. Update state
    state['retrieved_data'] = results
    return state
```

**Key Changes:**
- Agents invoke OTHER agents as tools
- Results stored in variable_store
- Next agent can reference with $variable

### Phase 3: Capability Assessment (1-2h) 🔍 MEDIUM

**Problema:** Sistema no sabe qué puede buscar

**Solution:** Pre-flight check antes de generar plan

```python
# In orchestrator.py - before generate_plan()

def assess_capabilities(self, query: str) -> Dict[str, Any]:
    """
    Assess what the system CAN answer before planning.
    
    Returns:
        {
            'can_answer': bool,
            'available_indices': [...],
            'available_entities': [...],
            'limitations': [...]
        }
    """
    from marie_agent.agents.opensearch_expert import get_opensearch_manager
    
    manager = get_opensearch_manager()
    
    # Get all impactu indices
    indices = manager.list_indices(pattern='impactu_*')
    
    # Get entity types available
    entities = self._detect_entity_types_in_indices(indices)
    
    # Determine if query is answerable
    query_entities = self._extract_entities_from_query(query)
    
    can_answer = any(
        entity in entities for entity in query_entities
    )
    
    return {
        'can_answer': can_answer,
        'available_indices': indices,
        'available_entities': entities,
        'suggested_refinement': self._suggest_query_refinement(query, entities)
    }
```

**Usage:**
```python
# Before planning
capabilities = self.assess_capabilities(query)

if not capabilities['can_answer']:
    # Generate clarifying question or refine query
    return self._ask_for_clarification(capabilities)

# Generate plan WITH capability context
plan = self.plan_generator.generate_plan(
    query=query,
    capabilities=capabilities  # NEW
)
```

### Phase 4: OpenSearch Expert as Reusable Tool (1h) 🛠️ MEDIUM

**Problema:** OpenSearch Expert no es reutilizable por otros agentes

**Solution:** Extract as standalone service

```python
# In marie_agent/services/opensearch_tool.py

class OpenSearchTool:
    """
    Reusable tool for dynamic OpenSearch query generation.
    Can be invoked by any agent.
    """
    
    def __init__(self):
        from marie_agent.agents.opensearch_expert import opensearch_expert_agent
        self.expert = opensearch_expert_agent
    
    def build_and_execute(
        self,
        query: str,
        filters: Dict[str, Any],
        index_pattern: str = "impactu_*",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Build dynamic query and execute it.
        
        Returns:
            List of matching documents
        """
        # Build query using expert
        opensearch_query = self.expert.build_dynamic_query(
            user_query=query,
            filters=filters,
            index_pattern=index_pattern
        )
        
        # Execute
        results = self._execute(opensearch_query, limit)
        
        return results
```

**Usage by any agent:**
```python
from marie_agent.services.opensearch_tool import OpenSearchTool

tool = OpenSearchTool()
results = tool.build_and_execute(
    query="AI researchers in Colombia",
    filters={'country': 'Colombia', 'field': 'AI'}
)
```

## 📊 **PRIORIZACIÓN:**

### URGENT (Do Now - 30min):
1. ✅ Fix routing to use explicit agent_name from metadata
   - File: `core/routing.py`
   - Change: Check metadata.agent_name first

### HIGH PRIORITY (Today - 2-3h):
2. 🔧 Integrate OpenSearch Expert as tool in Retrieval
   - File: `agents/retrieval.py`
   - Change: Invoke expert, execute query, store variables

### MEDIUM (This Week - 3-4h):
3. 🔍 Add capability assessment
   - File: `orchestrator.py`
   - Change: Check indices before planning
4. 🛠️ Extract OpenSearch Expert as service
   - File: `services/opensearch_tool.py` (NEW)
   - Change: Standalone tool for all agents

## 🎯 **EXPECTED RESULTS:**

### After Phase 1 (Routing Fix):
```
Query: "Top 3 Colombian AI researchers"
Plan: 4 steps
Execution: retrieval ACTUALLY RUNS ✅
Result: Still no data (agents don't invoke tools yet)
```

### After Phase 2 (Tools Integration):
```
Query: "Top 3 Colombian AI researchers"
Plan: 4 steps
Execution: 
  E1: entity_resolution → $entities (resolved)
  E2: retrieval invokes OpenSearch Expert → $researchers (50 found)
  E3: metrics calculates from $researchers → $h_indices
  E4: reporting ranks by $h_indices → "Top 3: Dr. X, Dr. Y, Dr. Z"
Result: REAL DATA ✅
```

### After Phase 3 (Capability Assessment):
```
Query: "Quantum physics papers in Colombia"
Assessment: can_answer=False (no quantum index)
Result: "No tengo índice de quantum physics. ¿Buscas papers de física en general?"
Better UX ✅
```

## 🚀 **QUICK WIN:**

**START HERE (30 min):**

```python
# File: marie_agent/core/routing.py
# Line: ~85 (in should_continue function)

def should_continue(state: AgentState) -> AgentRoute:
    # ... existing code ...
    
    # Get current step
    step = steps[current_step]
    
    # NEW: Check for explicit agent_name in metadata
    if hasattr(step, 'metadata') and step.metadata.get('agent_name'):
        agent = step.metadata['agent_name']
        logger.info(f"Routing to explicit agent: {agent}")
        return agent
    
    # Fallback to title parsing
    step_lower = step.lower()
    if "resolve" in step_lower:
        return "entity_resolution"
    elif "retrieve" in step_lower or "search" in step_lower or "find" in step_lower:  # ADD "find"
        return "retrieval"
    # ... rest of routing ...
```

**Test:**
```bash
marie_chat
# Query: "Dame el conteo de productos para las 5 más grandes universidades"
# Expected: retrieval agent RUNS (not skipped)
```

---

**Status:** Ready to implement  
**Total Time:** 6-8 hours for all phases  
**Quick Win:** 30 minutes for Phase 1
