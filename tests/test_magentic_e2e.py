"""
End-to-End Test for Magentic Architecture

Tests the complete Magentic orchestration flow.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marie_agent.state import create_initial_state
from marie_agent.orchestrator import orchestrator_node
import uuid


def test_conceptual_query():
    """Test with conceptual query (but may need RAG for citations)."""
    print("\n" + "="*70)
    print("TEST: Conceptual Query")
    print("="*70)
    
    query = "¿Qué es machine learning?"
    request_id = str(uuid.uuid4())
    
    # Create state
    state = create_initial_state(query, request_id)
    
    print(f"\n📝 Query: {query}")
    print(f"Initial Status: {state['status']}")
    
    # Run orchestrator (Planning Mode)
    print("\n→ Running orchestrator...")
    state = orchestrator_node(state)
    
    print(f"\n✓ Plan generated:")
    plan = state.get('plan', {})
    print(f"  Mode: {plan.get('mode')}")
    print(f"  Needs RAG: {state.get('needs_rag')}")
    print(f"  Steps: {len(plan.get('steps', []))}")
    for i, step in enumerate(plan.get('steps', []), 1):
        print(f"    {i}. {step}")
    print(f"  Next Agent: {state.get('next_agent')}")
    print(f"  Status: {state['status']}")
    
    # Verify - conceptual queries get retrieval for citations
    assert state['status'] == 'executing'
    assert state.get('next_agent') in ['retrieval', 'reporting']
    
    print("\n✅ Conceptual query test PASSED")
    return state


def test_data_driven_query():
    """Test with data-driven query (RAG needed)."""
    print("\n" + "="*70)
    print("TEST: Data-Driven Query (RAG Pipeline)")
    print("="*70)
    
    query = "¿Cuántos papers tiene la Universidad de Antioquia?"
    request_id = str(uuid.uuid4())
    
    # Create state
    state = create_initial_state(query, request_id)
    
    print(f"\n📝 Query: {query}")
    print(f"Initial Status: {state['status']}")
    
    # Run orchestrator (Planning Mode)
    print("\n→ Running orchestrator...")
    state = orchestrator_node(state)
    
    print(f"\n✓ Plan generated:")
    plan = state.get('plan', {})
    print(f"  Mode: {plan.get('mode')}")
    print(f"  Needs RAG: {state.get('needs_rag')}")
    print(f"  Steps: {len(plan.get('steps', []))}")
    for i, step in enumerate(plan.get('steps', []), 1):
        agent = plan.get('agents_required', [])[i-1] if i-1 < len(plan.get('agents_required', [])) else 'unknown'
        print(f"    {i}. [{agent}] {step}")
    print(f"  Next Agent: {state.get('next_agent')}")
    print(f"  Status: {state['status']}")
    
    # Verify
    assert state['status'] == 'executing'
    assert state.get('needs_rag') == True
    assert 'entity_resolution' in plan.get('agents_required', []) or 'retrieval' in plan.get('agents_required', [])
    
    print("\n✅ Data-driven query test PASSED")
    return state


def test_complex_query():
    """Test with complex query (multiple agents)."""
    print("\n" + "="*70)
    print("TEST: Complex Query (Multi-Agent)")
    print("="*70)
    
    query = "¿Cuáles son los top 10 papers más citados de la UdeA en machine learning?"
    request_id = str(uuid.uuid4())
    
    # Create state
    state = create_initial_state(query, request_id)
    
    print(f"\n📝 Query: {query}")
    print(f"Initial Status: {state['status']}")
    
    # Run orchestrator (Planning Mode)
    print("\n→ Running orchestrator...")
    state = orchestrator_node(state)
    
    print(f"\n✓ Plan generated:")
    plan = state.get('plan', {})
    print(f"  Mode: {plan.get('mode')}")
    print(f"  Needs RAG: {state.get('needs_rag')}")
    print(f"  Steps: {len(plan.get('steps', []))}")
    for i, step in enumerate(plan.get('steps', []), 1):
        agent = plan.get('agents_required', [])[i-1] if i-1 < len(plan.get('agents_required', [])) else 'unknown'
        print(f"    {i}. [{agent}] {step}")
    print(f"  Next Agent: {state.get('next_agent')}")
    print(f"  Status: {state['status']}")
    
    # Verify
    assert state['status'] == 'executing'
    assert state.get('needs_rag') == True
    agents = plan.get('agents_required', [])
    assert 'entity_resolution' in agents or 'retrieval' in agents
    assert 'metrics' in agents or 'reporting' in agents
    
    print("\n✅ Complex query test PASSED")
    return state


def test_plan_structure():
    """Verify plan structure has all required fields."""
    print("\n" + "="*70)
    print("TEST: Plan Structure Validation")
    print("="*70)
    
    query = "¿Cuántos papers tiene la UdeA?"
    state = create_initial_state(query, str(uuid.uuid4()))
    
    # Generate plan
    state = orchestrator_node(state)
    
    plan = state.get('plan', {})
    
    # Check required fields
    required_fields = ['mode', 'steps', 'agents_required', 'plan_steps']
    missing = [f for f in required_fields if f not in plan]
    
    if missing:
        print(f"❌ Missing fields: {missing}")
        assert False, f"Plan missing required fields: {missing}"
    
    # Check plan_steps structure
    plan_steps = plan.get('plan_steps', [])
    if plan_steps:
        step = plan_steps[0]
        required_step_fields = ['agent_name', 'title', 'details']
        missing_step = [f for f in required_step_fields if f not in step]
        
        if missing_step:
            print(f"❌ Plan step missing fields: {missing_step}")
            assert False, f"Plan step missing: {missing_step}"
    
    print(f"\n✓ Plan structure valid:")
    print(f"  ✓ mode: {plan['mode']}")
    print(f"  ✓ steps: {len(plan['steps'])} steps")
    print(f"  ✓ agents_required: {len(plan['agents_required'])} agents")
    print(f"  ✓ plan_steps: {len(plan_steps)} detailed steps")
    
    print("\n✅ Plan structure test PASSED")


def main():
    """Run all end-to-end tests."""
    print("\n" + "="*70)
    print("END-TO-END TESTING: MAGENTIC ARCHITECTURE")
    print("="*70)
    
    try:
        # Test different query types
        test_conceptual_query()
        test_data_driven_query()
        test_complex_query()
        test_plan_structure()
        
        print("\n" + "="*70)
        print("ALL END-TO-END TESTS PASSED ✅")
        print("="*70)
        print("\n🎉 Magentic architecture is working correctly!")
        print("   - Dynamic plan generation ✓")
        print("   - RAG detection ✓")
        print("   - Agent routing ✓")
        print("   - State management ✓")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
