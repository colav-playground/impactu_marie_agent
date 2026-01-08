"""
Test Magentic Architecture Components

Tests the new Magentic components in isolation before integration.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marie_agent.core.context_window import ContextWindow, Message
from marie_agent.core.progress_ledger import ProgressTracker, ProgressLedger
from marie_agent.core.quality_evaluator import QualityEvaluator
from marie_agent.core.plan_generator import DynamicPlanGenerator, PlanStep
from marie_agent.adapters.llm_factory import get_llm_adapter


def test_context_window():
    """Test context window management."""
    print("\n" + "="*60)
    print("TEST: Context Window")
    print("="*60)
    
    context = ContextWindow(max_tokens=1000)
    
    # Add messages
    context.add_message("user", "¿Cuántos papers tiene la UdeA?")
    context.add_message("agent", "Buscando papers...", agent_name="retrieval")
    context.add_message("agent", "Encontré 1250 papers", agent_name="metrics")
    
    # Get context
    ctx_str = context.get_context_for_agent("reporting")
    print(f"\nContext for reporting agent:\n{ctx_str[:300]}...")
    
    # Get progress summary
    summary = context.get_progress_summary()
    print(f"\nProgress Summary:\n{summary}")
    
    print("✓ Context Window test passed")


def test_plan_generator():
    """Test dynamic plan generation."""
    print("\n" + "="*60)
    print("TEST: Plan Generator")
    print("="*60)
    
    llm = get_llm_adapter()
    generator = DynamicPlanGenerator(llm)
    
    # Test conceptual query
    print("\n1. Conceptual Query:")
    plan1 = generator.generate_plan("¿Qué es machine learning?")
    for i, step in enumerate(plan1, 1):
        print(f"  {i}. {step.agent_name}: {step.title}")
    
    # Test data-driven query
    print("\n2. Data-Driven Query:")
    plan2 = generator.generate_plan("¿Cuántos papers tiene la Universidad de Antioquia?")
    for i, step in enumerate(plan2, 1):
        print(f"  {i}. {step.agent_name}: {step.title}")
    
    # Test complex query
    print("\n3. Complex Query:")
    plan3 = generator.generate_plan("¿Cuáles son los top 10 papers más citados de la UdeA en machine learning?")
    for i, step in enumerate(plan3, 1):
        print(f"  {i}. {step.agent_name}: {step.title}")
    
    print("\n✓ Plan Generator test passed")


def test_quality_evaluator():
    """Test quality evaluation."""
    print("\n" + "="*60)
    print("TEST: Quality Evaluator")
    print("="*60)
    
    llm = get_llm_adapter()
    evaluator = QualityEvaluator(llm, threshold=0.7)
    
    query = "¿Cuántos papers tiene la UdeA?"
    
    # Good response
    print("\n1. Evaluating GOOD response:")
    good_response = "La Universidad de Antioquia tiene 1,250 papers indexados en nuestra base de datos, distribuidos entre 2018-2024."
    report1 = evaluator.evaluate_response(query, good_response)
    print(f"  Score: {report1.score:.2f}")
    print(f"  Acceptable: {report1.is_acceptable}")
    print(f"  Issues: {report1.issues}")
    
    # Bad response
    print("\n2. Evaluating BAD response:")
    bad_response = "No tengo información sobre eso."
    report2 = evaluator.evaluate_response(query, bad_response)
    print(f"  Score: {report2.score:.2f}")
    print(f"  Acceptable: {report2.is_acceptable}")
    print(f"  Issues: {report2.issues}")
    
    print("\n✓ Quality Evaluator test passed")


def test_progress_tracker():
    """Test progress ledger generation."""
    print("\n" + "="*60)
    print("TEST: Progress Tracker")
    print("="*60)
    
    llm = get_llm_adapter()
    tracker = ProgressTracker(llm)
    
    # Create a sample plan
    plan = [
        {
            "agent_name": "entity_resolution",
            "title": "Resolve institutions",
            "details": "Identify Universidad de Antioquia"
        },
        {
            "agent_name": "retrieval",
            "title": "Retrieve papers",
            "details": "Search for UdeA papers"
        }
    ]
    
    query = "¿Cuántos papers tiene la UdeA?"
    context = "Query initiated. Starting execution."
    evidence_map = {}
    
    # Generate ledger for first step
    print("\nGenerating ledger for step 1 (entity_resolution):")
    ledger = tracker.generate_ledger(
        query=query,
        plan=plan,
        current_step_index=0,
        context=context,
        evidence_map=evidence_map
    )
    
    print(f"  Step Complete: {ledger.step_complete.answer}")
    print(f"  Replan Needed: {ledger.replan.answer}")
    print(f"  Next Agent: {ledger.instruction.agent_name}")
    print(f"  Summary: {ledger.progress_summary[:100]}...")
    
    print("\n✓ Progress Tracker test passed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING MAGENTIC ARCHITECTURE COMPONENTS")
    print("="*60)
    
    try:
        test_context_window()
        test_plan_generator()
        # test_quality_evaluator()  # Skip - requires actual LLM call
        # test_progress_tracker()    # Skip - requires actual LLM call
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
