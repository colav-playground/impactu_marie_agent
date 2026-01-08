#!/usr/bin/env python3
"""
Test script for MARIE Agent system.

Tests various queries to verify:
- Colombian entity detection
- RAG retrieval
- Response coherence
- Agent orchestration
"""

import logging
import sys
import uuid
from marie_agent.graph import create_marie_graph
from marie_agent.state import create_initial_state
from marie_agent.config import config

# Configure logging - suppress all logs except CRITICAL
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


def test_query(graph, query: str):
    """Test a single query."""
    print(f"\n{'='*100}")
    print(f"📝 PREGUNTA: {query}")
    print(f"{'='*100}\n")
    
    try:
        # Create initial state with unique request ID
        request_id = str(uuid.uuid4())
        state = create_initial_state(query, request_id)
        
        # Run the graph (suppress output)
        result_state = graph.invoke(state)
        
        # Extract response from top-level keys
        response = result_state.get("final_answer") or result_state.get("report") or "No response generated"
        
        print(f"🤖 RESPUESTA:\n")
        print(response)
        print(f"\n{'='*100}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        print(f"{'='*100}\n")


def main():
    """Run test queries."""
    print("\n" + "🇨🇴" * 50)
    print("MARIE Agent System - Test Suite")
    print("🇨🇴" * 50)
    
    # Initialize graph
    print("\n⚙️  Inicializando MARIE agent...")
    graph = create_marie_graph()
    print("✅ Listo!\n")
    
    # Test queries - Colombian context
    colombian_queries = [
        "¿Cuántos papers tiene la Universidad de Antioquia en machine learning?",
        "¿Quiénes son los investigadores más productivos de la UNAL?",
    ]
    
    # Test queries - Generic (should not trigger RAG)
    generic_queries = [
        "¿Qué es machine learning?",
    ]
    
    # Run Colombian queries
    print("\n" + "🇨🇴" * 50)
    print("Pruebas con Contexto Colombiano")
    print("🇨🇴" * 50)
    
    for query in colombian_queries:
        test_query(graph, query)
    
    # Run generic queries
    print("\n" + "🌍" * 50)
    print("Pruebas con Preguntas Genéricas")
    print("🌍" * 50)
    
    for query in generic_queries:
        test_query(graph, query)
    
    # Final summary
    print("\n" + "="*100)
    print("✅ Pruebas completadas!")
    print("="*100 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        logger.exception("Fatal error in test suite")
        sys.exit(1)
