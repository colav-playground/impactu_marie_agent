#!/usr/bin/env python3
"""
Test script for tools and retry logic.
"""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_search_tool():
    """Test search_publications tool."""
    print("\n" + "="*80)
    print("TEST 1: search_publications tool")
    print("="*80)
    
    try:
        from marie_agent.tools import search_publications
        
        print("\n✓ Tool imported successfully")
        print(f"Tool name: {search_publications.name}")
        print(f"Tool description: {search_publications.description}")
        
        # Test invocation
        print("\n→ Invoking tool with query='machine learning'...")
        result = search_publications.invoke({
            "query": "machine learning",
            "limit": 5
        })
        
        formatted, documents = result
        print(f"\n✓ Tool returned {len(documents)} documents")
        print(f"Formatted output:\n{formatted[:200]}...")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_resolve_tool():
    """Test resolve_entity tool."""
    print("\n" + "="*80)
    print("TEST 2: resolve_entity tool")
    print("="*80)
    
    try:
        from marie_agent.tools import resolve_entity
        
        print("\n✓ Tool imported successfully")
        print(f"Tool name: {resolve_entity.name}")
        
        # Test invocation
        print("\n→ Invoking tool with name='Universidad de Antioquia'...")
        result = resolve_entity.invoke({
            "name": "Universidad de Antioquia",
            "entity_type": "institution"
        })
        
        formatted, metadata = result
        print(f"\n✓ Tool executed")
        print(f"Formatted output:\n{formatted}")
        print(f"Metadata: {metadata}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retry_logic():
    """Test retry logic in retrieval agent."""
    print("\n" + "="*80)
    print("TEST 3: Retry logic in retrieval agent")
    print("="*80)
    
    try:
        from marie_agent.agents.retrieval import RetrievalAgent
        
        agent = RetrievalAgent()
        print("\n✓ RetrievalAgent created")
        
        # Test search with retry
        print("\n→ Testing _search_opensearch with retry logic...")
        results = agent._search_opensearch("test query", limit=3)
        
        print(f"\n✓ Search completed, returned {len(results)} results")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_logging():
    """Test improved memory logging."""
    print("\n" + "="*80)
    print("TEST 4: Memory logging improvements")
    print("="*80)
    
    try:
        from marie_agent.memory import ConversationSession, ConversationTurn
        
        session = ConversationSession("test_session", "test_user")
        print("\n✓ Session created")
        
        # Add turn with logging
        turn = ConversationTurn(
            turn_id="turn_1",
            user_query="Test query",
            agent_response="Test response",
            confidence=0.85,
            evidence_count=5
        )
        
        print("\n→ Adding turn to session (should show enhanced logging)...")
        session.add_turn(turn)
        
        print(f"\n✓ Turn added successfully")
        print(f"Session has {len(session.turns)} turns")
        
        # Get recent context
        print("\n→ Getting recent context...")
        recent = session.get_recent_context(n=2)
        print(f"✓ Retrieved {len(recent)} recent turns")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MARIE TOOLS & IMPROVEMENTS TEST SUITE")
    print("="*80)
    
    results = {
        "search_tool": test_search_tool(),
        "resolve_tool": test_resolve_tool(),
        "retry_logic": test_retry_logic(),
        "memory_logging": test_memory_logging()
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n{passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
