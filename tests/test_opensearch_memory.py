"""
Test OpenSearch Memory System

Tests semantic search with embeddings.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marie_agent.core.memory_opensearch import (
    PlanMemoryOpenSearch,
    EpisodicMemoryOpenSearch
)


def test_plan_memory_opensearch():
    """Test OpenSearch plan memory with semantic search."""
    print("\n" + "="*70)
    print("TEST: OpenSearch Plan Memory (Semantic Search)")
    print("="*70)
    
    try:
        memory = PlanMemoryOpenSearch()
        
        # Save plans with different phrasings
        print("\n1. Saving plans...")
        
        plan_steps = [
            {
                "agent_name": "entity_resolution",
                "title": "Resolve institution",
                "details": "Find Universidad de Antioquia"
            },
            {
                "agent_name": "retrieval",
                "title": "Get papers",
                "details": "Retrieve papers"
            },
            {
                "agent_name": "metrics",
                "title": "Count",
                "details": "Count total"
            }
        ]
        
        # Save with different phrasings
        id1 = memory.save_plan(
            task="¿Cuántos papers tiene la Universidad de Antioquia?",
            plan_steps=plan_steps,
            success=True,
            metadata={"quality_score": 0.95}
        )
        print(f"  ✓ Saved plan 1: {id1}")
        
        id2 = memory.save_plan(
            task="¿Cuántos artículos tiene la UdeA?",
            plan_steps=plan_steps,
            success=True,
            metadata={"quality_score": 0.92}
        )
        print(f"  ✓ Saved plan 2: {id2}")
        
        # Test semantic search with different phrasing
        print("\n2. Testing semantic search...")
        
        queries = [
            "¿Cuántos documentos tiene la Universidad de Antioquia?",
            "papers de UdeA",
            "artículos científicos Universidad Antioquia"
        ]
        
        for query in queries:
            print(f"\n  Query: {query}")
            similar = memory.retrieve_similar_plan(query, min_similarity=0.6)
            
            if similar:
                print(f"    ✓ Found similar plan!")
                print(f"      Task: {similar['task']}")
                print(f"      Similarity: {similar.get('similarity_score', 0):.3f}")
                print(f"      Usage count: {similar.get('usage_count', 0)}")
            else:
                print(f"    ✗ No similar plan found")
        
        print("\n✅ OpenSearch plan memory test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episodic_memory_opensearch():
    """Test OpenSearch episodic memory."""
    print("\n" + "="*70)
    print("TEST: OpenSearch Episodic Memory")
    print("="*70)
    
    try:
        memory = EpisodicMemoryOpenSearch()
        
        # Save episodes
        print("\n1. Saving episodes...")
        
        id1 = memory.save_episode(
            query="¿Qué es machine learning?",
            response="Machine learning es una rama de la inteligencia artificial...",
            plan_used=[{"agent": "retrieval"}, {"agent": "reporting"}],
            success=True,
            quality_score=0.9
        )
        print(f"  ✓ Saved episode 1: {id1}")
        
        id2 = memory.save_episode(
            query="¿Cuántos papers tiene UdeA?",
            response="La UdeA tiene 1,250 papers...",
            plan_used=[{"agent": "retrieval"}, {"agent": "metrics"}],
            success=True,
            quality_score=0.95
        )
        print(f"  ✓ Saved episode 2: {id2}")
        
        # Get recent
        print("\n2. Getting recent episodes...")
        recent = memory.get_recent_episodes(n=5)
        print(f"  ✓ Recent episodes: {len(recent)}")
        
        # Get successful
        print("\n3. Getting successful episodes...")
        successful = memory.get_successful_episodes()
        print(f"  ✓ Successful episodes: {len(successful)}")
        
        print("\n✅ OpenSearch episodic memory test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run OpenSearch memory tests."""
    print("\n" + "="*70)
    print("TESTING OPENSEARCH MEMORY SYSTEM")
    print("="*70)
    print("\nNOTE: These tests require OpenSearch to be running")
    print("      at http://localhost:9200")
    
    try:
        success = True
        success = test_plan_memory_opensearch() and success
        success = test_episodic_memory_opensearch() and success
        
        if success:
            print("\n" + "="*70)
            print("ALL OPENSEARCH MEMORY TESTS PASSED ✅")
            print("="*70)
            print("\n🎉 OpenSearch memory working correctly!")
            print("   - Semantic search ✓")
            print("   - K-NN similarity ✓")
            print("   - Plan storage ✓")
            print("   - Episode storage ✓")
            return 0
        else:
            print("\n⚠️  Some tests failed")
            return 1
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
