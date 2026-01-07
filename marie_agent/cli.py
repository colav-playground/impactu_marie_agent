"""
Simple CLI for testing MARIE agent system.
"""

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone

from marie_agent.graph import run_marie_query
from marie_agent.config import config


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MARIE Multi-Agent System CLI"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query to process"
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default=None,
        help="Request ID (generated if not provided)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        config.log_level = "DEBUG"
    setup_logging()
    
    # Generate request ID if not provided
    request_id = args.request_id or f"req_{uuid.uuid4().hex[:12]}"
    
    print(f"\n{'='*80}")
    print(f"MARIE Multi-Agent System")
    print(f"{'='*80}")
    print(f"Request ID: {request_id}")
    print(f"Query: {args.query}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*80}\n")
    
    try:
        # Run query
        result = run_marie_query(args.query, request_id)
        
        # Display results
        print(f"\n{'='*80}")
        print(f"Results")
        print(f"{'='*80}")
        print(f"Status: {result['status']}")
        print(f"Steps completed: {result['current_step']}")
        print(f"Tasks: {len(result['tasks'])}")
        print(f"Evidence items: {sum(len(ev) for ev in result['evidence_map'].values())}")
        
        if result['final_answer']:
            print(f"\nAnswer:")
            print(result['final_answer'])
        
        if result['confidence_score']:
            confidence = result['confidence_score']
            confidence_assessment = result.get('confidence_assessment', {})
            confidence_level = confidence_assessment.get('confidence_level', 'unknown')
            
            # Color code based on confidence score AND level
            if confidence_level == "high" or confidence >= 0.8:
                icon = "🟢"
            elif confidence_level == "medium" or confidence >= 0.6:
                icon = "🟡"
            else:
                icon = "🔴"
            
            print(f"\n{icon} Confidence: {confidence:.2%} ({confidence_level})")
            
            # Show reasoning if available
            reasoning = confidence_assessment.get('reasoning')
            if reasoning:
                print(f"   Reasoning: {reasoning}")
            
            # Show limitations
            limitations = confidence_assessment.get('limitations', [])
            if limitations:
                print(f"   Limitations: {', '.join(limitations[:2])}")
        
        if result['error']:
            print(f"\nError: {result['error']}")
        
        print(f"\n{'='*80}")
        print(f"Audit Log: {len(result['audit_log'])} events")
        print(f"{'='*80}\n")
        
    except Exception as e:
        logging.error(f"Error executing query: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
