#!/usr/bin/env python3
"""
IMPACTU MARIE - Real System Execution

Executes the full multi-agent graph (not a demo).
Shows real agent execution with actual LLM calls and processing.
"""

import logging
import time
from typing import Dict, Any

from marie_agent.state import create_initial_state
from marie_agent.graph import create_marie_graph

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("opensearchpy").setLevel(logging.ERROR)
logging.getLogger("opensearch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


# ANSI color codes
class Color:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_header():
    """Print welcome header."""
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'IMPACTU MARIE - Real Multi-Agent System':^80}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'Full Graph Execution':^80}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}\n")
    print(f"{Color.DIM}This executes the complete agent graph with real processing.{Color.END}")
    print(f"{Color.DIM}Agents will analyze, retrieve, compute, and generate responses.{Color.END}\n")


def print_final_answer(answer: str):
    """Print final answer."""
    print(f"\n{Color.BOLD}{Color.GREEN}✨ RESPUESTA FINAL:{Color.END}\n")
    print(f"{Color.CYAN}{answer}{Color.END}\n")


def print_error(error: str):
    """Print error message."""
    print(f"\n{Color.RED}❌ Error: {error}{Color.END}\n")


def main():
    """Main execution loop."""
    print_header()
    
    print(f"{Color.YELLOW}⚙️  Inicializando sistema completo...{Color.END}")
    
    try:
        # Create the full graph
        graph = create_marie_graph()
        logger.info("✓ Grafo multi-agente creado")
        
    except Exception as e:
        print_error(f"Error inicializando sistema: {e}")
        return 1
    
    print(f"{Color.GREEN}✓ Sistema listo{Color.END}\n")
    
    query_count = 0
    
    # Main loop
    while True:
        try:
            # Get user input
            print(f"{Color.BOLD}{Color.MAGENTA}💬 Tu pregunta{Color.END} {Color.DIM}(o 'salir' para terminar):{Color.END}")
            query = input(f"{Color.CYAN}➜ {Color.END}").strip()
            
            if not query:
                continue
            
            if query.lower() in ['salir', 'exit', 'quit', 'q']:
                print(f"\n{Color.YELLOW}👋 ¡Hasta luego!{Color.END}\n")
                break
            
            query_count += 1
            
            print(f"\n{Color.BOLD}{Color.BLUE}{'─'*80}{Color.END}")
            print(f"{Color.BOLD}🎯 Ejecutando consulta #{query_count}...{Color.END}\n")
            
            # Create initial state
            state = create_initial_state(query, f"query_{query_count}")
            
            # Execute the full graph
            print(f"{Color.YELLOW}🔄 Ejecutando grafo multi-agente...{Color.END}\n")
            start_time = time.time()
            
            # Stream execution (shows progress)
            for step_output in graph.stream(state):
                agent = list(step_output.keys())[0]
                logger.info(f"✓ Ejecutado: {agent}")
            
            # Get final state
            final_state = step_output[agent]
            elapsed = time.time() - start_time
            
            print(f"\n{Color.GREEN}✓ Procesamiento completado en {elapsed:.2f}s{Color.END}")
            
            # Extract and display answer
            answer = final_state.get("final_answer", "")
            report = final_state.get("report", {})
            
            if answer:
                print_final_answer(answer)
            elif isinstance(report, dict) and report.get("answer"):
                print_final_answer(report["answer"])
            else:
                print_final_answer("Sistema procesó la consulta pero no generó respuesta visible.")
            
            # Show some stats
            plan = final_state.get("plan", {})
            steps = plan.get("steps", [])
            if steps:
                print(f"{Color.DIM}📊 Ejecutados {len(steps)} pasos del plan{Color.END}")
            
            print(f"{Color.BOLD}{Color.BLUE}{'─'*80}{Color.END}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{Color.YELLOW}👋 Interrumpido por usuario. ¡Hasta luego!{Color.END}\n")
            break
        except Exception as e:
            logger.exception("Error procesando consulta")
            print_error(f"Error procesando consulta: {e}")
            print(f"{Color.DIM}Intenta con otra pregunta...{Color.END}\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}👋 ¡Hasta luego!{Color.END}\n")
        exit(0)
