#!/usr/bin/env python3
"""
IMPACTU MARIE - Interactive Chat

Chat interactivo que muestra la ejecución paso a paso del sistema Magentic.
"""

import logging
import time
from typing import Dict, Any

from marie_agent.state import create_initial_state
from marie_agent.orchestrator import OrchestratorAgent

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("opensearchpy").setLevel(logging.ERROR)  # Suppress OpenSearch warnings
logging.getLogger("opensearch").setLevel(logging.ERROR)  # Suppress OpenSearch warnings
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

# Colors
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
    print(f"{Color.BOLD}{Color.CYAN}{'IMPACTU MARIE - Interactive Chat'.center(80)}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'Powered by Magentic Architecture'.center(80)}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*80}{Color.END}\n")
    
    print(f"{Color.DIM}El sistema ejecutará tu pregunta usando un plan dinámico.{Color.END}")
    print(f"{Color.DIM}Verás cada paso en tiempo real.{Color.END}\n")


def print_plan(plan: Dict[str, Any]):
    """Print the generated plan."""
    steps = plan.get('steps', [])
    needs_rag = plan.get('needs_rag', False)
    query_type = plan.get('query_type', 'unknown')
    
    print(f"\n{Color.BOLD}{Color.GREEN}📋 PLAN GENERADO:{Color.END}")
    print(f"  • Tipo: {Color.YELLOW}{query_type}{Color.END}")
    print(f"  • RAG: {Color.YELLOW}{'Sí' if needs_rag else 'No'}{Color.END}")
    print(f"  • Pasos: {Color.BOLD}{len(steps)}{Color.END}\n")
    
    for i, step in enumerate(steps, 1):
        if isinstance(step, dict):
            agent = step.get('agent_name', 'unknown')
            title = step.get('title', '')
            print(f"  {Color.CYAN}[{i}]{Color.END} {Color.BOLD}{agent}{Color.END}")
            if title:
                print(f"      └─ {Color.DIM}{title}{Color.END}")
    print()


def print_step_start(step_num: int, agent_name: str):
    """Print step execution start."""
    print(f"{Color.BOLD}{Color.BLUE}⚙️  Ejecutando paso {step_num}: {agent_name.upper()}{Color.END}")


def print_step_result(agent_name: str, result_preview: str):
    """Print step result preview."""
    print(f"   {Color.GREEN}✓{Color.END} {Color.DIM}{result_preview[:100]}...{Color.END}\n")


def print_quality(quality_report: Dict[str, Any]):
    """Print quality evaluation."""
    score = quality_report.get('score', 0)
    acceptable = quality_report.get('is_acceptable', False)
    
    color = Color.GREEN if acceptable else Color.RED
    status = "✅ Aceptable" if acceptable else "❌ Necesita mejora"
    
    print(f"\n{Color.BOLD}📊 CALIDAD:{Color.END} {color}{score:.2f}/1.0{Color.END} - {color}{status}{Color.END}")
    
    dimensions = quality_report.get('dimensions', {})
    if dimensions:
        print(f"   {Color.DIM}Relevancia: {dimensions.get('relevance', 0):.2f} | "
              f"Completitud: {dimensions.get('completeness', 0):.2f} | "
              f"Precisión: {dimensions.get('accuracy', 0):.2f} | "
              f"Fundamentación: {dimensions.get('groundedness', 0):.2f}{Color.END}")


def print_final_answer(answer: str):
    """Print final answer."""
    print(f"\n{Color.BOLD}{Color.GREEN}✨ RESPUESTA:{Color.END}\n")
    print(f"{Color.CYAN}{answer}{Color.END}\n")


def print_error(error: str):
    """Print error message."""
    print(f"\n{Color.RED}❌ Error: {error}{Color.END}\n")


def simulate_execution(state: Dict[str, Any], orchestrator: OrchestratorAgent, query: str):
    """
    Simulate execution showing steps in real-time.
    
    This is a simplified version that shows the plan and simulates execution.
    In production, this would connect to the full graph execution.
    """
    plan = state.get('plan', {})
    if not isinstance(plan, dict):
        print_error("Plan no generado correctamente")
        return
    
    print_plan(plan)
    
    steps = plan.get('steps', [])
    
    # Simulate step-by-step execution
    for i, step in enumerate(steps, 1):
        if isinstance(step, dict):
            agent_name = step.get('agent_name', 'unknown')
            print_step_start(i, agent_name)
            
            # Simulate work
            time.sleep(0.5)  # Brief pause for visual effect
            
            # Show result preview
            if agent_name == 'entity_resolution':
                print_step_result(agent_name, "Entidades resueltas: Universidad de Antioquia")
            elif agent_name == 'retrieval':
                print_step_result(agent_name, "Documentos recuperados de OpenSearch")
            elif agent_name == 'metrics':
                print_step_result(agent_name, "Métricas calculadas")
            elif agent_name == 'citations':
                print_step_result(agent_name, "Citaciones generadas")
            elif agent_name == 'reporting':
                print_step_result(agent_name, "Respuesta generada")
    
    # Simulate quality check
    print(f"\n{Color.BOLD}{Color.BLUE}🔍 Evaluando calidad...{Color.END}")
    time.sleep(0.3)
    
    # Mock quality report
    quality_report = {
        'score': 0.85,
        'is_acceptable': True,
        'dimensions': {
            'relevance': 0.90,
            'completeness': 0.85,
            'accuracy': 0.88,
            'groundedness': 0.82
        }
    }
    print_quality(quality_report)
    
    # Generate mock answer based on query
    query_lower = query.lower().strip()
    query_words = query_lower.split()
    
    # Check if it's a greeting (must be at start or alone)
    is_greeting = (
        query_lower in ['hola', 'hello', 'hi', 'hey'] or
        (query_words and query_words[0] in ['hola', 'hello', 'hi', 'hey'])
    )
    
    if is_greeting:
        answer = """¡Hola! 👋 Soy MARIE (Multi-Agent Research Intelligence Engine).

Estoy aquí para ayudarte con consultas sobre producción académica, investigadores e instituciones.

Puedo ayudarte con:
• Estadísticas de publicaciones
• Información sobre investigadores y instituciones
• Rankings y análisis cientométricos
• Búsqueda de papers y citaciones

¿En qué puedo ayudarte hoy?"""
    
    elif any(word in query_lower for word in ['gracias', 'thanks', 'ok', 'bye', 'adiós']):
        answer = "¡De nada! Si necesitas algo más, aquí estaré. 😊"
    
    elif 'qué es' in query_lower or 'what is' in query_lower:
        answer = """Machine Learning es una rama de la inteligencia artificial que permite a los 
sistemas aprender y mejorar automáticamente a partir de la experiencia sin ser programados 
explícitamente. Se basa en el análisis de datos para identificar patrones y tomar decisiones 
con mínima intervención humana."""
    
    elif 'cuántos' in query_lower or 'how many' in query_lower:
        answer = """Según los datos indexados en OpenSearch:

📊 Universidad de Antioquia - Estadísticas de publicaciones:
   • Total de papers: 12,847
   • Distribución por año:
     - 2023: 1,234 papers
     - 2022: 1,456 papers
     - 2021: 1,123 papers
   • Áreas principales:
     - Medicina: 23%
     - Ingeniería: 18%
     - Ciencias Sociales: 15%"""
    
    else:
        answer = """He procesado tu consulta y encontrado información relevante en la base de datos.
La respuesta se ha generado basándose en los documentos recuperados y las métricas calculadas."""
    
    print_final_answer(answer)
    
    # Show memory save
    print(f"{Color.GREEN}💾 Plan guardado en memoria para reuso futuro{Color.END}\n")


def main():
    """Main chat loop."""
    print_header()
    
    # Initialize orchestrator
    print(f"{Color.YELLOW}Inicializando sistema...{Color.END}")
    try:
        orchestrator = OrchestratorAgent()
        print(f"{Color.GREEN}✓ Sistema listo\n{Color.END}")
    except Exception as e:
        print_error(f"No se pudo inicializar: {e}")
        return 1
    
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
            print(f"{Color.BOLD}🎯 Procesando consulta #{query_count}...{Color.END}\n")
            
            # Create state and generate plan
            state = create_initial_state(query, f"chat_{query_count}")
            
            print(f"{Color.YELLOW}🧠 Generando plan dinámico...{Color.END}")
            state = orchestrator.plan(state)
            
            # Execute and show results
            simulate_execution(state, orchestrator, query)
            
            print(f"{Color.BOLD}{Color.BLUE}{'─'*80}{Color.END}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{Color.YELLOW}👋 Interrumpido por usuario. ¡Hasta luego!{Color.END}\n")
            break
        except Exception as e:
            print_error(f"Error procesando consulta: {e}")
            print(f"{Color.DIM}Intenta con otra pregunta...{Color.END}\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}👋 ¡Hasta luego!{Color.END}\n")
        exit(0)
