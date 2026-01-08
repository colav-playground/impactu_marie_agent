"""
Demo Simple - Muestra planes y respuestas simuladas

NO ejecuta el sistema completo, solo muestra cómo funciona.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Colors
class C:
    B = '\033[94m'  # Blue
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    R = '\033[91m'  # Red
    BOLD = '\033[1m'
    END = '\033[0m'


def show_query(num, query, query_type):
    """Show query being processed."""
    print(f"\n{C.BOLD}{C.B}{'═'*80}{C.END}")
    print(f"{C.BOLD}{C.B}EJEMPLO {num}: {query_type}{C.END}")
    print(f"{C.BOLD}{C.B}{'═'*80}{C.END}\n")
    print(f"{C.BOLD}📋 Query:{C.END} {C.Y}{query}{C.END}\n")


def show_plan(plan_type, agents, needs_rag):
    """Show generated plan."""
    print(f"{C.G}✓ Plan generado:{C.END}")
    print(f"  • Tipo: {C.BOLD}{plan_type}{C.END}")
    print(f"  • Necesita RAG: {C.BOLD}{needs_rag}{C.END}")
    print(f"  • Agentes: {C.BOLD}{len(agents)}{C.END}\n")
    
    print(f"{C.BOLD}{C.G}📝 PLAN:{C.END}")
    for i, agent in enumerate(agents, 1):
        print(f"  [{i}] {C.Y}{agent}{C.END}")
    print()


def show_response(response, quality_score):
    """Show generated response with quality."""
    print(f"{C.BOLD}{C.G}✨ RESPUESTA:{C.END}")
    print(f"{C.Y}{response}{C.END}\n")
    
    color = C.G if quality_score >= 0.7 else C.R
    status = "✅ ACEPTABLE" if quality_score >= 0.7 else "❌ NECESITA REFINAMIENTO"
    
    print(f"{C.BOLD}📊 CALIDAD:{C.END}")
    print(f"  Score: {color}{quality_score:.2f}/1.0{C.END}")
    print(f"  Estado: {color}{status}{C.END}\n")
    
    # Show dimensions
    print(f"  Dimensiones:")
    dimensions = {
        "Relevancia": 0.95 if quality_score >= 0.7 else 0.6,
        "Completitud": 0.90 if quality_score >= 0.7 else 0.5,
        "Precisión": 0.92 if quality_score >= 0.7 else 0.7,
        "Fundamentación": 0.88 if quality_score >= 0.7 else 0.6
    }
    
    for dim, score in dimensions.items():
        dim_color = C.G if score >= 0.7 else C.Y if score >= 0.5 else C.R
        print(f"    • {dim}: {dim_color}{score:.2f}{C.END}")
    print()


def demo():
    """Run demo with simulated responses."""
    print(f"\n{C.BOLD}{'='*80}{C.END}")
    print(f"{C.BOLD}🎭 MAGENTIC ARCHITECTURE - DEMO CON EVALUACIÓN DE CALIDAD{C.END}")
    print(f"{C.BOLD}{'='*80}{C.END}\n")
    
    print(f"{C.BOLD}Este demo muestra:{C.END}")
    print(f"  • Planes dinámicos según tipo de query")
    print(f"  • Respuestas generadas")
    print(f"  • Evaluación de calidad en 4 dimensiones")
    print(f"  • Refinamiento automático si calidad < 0.7\n")
    
    # === QUERY 1: CONCEPTUAL ===
    show_query(1, "¿Qué es machine learning?", "CONCEPTUAL")
    show_plan("conceptual", ["reporting"], needs_rag=False)
    
    response1 = """Machine Learning es una rama de la inteligencia artificial que permite a los 
sistemas aprender y mejorar automáticamente a partir de la experiencia sin ser programados 
explícitamente. Se basa en algoritmos que pueden identificar patrones en datos y hacer 
predicciones o decisiones basadas en esos patrones."""
    
    show_response(response1, quality_score=0.92)
    
    print(f"{C.G}💾 Plan guardado en memoria OpenSearch{C.END}")
    print(f"{C.G}✓ Disponible para reuso en queries similares{C.END}\n")
    
    # === QUERY 2: DATA-DRIVEN ===
    show_query(2, "¿Cuántos papers tiene la Universidad de Antioquia?", "DATA-DRIVEN")
    show_plan("data_driven", 
              ["entity_resolution", "retrieval", "metrics", "reporting"],
              needs_rag=True)
    
    response2 = """Según los datos indexados en OpenSearch, la Universidad de Antioquia tiene 
un total de 12,847 papers registrados. La distribución por año muestra:
- 2023: 1,234 papers
- 2022: 1,456 papers  
- 2021: 1,123 papers
Las áreas más productivas son Medicina (23%), Ingeniería (18%) y Ciencias Sociales (15%)."""
    
    show_response(response2, quality_score=0.95)
    
    print(f"{C.G}💾 Plan guardado en memoria (usage_count: 1){C.END}\n")
    
    # === QUERY 3: COMPLEJO ===
    show_query(3, "Dame los top 5 papers de Colombia sobre IA con citaciones", "COMPLEJO")
    show_plan("complex",
              ["retrieval", "metrics", "citations", "reporting"],
              needs_rag=True)
    
    # Simular calidad baja primero
    response3_v1 = "Aquí están algunos papers sobre IA en Colombia..."
    
    print(f"{C.BOLD}{C.G}✨ RESPUESTA (Intento 1):{C.END}")
    print(f"{C.Y}{response3_v1}{C.END}\n")
    
    show_response(response3_v1, quality_score=0.58)
    
    print(f"{C.BOLD}{C.R}🔄 REFINANDO PLAN (score < 0.7)...{C.END}\n")
    
    # Plan refinado
    show_plan("complex_refined",
              ["retrieval", "metrics", "citations", "reporting"],
              needs_rag=True)
    
    response3_v2 = """Top 5 papers de Colombia sobre Inteligencia Artificial:

1. "Deep Learning for Medical Image Analysis" (2023)
   Autores: García, J., Martínez, A.
   Citaciones: 247
   DOI: 10.1234/example.2023.001

2. "Natural Language Processing in Spanish" (2022)
   Autores: Rodríguez, M., López, C.
   Citaciones: 189
   DOI: 10.1234/example.2022.045

3. "Computer Vision for Agriculture" (2023)
   Autores: Pérez, L., González, R.
   Citaciones: 156
   DOI: 10.1234/example.2023.078

4. "Reinforcement Learning Applications" (2022)
   Autores: Torres, S., Ramírez, D.
   Citaciones: 134
   DOI: 10.1234/example.2022.092

5. "AI Ethics and Governance" (2023)
   Autores: Hernández, P., Castro, F.
   Citaciones: 98
   DOI: 10.1234/example.2023.115"""
    
    print(f"{C.BOLD}{C.G}✨ RESPUESTA (Intento 2 - Refinada):{C.END}")
    print(f"{C.Y}{response3_v2}{C.END}\n")
    
    show_response(response3_v2, quality_score=0.91)
    
    print(f"{C.G}💾 Plan refinado guardado en memoria{C.END}\n")
    
    # === SUMMARY ===
    print(f"\n{C.BOLD}{'='*80}{C.END}")
    print(f"{C.BOLD}📈 RESUMEN{C.END}")
    print(f"{C.BOLD}{'='*80}{C.END}\n")
    
    features = [
        ("✅ Planes dinámicos", "Query conceptual: 1 agente | Data-driven: 4 agentes"),
        ("✅ RAG automático", "Detecta cuándo necesita buscar en OpenSearch"),
        ("✅ Evaluación 4D", "Relevancia, completitud, precisión, fundamentación"),
        ("✅ Refinamiento", "Si score < 0.7 → refina plan y re-ejecuta"),
        ("✅ Memoria semántica", "Planes guardados en OpenSearch con embeddings"),
        ("✅ Auto-corrección", "Máximo 2 iteraciones de refinamiento"),
    ]
    
    for feature, desc in features:
        print(f"  {C.G}{feature:25}{C.END} {desc}")
    
    print(f"\n{C.BOLD}{C.G}🎉 Arquitectura Magentic funcionando con calidad garantizada{C.END}\n")


if __name__ == "__main__":
    demo()
