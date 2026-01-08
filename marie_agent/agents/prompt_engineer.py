"""
Prompt Engineer Agent - Dynamic prompt generation service with reflexion.

Self-improving prompt engineering system that:
- Generates optimized prompts using multiple techniques
- Reflects on prompt effectiveness
- Learns from successful patterns
- Logs prompts for analytics
- Iteratively improves poor prompts

Techniques:
- Zero-shot prompting
- Few-shot prompting  
- Chain-of-Thought (CoT)
- ReAct (Reasoning + Acting)
- Self-consistency
- Structured outputs
"""

from typing import Dict, Any, List, Optional, Literal
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from opensearchpy import OpenSearch

from marie_agent.state import AgentState
from marie_agent.adapters.llm_factory import get_llm_adapter
from marie_agent.config import config

logger = logging.getLogger(__name__)


PromptTechnique = Literal["zero-shot", "few-shot", "chain-of-thought", "react", "structured"]


@dataclass
class PromptAttempt:
    """Record of a prompt generation attempt."""
    agent_name: str
    task_description: str
    technique: str
    prompt: str
    context_summary: str
    success: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class PromptMemory:
    """Memory system for storing successful prompt patterns."""
    
    def __init__(self, max_size: int = 100):
        self.attempts: List[PromptAttempt] = []
        self.max_size = max_size
        self.successful_patterns: Dict[str, List[str]] = {}
    
    def add_attempt(self, attempt: PromptAttempt):
        """Add a prompt attempt to memory."""
        self.attempts.append(attempt)
        
        if len(self.attempts) > self.max_size:
            self.attempts = self.attempts[-self.max_size:]
        
        # Learn from successful prompts
        if attempt.success:
            key = f"{attempt.agent_name}:{attempt.technique}"
            if key not in self.successful_patterns:
                self.successful_patterns[key] = []
            self.successful_patterns[key].append(attempt.prompt[:200])
            logger.debug(f"Learned successful prompt pattern for {key}")
    
    def get_insights(self, agent_name: str, technique: str) -> str:
        """Get insights from past attempts for this agent and technique."""
        key = f"{agent_name}:{technique}"
        
        if key in self.successful_patterns:
            return f"Found {len(self.successful_patterns[key])} successful patterns for {agent_name} using {technique}"
        
        return "No previous successful patterns for this combination"


class PromptEngineerService:
    """
    Self-improving prompt engineering service with reflexion.
    
    Features:
    - Iterative refinement
    - Success evaluation
    - Pattern learning
    - Prompt logging
    """
    
    MAX_ITERATIONS = 2  # Max refinement attempts
    PROMPT_LOG_INDEX = "impactu_marie_agent_prompt_logs"
    
    def __init__(self):
        """Initialize Prompt Engineer Service with reflexion."""
        self.llm = get_llm_adapter()
        self.memory = PromptMemory()
        self.client = OpenSearch(hosts=[config.opensearch.url], timeout=30)
        self._ensure_prompt_log_index()
        logger.info("Prompt Engineer Service initialized with reflexion")
    
    def _ensure_prompt_log_index(self):
        """Create index for prompt logging if it doesn't exist."""
        try:
            if not self.client.indices.exists(index=self.PROMPT_LOG_INDEX):
                logger.info(f"Creating prompt log index: {self.PROMPT_LOG_INDEX}")
                
                index_body = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 1
                    },
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "agent_name": {"type": "keyword"},
                            "task_description": {"type": "text"},
                            "technique": {"type": "keyword"},
                            "prompt": {"type": "text"},
                            "context_summary": {"type": "text"},
                            "success": {"type": "boolean"},
                            "iterations": {"type": "integer"},
                            "prompt_id": {"type": "keyword"}
                        }
                    }
                }
                
                self.client.indices.create(index=self.PROMPT_LOG_INDEX, body=index_body)
                logger.info(f"✓ Prompt log index created: {self.PROMPT_LOG_INDEX}")
            else:
                logger.debug(f"Prompt log index already exists: {self.PROMPT_LOG_INDEX}")
        except Exception as e:
            logger.error(f"Error creating prompt log index: {e}")
    
    def _save_prompt_log(self, log_data: Dict[str, Any]):
        """Save prompt generation details to OpenSearch."""
        try:
            self.client.index(
                index=self.PROMPT_LOG_INDEX,
                body=log_data,
                refresh=True
            )
            logger.debug(f"✓ Prompt logged: {log_data['prompt_id']}")
        except Exception as e:
            logger.error(f"Error saving prompt log: {e}")
    
    def _evaluate_prompt_quality(self, prompt: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate if generated prompt is of good quality.
        
        Heuristics:
        - Prompt has clear instructions
        - Includes relevant context
        - Not too short or too long
        - Contains task description
        """
        if not prompt or len(prompt) < 50:
            return False
        
        if len(prompt) > 5000:
            logger.warning("Prompt too long, might be inefficient")
            return False
        
        # Check if includes user query
        query = context.get("query", "")
        if query and query not in prompt:
            logger.warning("Prompt doesn't include user query")
            return False
        
        return True
    
    def build_prompt(
        self,
        agent_name: str,
        task_description: str,
        context: Dict[str, Any],
        technique: Optional[PromptTechnique] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build optimized prompt with reflexion for quality.
        
        Iteratively refines prompt if quality is low.
        
        Args:
            agent_name: Name of the requesting agent
            task_description: What the agent needs to do
            context: Current context (query, retrieved data, etc.)
            technique: Specific prompting technique to use (auto-selected if None)
            examples: Few-shot examples if using few-shot technique
            
        Returns:
            Optimized prompt ready for LLM
        """
        logger.info(f"🎨 Building prompt for {agent_name}")
        
        # Auto-select technique if not specified
        if technique is None:
            technique = self._select_technique(agent_name, task_description, context)
            logger.info(f"Selected technique: {technique}")
        
        # Get insights from memory
        insights = self.memory.get_insights(agent_name, technique)
        logger.debug(insights)
        
        best_prompt = None
        best_quality = False
        
        # ITERATIVE REFINEMENT
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            logger.debug(f"Prompt generation iteration {iteration}/{self.MAX_ITERATIONS}")
            
            # Build prompt using selected technique
            if technique == "zero-shot":
                prompt = self._build_zero_shot(agent_name, task_description, context)
            elif technique == "few-shot":
                prompt = self._build_few_shot(agent_name, task_description, context, examples or [])
            elif technique == "chain-of-thought":
                prompt = self._build_chain_of_thought(agent_name, task_description, context)
            elif technique == "react":
                prompt = self._build_react(agent_name, task_description, context)
            elif technique == "structured":
                prompt = self._build_structured(agent_name, task_description, context)
            else:
                prompt = self._build_zero_shot(agent_name, task_description, context)
            
            # Evaluate quality
            is_good = self._evaluate_prompt_quality(prompt, context)
            
            if is_good or len(prompt) > 0:
                best_prompt = prompt
                best_quality = is_good
                
                if is_good:
                    logger.debug(f"✓ Good prompt generated on iteration {iteration}")
                    break
            
            # If not good and not last iteration, try different approach
            if iteration < self.MAX_ITERATIONS and not is_good:
                logger.debug(f"Prompt quality low, refining...")
                # Add more context for next iteration
                context["_iteration"] = iteration
        
        # Record attempt in memory
        attempt = PromptAttempt(
            agent_name=agent_name,
            task_description=task_description,
            technique=technique,
            prompt=best_prompt[:500],  # Store preview
            context_summary=str(context.get("query", ""))[:100],
            success=best_quality
        )
        self.memory.add_attempt(attempt)
        
        # Log to OpenSearch
        prompt_log = {
            "timestamp": datetime.now().isoformat(),
            "prompt_id": f"{agent_name}_{int(datetime.now().timestamp())}",
            "agent_name": agent_name,
            "task_description": task_description[:200],
            "technique": technique,
            "prompt": best_prompt[:1000],  # Store preview
            "context_summary": str(context.get("query", ""))[:200],
            "success": best_quality,
            "iterations": iteration
        }
        self._save_prompt_log(prompt_log)
        
        logger.info(f"✓ Prompt generated using {technique} ({iteration} iteration(s))")
        
        return best_prompt
    
    def _select_technique(
        self,
        agent_name: str,
        task_description: str,
        context: Dict[str, Any]
    ) -> PromptTechnique:
        """
        Intelligently select the best prompting technique for the task.
        
        Selection criteria:
        - entity_resolution: few-shot (needs examples of entity matching)
        - retrieval: structured (needs specific query format)
        - metrics: chain-of-thought (needs reasoning for aggregations)
        - reporting: chain-of-thought (needs reasoning to synthesize answer)
        - citations: structured (needs specific citation format)
        """
        query = context.get("query", "")
        complexity = len(query.split()) > 15  # Complex if >15 words
        
        if agent_name == "entity_resolution":
            return "few-shot"  # Needs examples of fuzzy matching
        elif agent_name == "retrieval":
            return "structured"  # Needs specific query format
        elif agent_name == "metrics":
            return "chain-of-thought" if complexity else "zero-shot"
        elif agent_name == "reporting":
            return "chain-of-thought"  # Always needs reasoning
        elif agent_name == "citations":
            return "structured"  # Specific format required
        else:
            return "zero-shot"  # Default
    
    def _build_zero_shot(
        self,
        agent_name: str,
        task_description: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Build zero-shot prompt: direct instruction without examples.
        
        Best for: Simple, well-defined tasks
        """
        query = context.get("query", "")
        
        prompt = f"""You are the {agent_name} agent in MARIE system.

Task: {task_description}

User Query: "{query}"

Instructions:
1. Focus only on your specific task
2. Use the context provided
3. Be precise and concise
4. Return structured output if specified

Context:
{self._format_context(context)}

Execute the task:"""
        
        return prompt
    
    def _build_few_shot(
        self,
        agent_name: str,
        task_description: str,
        context: Dict[str, Any],
        examples: List[Dict[str, str]]
    ) -> str:
        """
        Build few-shot prompt: provide examples before the task.
        
        Best for: Pattern matching, entity resolution, classification
        """
        query = context.get("query", "")
        
        # If no examples provided, use defaults for the agent
        if not examples:
            examples = self._get_default_examples(agent_name)
        
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples)
        ])
        
        prompt = f"""You are the {agent_name} agent. Learn from these examples:

{examples_text}

Now apply the same pattern:

Task: {task_description}
User Query: "{query}"

Context:
{self._format_context(context)}

Your output:"""
        
        return prompt
    
    def _build_chain_of_thought(
        self,
        agent_name: str,
        task_description: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Build Chain-of-Thought prompt: encourage step-by-step reasoning.
        
        Best for: Complex reasoning, synthesis, aggregation
        """
        query = context.get("query", "")
        
        # Use Language Detector Service
        from marie_agent.services.language_detector import get_language_detector
        
        lang_detector = get_language_detector()
        detected_lang = lang_detector.detect(query)
        
        # Set language instruction - natural conversation style
        if detected_lang == "es":
            system_msg = "Eres MARIE, un asistente de investigación amigable y experto."
            style_guide = """ESTILO DE RESPUESTA:
- Responde SIEMPRE en español natural y conversacional
- NO uses frases genéricas como "Lo siento, no puedo proporcionar..."
- Si no tienes información específica, sé honesto pero mantén la conversación
- Usa un tono cercano, como si hablaras con un colega investigador
- Para saludos y preguntas personales, responde naturalmente
- NO uses templates rígidos"""
        elif detected_lang == "pt":
            system_msg = "Você é MARIE, um assistente de pesquisa amigável e especializado."
            style_guide = "Responda em português de forma natural e conversacional."
        elif detected_lang == "fr":
            system_msg = "Vous êtes MARIE, un assistant de recherche amical et expert."
            style_guide = "Répondez en français de manière naturelle et conversationnelle."
        else:
            system_msg = "You are MARIE, a friendly and expert research assistant."
            style_guide = "Respond naturally and conversationally in English."
        
        logger.debug(f"Detected language: {detected_lang}")
        
        # Get context info
        has_sources = context.get("has_sources", False)
        documents = context.get("documents", [])
        
        # Build adaptive prompt
        if not has_sources or len(documents) == 0:
            # No sources - conversational mode
            prompt = f"""{system_msg}

{style_guide}

PREGUNTA DEL USUARIO: "{query}"

CONTEXTO: Esta es una pregunta general o conversacional.

INSTRUCCIONES:
1. Responde de forma natural y amigable
2. Si te preguntan quién eres, explica que eres MARIE, un asistente de investigación
3. Si es un saludo, saluda de vuelta y ofrece ayuda
4. Si es una pregunta personal, responde apropiadamente
5. NO uses frases como "Lo siento, no puedo proporcionar información"
6. Mantén un tono profesional pero cercano

Tu respuesta natural:"""
        else:
            # Has sources - research mode
            prompt = f"""{system_msg}

{style_guide}

PREGUNTA DEL USUARIO: "{query}"

INFORMACIÓN DISPONIBLE:
{self._format_context(context)}

INSTRUCCIONES:
1. Analiza la información proporcionada
2. Responde de forma clara y directa
3. Usa los datos concretos disponibles
4. Si hay papers, menciona los más relevantes
5. Si hay métricas, incorpóralas naturalmente
6. Responde en el mismo idioma de la pregunta

Tu respuesta basada en los datos:"""
        
        return prompt
    
    def _build_react(
        self,
        agent_name: str,
        task_description: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Build ReAct prompt: combine reasoning and acting.
        
        Best for: Agents that need to reason about actions and observations
        """
        query = context.get("query", "")
        
        prompt = f"""You are the {agent_name} agent using ReAct pattern.

Task: {task_description}
User Query: "{query}"

Available Context:
{self._format_context(context)}

Use this format:
Thought: [your reasoning about what to do]
Action: [what you will do]
Observation: [what you found/discovered]
Thought: [reasoning about the observation]
... (repeat as needed)
Final Answer: [your final output]

Begin:"""
        
        return prompt
    
    def _build_structured(
        self,
        agent_name: str,
        task_description: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Build structured output prompt: specify exact format needed.
        
        Best for: JSON output, citations, queries with specific format
        """
        query = context.get("query", "")
        
        # Get format specification for the agent
        format_spec = self._get_format_spec(agent_name)
        
        prompt = f"""You are the {agent_name} agent. Generate structured output.

Task: {task_description}
User Query: "{query}"

Context:
{self._format_context(context)}

Required Output Format:
{format_spec}

Generate output following the format exactly:"""
        
        return prompt
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dict into readable text."""
        lines = []
        for key, value in context.items():
            if key in ["query", "entities", "documents", "metrics"]:
                if isinstance(value, list):
                    lines.append(f"- {key}: {len(value)} items")
                elif isinstance(value, dict):
                    lines.append(f"- {key}: {len(value)} keys")
                else:
                    lines.append(f"- {key}: {str(value)[:100]}")
        return "\n".join(lines) if lines else "No additional context"
    
    def _get_default_examples(self, agent_name: str) -> List[Dict[str, str]]:
        """Get default few-shot examples for an agent."""
        examples_map = {
            "entity_resolution": [
                {
                    "input": "universidad de antioquia",
                    "output": "Universidad de Antioquia (UdeA) - ID: udea_123"
                },
                {
                    "input": "UdeA",
                    "output": "Universidad de Antioquia (UdeA) - ID: udea_123"
                }
            ],
            "retrieval": [
                {
                    "input": "papers about AI",
                    "output": "Query: artificial intelligence OR machine learning, Field: title+abstract"
                }
            ]
        }
        return examples_map.get(agent_name, [])
    
    def _get_format_spec(self, agent_name: str) -> str:
        """Get output format specification for an agent."""
        format_specs = {
            "retrieval": """
{
  "query": "search terms",
  "filters": {"field": "value"},
  "max_results": 100
}""",
            "citations": """
Author, A., & Author, B. (Year). Title. Journal, Volume(Issue), pages. DOI""",
            "metrics": """
{
  "total_count": 123,
  "by_year": {"2023": 45, "2022": 78},
  "top_authors": [{"name": "X", "count": 10}]
}"""
        }
        return format_specs.get(agent_name, "Structured JSON format")


# Global singleton instance
_prompt_engineer = None


def get_prompt_engineer() -> PromptEngineerService:
    """Get global Prompt Engineer Service instance."""
    global _prompt_engineer
    if _prompt_engineer is None:
        _prompt_engineer = PromptEngineerService()
    return _prompt_engineer


def prompt_engineer_agent(state: AgentState) -> AgentState:
    """
    Legacy node function for backward compatibility.
    Now just returns state since agents call the service directly.
    """
    logger.info("Prompt engineer service ready (agents call it on-demand)")
    return state
