"""
Prompt Engineer Agent - Dynamic prompt generation service.

Helps other agents construct optimized prompts using techniques like:
- Zero-shot prompting
- Few-shot prompting  
- Chain-of-Thought (CoT)
- ReAct (Reasoning + Acting)
- Self-consistency
- Structured outputs

Agents call this service to get task-specific prompts based on context.
"""

from typing import Dict, Any, List, Optional, Literal
import logging

from marie_agent.state import AgentState
from marie_agent.adapters.llm_factory import get_llm_adapter

logger = logging.getLogger(__name__)


PromptTechnique = Literal["zero-shot", "few-shot", "chain-of-thought", "react", "structured"]


class PromptEngineerService:
    """
    Service that generates optimized prompts for agents on demand.
    
    Other agents call this service with their task requirements
    and receive back an optimized prompt using appropriate techniques.
    """
    
    def __init__(self):
        """Initialize Prompt Engineer Service."""
        self.llm = get_llm_adapter()
        logger.info("Prompt Engineer Service initialized")
    
    def build_prompt(
        self,
        agent_name: str,
        task_description: str,
        context: Dict[str, Any],
        technique: Optional[PromptTechnique] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build optimized prompt for an agent's task.
        
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
        
        # Build prompt using selected technique
        if technique == "zero-shot":
            return self._build_zero_shot(agent_name, task_description, context)
        elif technique == "few-shot":
            return self._build_few_shot(agent_name, task_description, context, examples or [])
        elif technique == "chain-of-thought":
            return self._build_chain_of_thought(agent_name, task_description, context)
        elif technique == "react":
            return self._build_react(agent_name, task_description, context)
        elif technique == "structured":
            return self._build_structured(agent_name, task_description, context)
        else:
            # Fallback to zero-shot
            return self._build_zero_shot(agent_name, task_description, context)
    
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
        
        # Detect language with more keywords
        spanish_words = ['qué', 'cuántos', 'cómo', 'dónde', 'cuál', 'cuáles', 'por qué', 'para qué',
                        'hola', 'gracias', 'buenos días', 'buenas tardes', 'haz', 'dame', 'muestra',
                        'universidad', 'investigador', 'papers', 'artículos', 'publicaciones']
        
        is_spanish = any(word in query.lower() for word in spanish_words)
        
        if is_spanish:
            lang_instruction = "IMPORTANT: Respond ONLY in Spanish. Do not use English."
            system_msg = "Eres un asistente experto en español."
        else:
            lang_instruction = "Respond in English."
            system_msg = "You are an expert assistant."
        
        prompt = f"""{system_msg}

You are the {agent_name} agent. Think step by step.

Task: {task_description}
User Query: "{query}"
{lang_instruction}

Context:
{self._format_context(context)}

Let's approach this step by step:

Step 1: Understand what the user is asking
Step 2: Analyze the available data
Step 3: Reason about the best approach
Step 4: Execute and generate output

Your reasoning and output:"""
        
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
