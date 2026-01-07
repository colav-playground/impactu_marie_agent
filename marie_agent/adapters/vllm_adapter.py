"""
vLLM Adapter - Local LLM inference with vLLM.

Adapter implementation for running local models with GPU acceleration.
Optimized for 4GB VRAM using small, efficient models.
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os

from marie_agent.ports.llm_port import LLMPort

logger = logging.getLogger(__name__)


class VLLMAdapter(LLMPort):
    """
    vLLM adapter for local model inference.
    
    Supports multiple models optimized for 4GB VRAM:
    - Qwen/Qwen2-1.5B-Instruct: Best quality for size
    - microsoft/phi-2: Strong reasoning
    - HuggingFaceTB/SmolLM-1.7B-Instruct: Fast inference
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048
    ):
        """
        Initialize vLLM adapter.
        
        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs (1 for single GPU)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
        """
        self.model_name = model_name
        self.llm = None
        self._available = False
        
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Max model length: {max_model_len}")
            logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
            
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
                dtype="auto",
                enforce_eager=True  # Disable CUDA graphs for compatibility
            )
            
            self.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=512
            )
            
            self._available = True
            logger.info(f"vLLM initialized successfully: {model_name}")
            
        except ImportError:
            logger.warning("vLLM not installed. Install with: uv pip install vllm")
        except Exception as e:
            logger.error(f"Error initializing vLLM: {e}")
            logger.warning("Falling back to rule-based methods")
    
    def is_available(self) -> bool:
        """Check if vLLM is available."""
        return self._available
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text completion."""
        if not self.is_available():
            return ""
        
        try:
            from vllm import SamplingParams
            
            sampling = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=max_tokens
            )
            
            outputs = self.llm.generate([prompt], sampling)
            return outputs[0].outputs[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def think(self, prompt: str, context: str = "") -> str:
        """
        Generate reasoning/thinking response.
        
        Uses chain-of-thought prompting for better reasoning.
        """
        if not self.is_available():
            return ""
        
        # Chain-of-thought prompt
        full_prompt = f"""Think step by step about this problem:

{prompt}

{f'Context: {context}' if context else ''}

Let's approach this systematically:
1. First, let me understand what's being asked
2. Then, I'll break down the problem
3. Finally, I'll provide a clear answer

Reasoning:"""
        
        return self.generate(full_prompt, max_tokens=1024)
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse user query using LLM."""
        if not self.is_available():
            return self._parse_query_fallback(query)
        
        prompt = f"""Extract structured information from this scientometric query.

Query: "{query}"

Analyze and extract:
1. Intent: What type of analysis is requested
2. Entities: Institutions, authors, research groups mentioned
3. Filters: Year ranges, document types, criteria
4. Metrics: What metrics to compute
5. Limit: Number of results

Respond ONLY with valid JSON:
{{
  "intent": "top_papers|author_productivity|collaboration_network|search",
  "entities": {{
    "institutions": ["name1", "name2"],
    "authors": ["name1"],
    "groups": ["name1"]
  }},
  "filters": {{
    "year_start": null,
    "year_end": null,
    "document_types": []
  }},
  "metrics": ["citations", "publications"],
  "limit": 5,
  "complexity": "simple|medium|complex"
}}

JSON:"""
        
        response = self.generate(prompt, max_tokens=512)
        
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                logger.info(f"Query parsed: intent={result.get('intent')}")
                return result
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
        
        return self._parse_query_fallback(query)
    
    def _parse_query_fallback(self, query: str) -> Dict[str, Any]:
        """Rule-based fallback for query parsing."""
        query_lower = query.lower()
        
        # Detect intent
        intent = "search"
        if "top" in query_lower or "most cited" in query_lower:
            intent = "top_papers"
        elif "author" in query_lower and ("productivity" in query_lower or "publications" in query_lower):
            intent = "author_productivity"
        
        # Extract limit
        limit = 5
        for num in [3, 5, 10, 20, 50]:
            if str(num) in query:
                limit = num
                break
        
        return {
            "intent": intent,
            "entities": {"institutions": [], "authors": [], "groups": []},
            "filters": {},
            "metrics": ["citations"],
            "limit": limit,
            "complexity": "simple"
        }
    
    def extract_entities(self, query: str, entity_type: str) -> List[Dict[str, Any]]:
        """Extract entities using LLM."""
        if not self.is_available():
            return []
        
        prompt = f"""Extract {entity_type} names from this query.

Query: "{query}"

For each {entity_type}, provide:
- name: The extracted name
- aliases: Possible alternative names
- confidence: Your confidence (0.0-1.0)

Respond ONLY with valid JSON:
{{
  "entities": [
    {{"name": "string", "aliases": ["string"], "confidence": 0.9}}
  ]
}}

JSON:"""
        
        response = self.generate(prompt, max_tokens=256)
        
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result.get("entities", [])
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return []
    
    def create_plan(self, query: str, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan using LLM."""
        if not self.is_available():
            return self._create_default_plan()
        
        prompt = f"""Create an execution plan for this scientometric query.

Query: "{query}"
Parsed intent: {parsed_query.get('intent')}
Complexity: {parsed_query.get('complexity')}

Available agents:
- entity_resolution: Disambiguate authors/institutions
- retrieval: Search databases
- validation: Check data consistency  
- metrics: Compute indicators
- citations: Build evidence map
- reporting: Generate report

Determine:
1. Which agents are needed
2. Order of execution
3. Whether human input is required

Respond ONLY with valid JSON:
{{
  "steps": ["step description"],
  "agents_required": ["agent_name"],
  "requires_human_input": false,
  "estimated_complexity": "simple|medium|complex",
  "reasoning": "brief explanation"
}}

JSON:"""
        
        response = self.generate(prompt, max_tokens=512)
        
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                logger.info(f"Plan created: {len(result.get('steps', []))} steps")
                return result
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
        
        return self._create_default_plan()
    
    def _create_default_plan(self) -> Dict[str, Any]:
        """Default execution plan."""
        return {
            "steps": [
                "Resolve entities",
                "Retrieve evidence",
                "Validate data",
                "Compute metrics",
                "Build citations",
                "Generate report"
            ],
            "agents_required": [
                "entity_resolution",
                "retrieval",
                "validation",
                "metrics",
                "citations",
                "reporting"
            ],
            "requires_human_input": False,
            "estimated_complexity": "medium",
            "reasoning": "Default plan"
        }
    
    def assess_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence using LLM reasoning."""
        if not self.is_available():
            return self._assess_confidence_fallback(state)
        
        evidence_count = len(state.get("evidence_map", {}))
        docs_count = len(state.get("retrieved_data", []))
        citations_count = len(state.get("citations", []))
        entities_count = len(state.get("entities_resolved", {}).get("institutions", []))
        
        prompt = f"""Assess confidence in these scientometric results.

Data available:
- Evidence sources: {evidence_count}
- Documents retrieved: {docs_count}
- Citations: {citations_count}
- Entities resolved: {entities_count}

Consider:
1. Data completeness
2. Entity resolution quality
3. Evidence strength
4. Metric reliability

Provide assessment with valid JSON:
{{
  "confidence_score": 0.85,
  "confidence_level": "high",
  "reasoning": "Strong evidence base with good entity resolution",
  "limitations": ["Limited to indexed data", "Citation counts may be outdated"]
}}

JSON:"""
        
        response = self.generate(prompt, max_tokens=256)
        
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                logger.info(f"Confidence assessed: {result.get('confidence_level')}")
                return result
        except Exception as e:
            logger.error(f"Error assessing confidence: {e}")
        
        return self._assess_confidence_fallback(state)
    
    def _assess_confidence_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based confidence assessment."""
        evidence_count = len(state.get("evidence_map", {}))
        docs_count = len(state.get("retrieved_data", []))
        citations_count = len(state.get("citations", []))
        
        score = 0.0
        if evidence_count > 0:
            score += 0.3
        if docs_count >= 5:
            score += 0.4
        if citations_count >= 3:
            score += 0.3
        
        level = "low"
        if score >= 0.7:
            level = "high"
        elif score >= 0.4:
            level = "medium"
        
        return {
            "confidence_score": score,
            "confidence_level": level,
            "reasoning": f"Based on {evidence_count} evidence sources, {docs_count} documents, {citations_count} citations",
            "limitations": ["Rule-based assessment without LLM reasoning"]
        }
