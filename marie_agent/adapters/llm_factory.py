"""
LLM Factory - Creates appropriate LLM adapter based on configuration.

Implements hexagonal architecture by selecting adapter at runtime.
"""

from typing import Optional
import logging

from marie_agent.ports.llm_port import LLMPort
from marie_agent.config import config

logger = logging.getLogger(__name__)


def create_llm_adapter() -> LLMPort:
    """
    Create LLM adapter based on configuration.
    
    Returns:
        LLMPort implementation (Ollama, vLLM, Anthropic, or fallback)
    """
    provider = config.llm.provider.lower()
    
    logger.info(f"Creating LLM adapter: {provider}")
    logger.info(f"📦 Model from config: {config.llm.model}")
    
    if provider == "ollama":
        try:
            from marie_agent.adapters.ollama_adapter import OllamaAdapter
            
            logger.info("Attempting to initialize Ollama...")
            adapter = OllamaAdapter(
                model_name=config.llm.model,
                temperature=config.llm.temperature
            )
            
            if adapter.is_available():
                logger.info("✓ Ollama adapter initialized successfully")
                return adapter
            else:
                logger.warning("✗ Ollama not available, using fallback")
                
        except ImportError as e:
            logger.warning(f"Ollama adapter not available: {e}")
        except Exception as e:
            logger.error(f"Error creating Ollama adapter: {e}", exc_info=True)
    
    elif provider == "vllm":
        try:
            from marie_agent.adapters.vllm_adapter import VLLMAdapter
            
            logger.info("Attempting to initialize vLLM...")
            adapter = VLLMAdapter(
                model_name=config.llm.model,
                tensor_parallel_size=config.llm.vllm_tensor_parallel_size,
                gpu_memory_utilization=config.llm.vllm_gpu_memory_utilization,
                max_model_len=config.llm.vllm_max_model_len
            )
            
            if adapter.is_available():
                logger.info("✓ vLLM adapter initialized successfully")
                return adapter
            else:
                logger.warning("✗ vLLM initialization failed, using fallback")
                
        except ImportError as e:
            logger.warning(f"vLLM not available: {e}")
        except Exception as e:
            logger.error(f"Error creating vLLM adapter: {e}", exc_info=True)
    
    elif provider == "anthropic":
        try:
            from marie_agent.llm import MarieLLM
            
            adapter = MarieLLM()
            if adapter.available:
                logger.info("Anthropic adapter initialized successfully")
                return adapter
            else:
                logger.warning("Anthropic not available, falling back")
                
        except Exception as e:
            logger.error(f"Error creating Anthropic adapter: {e}")
    
    # Fallback to rule-based adapter
    logger.info("Using rule-based fallback adapter (no LLM)")
    from marie_agent.adapters.fallback_adapter import FallbackAdapter
    return FallbackAdapter()


# Global adapter instance
_llm_adapter: Optional[LLMPort] = None


def get_llm_adapter() -> LLMPort:
    """Get or create global LLM adapter instance."""
    global _llm_adapter
    if _llm_adapter is None:
        _llm_adapter = create_llm_adapter()
    return _llm_adapter
