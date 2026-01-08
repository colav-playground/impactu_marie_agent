"""
Configuration for MARIE agent system.

Centralizes ALL configuration values to avoid hardcoded constants.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# ============================================================================
# CONSTANTS - OpenSearch Indices
# ============================================================================
class OpenSearchIndices:
    """OpenSearch index names following naming standard: impactu_marie_agent_*"""
    QUERY_LOGS = "impactu_marie_agent_query_logs"
    PROMPT_LOGS = "impactu_marie_agent_prompt_logs"
    PLAN_MEMORY = "impactu_marie_agent_plan_memory"
    EPISODIC_MEMORY = "impactu_marie_agent_episodic_memory"
    AGENT_MEMORY = "impactu_marie_agent_memory"


# ============================================================================
# CONSTANTS - Agent Parameters
# ============================================================================
class AgentConstants:
    """Constants for agent behavior"""
    # OpenSearch Expert
    OPENSEARCH_MAX_ITERATIONS = 3  # Max query refinement attempts
    OPENSEARCH_MIN_RESULTS = 3     # Minimum results to consider success
    OPENSEARCH_TIMEOUT = 30        # Seconds
    
    # Prompt Engineer
    PROMPT_MAX_ITERATIONS = 2      # Max prompt refinement attempts
    PROMPT_MIN_LENGTH = 50         # Minimum prompt length (chars)
    PROMPT_MAX_LENGTH = 5000       # Maximum prompt length (chars)
    PROMPT_PREVIEW_LENGTH = 1000   # Length of prompt to log (chars)
    
    # Memory
    MEMORY_MAX_SIZE = 100          # Max items in memory
    
    # LLM Generation
    LLM_DEFAULT_MAX_TOKENS = 512   # Default for most operations
    LLM_LONG_MAX_TOKENS = 1024     # For longer responses
    
    # Confidence
    DEFAULT_CONFIDENCE = 0.8       # Default confidence score
    
    # Retrieval
    DEFAULT_LIMIT = 10             # Default documents to retrieve


# ============================================================================
# CONSTANTS - Timeouts & Limits
# ============================================================================
class SystemConstants:
    """System-wide constants"""
    OPENSEARCH_TIMEOUT = 30        # OpenSearch operation timeout (seconds)
    MONGODB_TIMEOUT = 10           # MongoDB operation timeout (seconds)
    HTTP_TIMEOUT = 30              # HTTP request timeout (seconds)


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default="ollama", description="LLM provider (ollama, vllm, anthropic)")
    model: str = Field(default="nemotron-3-nano:latest", description="Model name")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=4096, description="Max tokens for response")
    api_key: Optional[str] = Field(default=None, description="API key")
    
    # vLLM specific
    vllm_gpu_memory_utilization: float = Field(default=0.9, description="GPU memory utilization")
    vllm_max_model_len: int = Field(default=2048, description="Max model sequence length")
    vllm_tensor_parallel_size: int = Field(default=1, description="Number of GPUs")


class MongoDBConfig(BaseModel):
    """MongoDB configuration."""
    uri: str = Field(default="mongodb://localhost:27017/", description="MongoDB URI")
    database: str = Field(default="kahi", description="Database name")


class OpenSearchConfig(BaseModel):
    """OpenSearch configuration."""
    url: str = Field(default="http://localhost:9200", description="OpenSearch URL")
    index_prefix: str = Field(default="impactu_marie_agent", description="Index prefix")


class RedisConfig(BaseModel):
    """Redis configuration for distributed caching."""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    enabled: bool = Field(default=False, description="Enable Redis caching")
    
    # TTL settings (seconds)
    schema_ttl: int = Field(default=600, description="Schema cache TTL (10 min)")
    entity_ttl: int = Field(default=300, description="Entity cache TTL (5 min)")
    query_ttl: int = Field(default=60, description="Query cache TTL (1 min)")


class MarieConfig(BaseModel):
    """Main MARIE configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    opensearch: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for autonomous decisions"
    )
    
    enable_human_interaction: bool = Field(
        default=True,
        description="Enable human-in-the-loop features"
    )
    
    log_level: str = Field(default="INFO", description="Logging level")
    
    @classmethod
    def from_env(cls) -> "MarieConfig":
        """Load configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("MARIE_LLM_PROVIDER", "ollama"),
                model=os.getenv("MARIE_LLM_MODEL", "qwen2:1.5b"),
                temperature=float(os.getenv("MARIE_LLM_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MARIE_LLM_MAX_TOKENS", "4096")),
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
            mongodb=MongoDBConfig(
                uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
                database=os.getenv("MONGODB_DATABASE", "kahi")
            ),
            opensearch=OpenSearchConfig(
                url=os.getenv("OPENSEARCH_URL", "http://localhost:9200"),
                index_prefix=os.getenv("OPENSEARCH_INDEX_PREFIX", "impactu_marie_agent")
            ),
            redis=RedisConfig(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD"),
                db=int(os.getenv("REDIS_DB", "0")),
                enabled=os.getenv("REDIS_ENABLED", "false").lower() == "true"
            ),
            confidence_threshold=float(os.getenv("MARIE_CONFIDENCE_THRESHOLD", "0.7")),
            enable_human_interaction=os.getenv("MARIE_ENABLE_HUMAN", "true").lower() == "true",
            log_level=os.getenv("MARIE_LOG_LEVEL", "INFO")
        )


# Global config instance
config = MarieConfig.from_env()
