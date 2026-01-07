"""
Configuration for MARIE agent system.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default="vllm", description="LLM provider (vllm, anthropic)")
    model: str = Field(default="Qwen/Qwen2-1.5B-Instruct", description="Model name")
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


class MarieConfig(BaseModel):
    """Main MARIE configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    opensearch: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    
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
                provider=os.getenv("MARIE_LLM_PROVIDER", "anthropic"),
                model=os.getenv("MARIE_LLM_MODEL", "claude-3-5-sonnet-20241022"),
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
            confidence_threshold=float(os.getenv("MARIE_CONFIDENCE_THRESHOLD", "0.7")),
            enable_human_interaction=os.getenv("MARIE_ENABLE_HUMAN", "true").lower() == "true",
            log_level=os.getenv("MARIE_LOG_LEVEL", "INFO")
        )


# Global config instance
config = MarieConfig.from_env()
