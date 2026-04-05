from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model
    model_name: str = "facebook/opt-125m"
    device: str = "cpu"
    max_new_tokens: int = 50

    # Batching
    max_batch_size: int = 8
    batch_timeout_ms: float = 50.0

    # Caching
    cache_backend: str = "redis"
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_prefix = "MLOPS_"

settings = Settings()
