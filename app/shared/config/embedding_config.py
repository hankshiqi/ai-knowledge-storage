"""Embedding model and vector cache configuration."""

from dataclasses import dataclass

from app.shared.config.common import env_bool, env_int, env_str


@dataclass
class EmbeddingConfig:
    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool
    cache_enabled: bool = True
    cache_collection: str = "embedding_cache"
    cache_ttl_seconds: int = 0
    cache_server_selection_timeout_ms: int = 500
    cache_model_version: str = "v1"
    cache_max_entries: int = 1024


embedding_config = EmbeddingConfig(
    bge_m3_path=env_str("BGE_M3_PATH"),
    bge_m3=env_str("BGE_M3"),
    bge_device=env_str("BGE_DEVICE"),
    bge_fp16=env_bool("BGE_FP16"),
    cache_enabled=env_bool("EMBEDDING_CACHE_ENABLED", True),
    cache_collection=env_str("EMBEDDING_CACHE_COLLECTION", "embedding_cache"),
    cache_ttl_seconds=max(0, env_int("EMBEDDING_CACHE_TTL_SECONDS", 0)),
    cache_server_selection_timeout_ms=max(
        100, env_int("EMBEDDING_CACHE_SERVER_SELECTION_TIMEOUT_MS", 500)
    ),
    cache_model_version=env_str("EMBEDDING_CACHE_MODEL_VERSION", "v1"),
    cache_max_entries=max(1, env_int("EMBEDDING_CACHE_MAX_ENTRIES", 1024)),
)
