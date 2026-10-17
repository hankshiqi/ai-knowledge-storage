"""
Embedding 配置模块，负责读取向量模型相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_str


@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # 本地模型路径
    bge_m3: str       # HuggingFace 模型名或云端模型名
    bge_device: str   # 运行设备 (cpu/cuda:0)
    bge_fp16: bool    # 是否使用半精度浮点数

embedding_config = EmbeddingConfig(
    bge_m3_path=env_str("BGE_M3_PATH"),
    bge_m3=env_str("BGE_M3"),
    bge_device=env_str("BGE_DEVICE"),
    bge_fp16=env_bool("BGE_FP16"),
)