"""
MinerU 配置模块，负责读取文档解析服务相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class MinerUConfig:
    base_url: str
    api_key: str
    poll_timeout_seconds: int       # 轮询总超时时间
    poll_interval_seconds: int      # 轮询间隔时间


mineru_config = MinerUConfig(
    base_url=env_str("MINERU_BASE_URL"),
    api_key=env_str("MINERU_API_TOKEN"),
    # 注意这里加了 int() 转换
    poll_timeout_seconds=int(env_str("MINERU_POLL_TIMEOUT_SECONDS", default=600)),
    poll_interval_seconds=int(env_str("MINERU_POLL_INTERVAL_SECONDS", default=3))
)
