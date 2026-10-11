from app.shared.config.embedding_config import EmbeddingConfig, embedding_config
from app.shared.config.lm_config import LLMConfig, lm_config
from app.shared.config.bailian_mcp_config import McpConfig, mcp_config
from app.shared.config.milvus_config import MilvusConfig, milvus_config
from app.shared.config.mineru_config import MinerUConfig, mineru_config
from app.shared.config.minio_config import MinIOConfig, minio_config
from app.shared.config.reranker_config import RerankerConfig, reranker_config
from app.shared.config.settings_config import AppSettings, settings

from dataclasses import dataclass , field

#创建一个类赋值当前config对象，导入当前一个类型就可以获得全部config
@dataclass
class InfraConfig:#这里必须要定义一个函数，不能直接赋值，所以用了lambda匿名函数
    app: AppSettings = field(default_factory=lambda: settings)
    llm: LLMConfig = field(default_factory=lambda: lm_config)
    embedding: EmbeddingConfig = field(default_factory=lambda: embedding_config)
    reranker: RerankerConfig = field(default_factory=lambda: reranker_config)
    mcp: McpConfig = field(default_factory=lambda: mcp_config)
    milvus: MilvusConfig = field(default_factory=lambda: milvus_config)
    mineru: MinerUConfig = field(default_factory=lambda: mineru_config)
    minio: MinIOConfig = field(default_factory=lambda: minio_config)

infra_config = InfraConfig()

# print(infra_config.app.import_app_name)