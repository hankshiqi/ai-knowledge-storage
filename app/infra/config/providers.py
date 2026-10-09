from app.shared.config.embedding_config import embedding_config
from app.shared.config.lm_config import lm_config
from app.shared.config.bailian_mcp_config import mcp_config
from app.shared.config.milvus_config import milvus_config
from app.shared.config.mineru_config import mineru_config
from app.shared.config.minio_config import minio_config
from app.shared.config.reranker_config import reranker_config
from app.shared.config.settings_config import settings

from dataclasses import dataclass , field

#创建一个类赋值当前config对象，导入当前一个类型就可以获得全部config
@dataclass
class InfraConfig:#这里必须要定义一个函数，不能直接赋值，所以用了lambda匿名函数
    app: object = field(default_factory=lambda: settings) #注意这里要用这种工厂方式来进行复制，右边不能直接等于可变类型的对象，会有类型安全问题。所以需要”复制“一份出来
    llm: object = field(default_factory=lambda: lm_config)
    embedding: object = field(default_factory=lambda: embedding_config)
    reranker: object = field(default_factory=lambda: reranker_config)
    mcp: object = field(default_factory=lambda: mcp_config)
    milvus: object = field(default_factory=lambda: milvus_config)
    mineru: object = field(default_factory=lambda: mineru_config)
    minio: object = field(default_factory=lambda: minio_config)

infra_config = InfraConfig()

print(infra_config.app.import_app_name)