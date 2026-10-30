from app.shared.model.embedding_utils import generate_embeddings, get_bge_m3_ef
from app.shared.model.lm_utils import get_llm_client
from app.infra.config.providers import infra_config
from app.shared.model.reranker_utils import get_reranker_model

class LLMProvider:
    def chat(self,model_name:str=None,json_mode:bool=None):
        return get_llm_client(model=model_name,json_mode=json_mode)
    #获取视觉模型
    def vision_chat(self,vision_model_name:str=None):
        return get_llm_client(vision_model_name or infra_config.llm.lv_model)
    #获取嵌入模型
    def bge_m3_embedding(self):
        return get_bge_m3_ef()
    #生成向量
    def generate_embeddings(self, texts: list[str], *, use_cache: bool = True):
        return generate_embeddings(texts, use_cache=use_cache)
    def get_reranker_model(self):
        return get_reranker_model()
llm_provider = LLMProvider()
