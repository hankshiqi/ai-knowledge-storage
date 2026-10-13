from app.shared.model.lm_utils import get_llm_client
from app.infra.config.providers import infra_config

class LLMProvider:
    def chat(self,model_name:str,json_mode:bool):
        return get_llm_client(model=model_name,json_mode=json_mode)
    #获取视觉模型
    def vision_chat(self,vision_model_name:str=None):
        return get_llm_client(vision_model_name or infra_config.llm.lv_model)

llm_provider = LLMProvider()