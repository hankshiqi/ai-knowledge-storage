from app.shared.clients.milvus_utils import get_milvus_client
from app.shared.config.milvus_config import milvus_config

class MilvusGateway:
    @property
    def get_chunks_collection(self):
        return milvus_config.chunks_collection
    @property
    def get_item_name_collection(self):
        return milvus_config.item_name_collection
    @property
    def milvus_client(self):
        return get_milvus_client()
milvus_gateway = MilvusGateway()

