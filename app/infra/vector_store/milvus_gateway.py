from app.shared.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
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
    def create_hybrid_search_requests(self, dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
        return create_hybrid_search_requests(dense_vector, sparse_vector, dense_params, sparse_params, expr, limit)
    def hybrid_search(self, client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,):
        return hybrid_search(client, collection_name, reqs, ranker_weights, norm_score, limit)
milvus_gateway = MilvusGateway()

