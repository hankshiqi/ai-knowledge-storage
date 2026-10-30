"""BGE-M3 向量生成工具，并提供稠密/稀疏向量缓存。"""

from __future__ import annotations

# RLock 用于保护并发请求下的“缓存未命中 -> 模型计算”过程。
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from app.shared.config.embedding_config import embedding_config
from app.shared.model.embedding_cache import (
    # 全局 embedding 缓存实例，内部包含进程内 LRU 和 MongoDB 持久化缓存。
    embedding_cache,
    # 根据模型配置和文本生成稳定的缓存键。
    make_embedding_cache_key,
    # 在生成缓存键和调用模型前统一规范化文本。
    normalize_embedding_text,
)
from app.shared.runtime.logger import logger

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DEFAULT_EMBEDDING_DEVICE = "cpu"
_bge_m3_ef: "BGEM3EmbeddingFunction | None" = None
# 同一个进程内只允许一个线程为相同的冷缓存执行 BGE-M3 计算，避免重复计算。
_embedding_generation_lock = RLock()


def get_bge_m3_ef() -> "BGEM3EmbeddingFunction":
    """Return the process-local BGE-M3 model singleton."""
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    model_name = embedding_config.bge_m3_path or embedding_config.bge_m3 or _DEFAULT_EMBEDDING_MODEL
    device = embedding_config.bge_device or _DEFAULT_EMBEDDING_DEVICE
    try:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction

        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=embedding_config.bge_fp16,
            normalize_embeddings=True,
        )
        logger.info(f"BGE-M3 initialized: model={model_name}, device={device}")
        return _bge_m3_ef
    except Exception as error:
        logger.error(f"BGE-M3 initialization failed: {error}", exc_info=True)
        raise


def generate_embeddings(texts: list[str], *, use_cache: bool = True) -> dict[str, list]:
    """生成稠密向量和稀疏向量，可按需使用缓存复用重复文本的结果。"""
    # 检查输入必须是非空的文本列表，避免缓存键和模型编码收到非法数据。
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list of strings")
    # 检查列表中的每一项都必须是字符串。
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("texts must be a list of strings")
    # 显式关闭缓存时，直接执行模型编码，不读写任何缓存。
    if not use_cache:
        return _generate_embeddings_uncached(texts)

    # 先统一文本格式，例如统一换行符、Unicode 形式并去除首尾空白。
    normalized_texts = [normalize_embedding_text(text) for text in texts]
    # 为每条规范化文本生成与模型配置绑定的稳定缓存键。
    cache_keys = [make_embedding_cache_key(text) for text in normalized_texts]
    # 一次批量查询进程内 LRU 和 MongoDB，得到已经存在的向量。
    cached = embedding_cache.get_many(cache_keys)
    # 只保留缓存未命中的文本；字典按缓存键去重，同一批重复文本只编码一次。
    missing_texts = {
        key: text
        for key, text in zip(cache_keys, normalized_texts)
        if key not in cached
    }

    # 只有存在缓存未命中时，才进入模型计算和缓存写回流程。
    if missing_texts:
        # 进入进程内锁，避免多个并发请求同时发现 miss 后重复调用模型。
        with _embedding_generation_lock:
            # 等待锁期间其他请求可能已经完成计算，因此需要再次查询缓存。
            cached.update(embedding_cache.get_many(list(missing_texts)))
            #将新查到的缓存结果合并到当前的cached字典当中
            # 删除二次检查后已经命中的键，只留下仍然需要模型计算的文本。
            missing_texts = {
                key: text for key, text in missing_texts.items() if key not in cached
            }
            # 二次检查后仍有未命中项，才真正调用 BGE-M3。
            if missing_texts:
                # 按去重后的文本列表批量生成 dense 和 sparse 两种向量。
                computed = _generate_embeddings_uncached(list(missing_texts.values()))
                # 将模型返回结果按缓存键重新组织，便于写入缓存和后续查找。
                computed_by_key = {
                    key: {
                        # 保存当前文本对应的稠密向量。
                        "dense": computed["dense"][index],
                        # 保存当前文本对应的稀疏向量。
                        "sparse": computed["sparse"][index],
                    }
                    for index, key in enumerate(missing_texts)
                }
                # 将新计算的向量写入 MongoDB，并同步更新进程内 LRU。
                embedding_cache.set_many(computed_by_key)
                # 合并到本次请求的结果集合，后面统一按原始输入顺序返回。
                cached.update(computed_by_key)

    # 按原始 texts 的顺序组装结果；重复文本也会得到对应的同一份向量。
    return {
        # 返回每条文本的稠密向量，顺序与输入 texts 完全一致。
        "dense": [cached[key]["dense"] for key in cache_keys],
        # 返回每条文本的稀疏向量，顺序与输入 texts 完全一致。
        "sparse": [cached[key]["sparse"] for key in cache_keys],
    }


def _generate_embeddings_uncached(texts: list[str]) -> dict[str, list]:
    """绕过缓存调用 BGE-M3，并把 CSR 稀疏矩阵转换为可序列化字典。"""
    # 获取进程内唯一的 BGE-M3 模型实例，避免重复加载模型。
    model = get_bge_m3_ef()
    # 这里是本文件中真正执行模型编码的地方。
    embeddings = model.encode_documents(texts)
    # 准备保存每条文本对应的稀疏向量字典。
    processed_sparse = []
    # BGE-M3 的稀疏结果是 CSR 矩阵，需要按行拆分。
    sparse_matrix = embeddings["sparse"]
    # 每一行代表一条输入文本的稀疏向量。
    for index in range(len(texts)):
        # indptr 给出当前行在 indices/data 数组中的起始位置。
        start = sparse_matrix.indptr[index]
        # indptr 的下一项给出当前行的结束位置。
        end = sparse_matrix.indptr[index + 1]
        # 取出当前文本所有非零特征的维度索引。
        indices = sparse_matrix.indices[start:end].tolist()
        # 取出这些非零特征对应的权重值。
        values = sparse_matrix.data[start:end].tolist()
        # 转成 {特征维度: 权重}，供 Milvus 检索和缓存持久化使用。
        processed_sparse.append(
            {int(dimension): float(weight) for dimension, weight in zip(indices, values)}
        )

    # 将稠密矩阵转成普通列表，并组合稀疏向量结果。
    result = {
        "dense": [vector.tolist() for vector in embeddings["dense"]],
        "sparse": processed_sparse,
    }
    # 记录本次确实执行了多少条未命中缓存的模型计算。
    logger.info(f"Generated {len(texts)} uncached BGE-M3 embeddings")
    # 返回给缓存写回逻辑或显式绕过缓存的调用方。
    return result
