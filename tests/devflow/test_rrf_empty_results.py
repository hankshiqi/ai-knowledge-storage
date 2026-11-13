"""DevFlow integration fixture: regression tests against real RRF functions."""
import pytest
from app.rag.query.rrf_service import fuse_by_rrf, rrf_process

@pytest.mark.parametrize(
    ("embedding_chunks", "hyde_embedding_chunks"),
    [([{"chunk_id": "chunk-1"}], []), ([], [{"chunk_id": "chunk-1"}]),
     ([{"chunk_id": "chunk-1"}], None), (None, [{"chunk_id": "chunk-1"}])],
    ids=["hyde-empty", "embedding-empty", "hyde-null", "embedding-null"],
)
def test_single_available_retriever_still_returns_results(embedding_chunks, hyde_embedding_chunks):
    state = {"embedding_chunks": embedding_chunks, "hyde_embedding_chunks": hyde_embedding_chunks}
    result = fuse_by_rrf(state)
    assert [item["chunk_id"] for item in result["rrf_chunks"]] == ["chunk-1"]

@pytest.mark.parametrize(("embedding_chunks", "hyde_embedding_chunks"), [([], []), (None, []), ([], None), (None, None)])
def test_no_results_produces_empty_fusion(embedding_chunks, hyde_embedding_chunks):
    result = fuse_by_rrf({"embedding_chunks": embedding_chunks, "hyde_embedding_chunks": hyde_embedding_chunks})
    assert result["rrf_chunks"] == []

def test_shared_chunk_is_ranked_first_and_scores_are_preserved():
    results = rrf_process([(1.0, [{"chunk_id": "a"}, {"chunk_id": "b"}]), (1.0, [{"chunk_id": "b"}])])
    assert [item["chunk_id"] for item in results] == ["b", "a"]
    assert results[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
