"""DevFlow integration fixture: input-contract regressions without model or database access."""
import pytest
from pydantic import ValidationError
from app.api.schema.query import QueryRequest

@pytest.mark.parametrize("query", [None, "", " ", "\t\r\n", "\u3000"], ids=["null", "empty", "space", "control-whitespace", "ideographic-space"])
def test_rejects_queries_without_visible_text(query):
    with pytest.raises(ValidationError):
        QueryRequest(query=query)

@pytest.mark.parametrize("query", ["HAK180如何操作？", "How does hybrid retrieval work?"])
def test_accepts_valid_queries(query):
    request = QueryRequest(query=query)
    assert request.query == query

def test_query_remains_required():
    with pytest.raises(ValidationError):
        QueryRequest()
