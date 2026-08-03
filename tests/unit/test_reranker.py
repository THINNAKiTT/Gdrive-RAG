"""
Unit tests for src/rag/reranker.py (Reranker)

CrossEncoder is mocked (no real model download or GPU/CPU inference)
so these tests focus on: correct integration with llama-index's
BaseNodePostprocessor interface, score reassignment, sort order, and
top_n truncation -- the logic that's actually ours, as opposed to the
sentence-transformers library's own correctness (which is out of
scope for this project's tests).
"""
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle

from src.rag.reranker import Reranker

pytestmark = pytest.mark.unit


def _make_node(text: str, node_id: str, initial_score: float = 0.5) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, id_=node_id), score=initial_score)


@pytest.fixture
def mock_cross_encoder():
    with patch("src.rag.reranker.CrossEncoder") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def reranker(mock_cross_encoder):
    return Reranker(top_n=4)


def test_reranker_class_name():
    assert Reranker.class_name() == "Reranker"


def test_reranker_loads_default_model_name(mock_cross_encoder, monkeypatch):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)

    r = Reranker()

    assert r.model_name == "BAAI/bge-reranker-v2-m3"


def test_reranker_uses_reranker_model_env_var(mock_cross_encoder, monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "some/other-model")

    r = Reranker()

    assert r.model_name == "some/other-model"


def test_reranker_explicit_model_name_overrides_env_var(mock_cross_encoder, monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "env-model")

    r = Reranker(model_name="explicit-model")

    assert r.model_name == "explicit-model"


def test_reranker_defaults_top_n_to_four(mock_cross_encoder):
    r = Reranker()

    assert r.top_n == 4


def test_postprocess_nodes_with_empty_query_bundle_returns_first_top_n_unchanged(reranker, mock_cross_encoder):
    nodes = [_make_node(f"doc {i}", f"id-{i}") for i in range(10)]

    result = reranker._postprocess_nodes(nodes, query_bundle=None)

    assert len(result) == 4
    mock_cross_encoder.predict.assert_not_called()


def test_postprocess_nodes_with_no_nodes_returns_empty_list(reranker, mock_cross_encoder):
    result = reranker._postprocess_nodes([], query_bundle=QueryBundle(query_str="a query"))

    assert result == []


def test_postprocess_nodes_calls_cross_encoder_with_query_document_pairs(reranker, mock_cross_encoder):
    mock_cross_encoder.predict.return_value = [0.1, 0.2, 0.3]
    nodes = [
        _make_node("Document A content", "id-a"),
        _make_node("Document B content", "id-b"),
        _make_node("Document C content", "id-c"),
    ]

    reranker._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="What is X?"))

    call_args = mock_cross_encoder.predict.call_args[0][0]
    assert call_args == [
        ("What is X?", "Document A content"),
        ("What is X?", "Document B content"),
        ("What is X?", "Document C content"),
    ]


def test_postprocess_nodes_reorders_by_cross_encoder_score(reranker, mock_cross_encoder):
    """
    The cross-encoder's scores should completely override the
    original vector-similarity scores -- a node with a low vector
    score but a high cross-encoder score must end up ranked first.
    """
    node_low_vector_high_rerank = _make_node("very relevant text", "id-1", initial_score=0.1)
    node_high_vector_low_rerank = _make_node("less relevant text", "id-2", initial_score=0.9)
    nodes = [node_low_vector_high_rerank, node_high_vector_low_rerank]

    # cross-encoder disagrees with the original vector ranking
    mock_cross_encoder.predict.return_value = [9.5, 0.2]

    result = reranker._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="query"))

    assert result[0].node.id_ == "id-1"
    assert result[1].node.id_ == "id-2"


def test_postprocess_nodes_overwrites_score_with_cross_encoder_value(reranker, mock_cross_encoder):
    nodes = [_make_node("some text", "id-1", initial_score=0.5)]
    mock_cross_encoder.predict.return_value = [3.7]

    result = reranker._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="query"))

    assert result[0].score == pytest.approx(3.7)


def test_postprocess_nodes_truncates_to_top_n(mock_cross_encoder):
    r = Reranker(top_n=2)
    nodes = [_make_node(f"doc {i}", f"id-{i}") for i in range(5)]
    mock_cross_encoder.predict.return_value = [0.5, 0.9, 0.1, 0.7, 0.3]

    result = r._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="query"))

    assert len(result) == 2
    # highest two scores were 0.9 (id-1) and 0.7 (id-3)
    assert {n.node.id_ for n in result} == {"id-1", "id-3"}


def test_postprocess_nodes_handles_negative_scores_correctly(reranker, mock_cross_encoder):
    """
    BGE-style cross-encoders return raw logits, which can be negative
    -- sort order must still work correctly (higher, even if negative,
    ranks above lower).
    """
    nodes = [
        _make_node("doc A", "id-a"),
        _make_node("doc B", "id-b"),
    ]
    mock_cross_encoder.predict.return_value = [-8.2, -1.5]

    result = reranker._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="query"))

    assert result[0].node.id_ == "id-b"  # -1.5 > -8.2
    assert result[1].node.id_ == "id-a"


def test_reranker_model_loaded_once_not_per_query(mock_cross_encoder):
    """
    Loading a CrossEncoder is expensive -- it must happen once at
    construction time, not on every _postprocess_nodes() call.
    """
    with patch("src.rag.reranker.CrossEncoder") as mock_cls:
        instance = MagicMock()
        instance.predict.return_value = [0.5]
        mock_cls.return_value = instance

        r = Reranker()
        nodes = [_make_node("doc", "id-1")]
        query_bundle = QueryBundle(query_str="q")

        r._postprocess_nodes(nodes, query_bundle)
        r._postprocess_nodes(nodes, query_bundle)
        r._postprocess_nodes(nodes, query_bundle)

        mock_cls.assert_called_once()