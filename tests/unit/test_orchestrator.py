"""
Unit tests for src/rag/orchestrator.py (RAGOrchestrator)

RAGOrchestrator now delegates all provider selection to
src.rag.providers (get_llm_provider, get_embedding_provider,
get_reranker_provider) -- these tests patch that single seam rather
than any specific provider's SDK class, so they pass regardless of
what LLM_PROVIDER/EMBEDDING_PROVIDER/RERANKER_PROVIDER happen to be
set to in the environment. Provider-specific behavior (e.g. "does
get_llm_provider('openai') set temperature=0.0") belongs in
test_providers.py, not here.
"""
import pytest
from llama_index.core import Settings

from src.rag.orchestrator import RAGOrchestrator

pytestmark = pytest.mark.unit

@pytest.fixture
def orchestrator(mock_ollama_llm, mock_ollama_embedding, mock_vector_db_manager, mock_reranker):
    return RAGOrchestrator()


def test_orchestrator_configures_embed_model_via_provider_factory(
    orchestrator, mock_ollama_embedding
):
    assert Settings.embed_model is mock_ollama_embedding


def test_orchestrator_configures_llm_via_provider_factory(
    orchestrator, mock_ollama_llm
):
    assert Settings.llm is mock_ollama_llm


def test_orchestrator_loads_index_via_db_manager(orchestrator, mock_vector_db_manager):
    assert orchestrator.index is mock_vector_db_manager.load_index.return_value


def test_orchestrator_configures_node_parser_globally(orchestrator):
    from llama_index.core.node_parser import SentenceSplitter

    assert isinstance(Settings.node_parser, SentenceSplitter)
    assert Settings.node_parser.chunk_size == 512
    assert Settings.node_parser.chunk_overlap == 50


def test_orchestrator_creates_reranker_via_provider_factory_with_top_n_four(
    monkeypatch, mock_ollama_llm, mock_ollama_embedding, mock_vector_db_manager
):
    from unittest.mock import MagicMock

    fake_reranker = MagicMock(name="Reranker")
    reranker_ctor = MagicMock(name="get_reranker_provider", return_value=fake_reranker)
    monkeypatch.setattr("src.rag.orchestrator.get_reranker_provider", reranker_ctor)

    orchestrator = RAGOrchestrator()

    reranker_ctor.assert_called_once_with(top_n=4)
    assert orchestrator.reranker is fake_reranker


def test_get_query_engine_uses_strict_rag_prompt(orchestrator):
    from src.rag.prompt_templates import STRICT_RAG_PROMPT

    orchestrator.get_query_engine()

    call_kwargs = orchestrator.index.as_query_engine.call_args.kwargs
    assert call_kwargs["text_qa_template"].template == STRICT_RAG_PROMPT


def test_get_query_engine_uses_top_k_fifteen_for_reranker_candidate_pool(orchestrator):
    orchestrator.get_query_engine()

    call_kwargs = orchestrator.index.as_query_engine.call_args.kwargs
    assert call_kwargs["similarity_top_k"] == 15


def test_get_query_engine_includes_reranker_as_node_postprocessor(orchestrator, mock_reranker):
    orchestrator.get_query_engine()

    call_kwargs = orchestrator.index.as_query_engine.call_args.kwargs
    assert call_kwargs["node_postprocessors"] == [mock_reranker]