"""
Unit tests for src/rag/orchestrator.py (RAGOrchestrator)

Ollama LLM/embedding classes and VectorDBManager are all mocked so
these tests never require a running Ollama server or a real ChromaDB.
"""
import pytest
from llama_index.core import Settings

from src.rag.orchestrator import RAGOrchestrator

pytestmark = pytest.mark.unit


@pytest.fixture
def orchestrator(mock_ollama_llm, mock_ollama_embedding, mock_vector_db_manager):
    return RAGOrchestrator()


def test_orchestrator_configures_ollama_embedding_with_env_defaults(
    orchestrator, mock_ollama_embedding
):
    assert Settings.embed_model is mock_ollama_embedding


def test_orchestrator_configures_ollama_llm_with_env_defaults(
    orchestrator, mock_ollama_llm
):
    assert Settings.llm is mock_ollama_llm


def test_orchestrator_uses_temperature_zero_for_determinism(monkeypatch, mock_vector_db_manager):
    from unittest.mock import MagicMock
    from llama_index.core.llms import MockLLM
    from llama_index.core.embeddings import MockEmbedding

    ollama_ctor = MagicMock(return_value=MockLLM())
    monkeypatch.setattr("src.rag.orchestrator.Ollama", ollama_ctor)
    monkeypatch.setattr(
        "src.rag.orchestrator.OllamaEmbedding",
        MagicMock(return_value=MockEmbedding(embed_dim=8)),
    )

    RAGOrchestrator()

    _, kwargs = ollama_ctor.call_args
    assert kwargs["temperature"] == 0.0


def test_orchestrator_loads_index_via_db_manager(orchestrator, mock_vector_db_manager):
    assert orchestrator.index is mock_vector_db_manager.load_index.return_value


def test_get_query_engine_uses_strict_rag_prompt(orchestrator):
    from src.rag.prompt_templates import STRICT_RAG_PROMPT

    orchestrator.get_query_engine()

    call_kwargs = orchestrator.index.as_query_engine.call_args.kwargs
    assert call_kwargs["text_qa_template"].template == STRICT_RAG_PROMPT


def test_get_query_engine_uses_top_k_four(orchestrator):
    orchestrator.get_query_engine()

    call_kwargs = orchestrator.index.as_query_engine.call_args.kwargs
    assert call_kwargs["similarity_top_k"] == 4
