"""
Unit tests for src/rag/query_rewriter.py (QueryRewriter)

The Ollama LLM is mocked so these tests need no running Ollama
server. Coverage focuses on: rewriting behavior with/without history,
the empty-history fast path (skips the LLM call entirely), and the
fallback-to-original-query behavior on any failure (timeout, circuit
breaker open, malformed response) -- since query rewriting is a
quality-of-life feature, not a hard dependency of the RAG pipeline.
"""
from unittest.mock import MagicMock

import httpx
import pytest

from src.rag.query_rewriter import QueryRewriter
from src.utils.resilience import CircuitOpenError

pytestmark = pytest.mark.unit


@pytest.fixture
def rewriter(mock_query_rewriter_llm):
    return QueryRewriter(max_turns=6)


def _mock_response(text: str):
    """QueryRewriter does str(response).strip() on whatever
    self.llm.complete() returns, so a MagicMock with a controlled
    __str__ stands in for llama-index's CompletionResponse."""
    response = MagicMock()
    response.__str__ = MagicMock(return_value=text)
    return response


# ---------------------------------------------------------------------------
# Empty history -- fast path, no LLM call at all
# ---------------------------------------------------------------------------


def test_rewrite_with_empty_history_returns_query_unchanged(rewriter, mock_query_rewriter_llm):
    result = rewriter.rewrite("What is RLHF?", history=[])

    assert result == "What is RLHF?"


def test_rewrite_with_empty_history_never_calls_the_llm(rewriter, mock_query_rewriter_llm):
    rewriter.rewrite("What is RLHF?", history=[])

    mock_query_rewriter_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Rewriting with history -- the main happy path
# ---------------------------------------------------------------------------


def test_rewrite_resolves_pronoun_using_history(rewriter, mock_query_rewriter_llm):
    mock_query_rewriter_llm.complete.return_value = _mock_response("How does RLHF work?")
    history = [
        {"role": "user", "content": "What is RLHF?"},
        {"role": "assistant", "content": "RLHF is Reinforcement Learning from Human Feedback."},
    ]

    result = rewriter.rewrite("How does it work?", history)

    assert result == "How does RLHF work?"


def test_rewrite_calls_llm_with_history_and_query_in_prompt(rewriter, mock_query_rewriter_llm):
    mock_query_rewriter_llm.complete.return_value = _mock_response("How does RLHF work?")
    history = [
        {"role": "user", "content": "What is RLHF?"},
        {"role": "assistant", "content": "RLHF is Reinforcement Learning from Human Feedback."},
    ]

    rewriter.rewrite("How does it work?", history)

    prompt_arg = mock_query_rewriter_llm.complete.call_args[0][0]
    assert "What is RLHF?" in prompt_arg
    assert "RLHF is Reinforcement Learning from Human Feedback." in prompt_arg
    assert "How does it work?" in prompt_arg


def test_rewrite_returns_original_when_already_standalone(rewriter, mock_query_rewriter_llm):
    """If the model decides the question needs no rewriting, it
    should return it unchanged -- this just verifies that pass-through
    result is returned faithfully, not mangled."""
    mock_query_rewriter_llm.complete.return_value = _mock_response("What is the capital of France?")
    history = [
        {"role": "user", "content": "What is RLHF?"},
        {"role": "assistant", "content": "RLHF is Reinforcement Learning from Human Feedback."},
    ]

    result = rewriter.rewrite("What is the capital of France?", history)

    assert result == "What is the capital of France?"


def test_rewrite_strips_whitespace_from_llm_response(rewriter, mock_query_rewriter_llm):
    mock_query_rewriter_llm.complete.return_value = _mock_response("  How does RLHF work?  \n")

    result = rewriter.rewrite(
        "How does it work?",
        [{"role": "user", "content": "What is RLHF?"}],
    )

    assert result == "How does RLHF work?"


# ---------------------------------------------------------------------------
# Fallback behavior -- rewriting must never block the main RAG pipeline
# ---------------------------------------------------------------------------


def test_rewrite_falls_back_to_original_query_on_circuit_open(rewriter, mock_query_rewriter_llm, monkeypatch):
    def raise_circuit_open(prompt):
        raise CircuitOpenError("circuit is open")

    monkeypatch.setattr(
        "src.rag.query_rewriter.with_resilience",
        lambda func: raise_circuit_open,
    )

    result = rewriter.rewrite(
        "How does it work?",
        [{"role": "user", "content": "What is RLHF?"}],
    )

    assert result == "How does it work?"


def test_rewrite_falls_back_to_original_query_on_timeout(rewriter, mock_query_rewriter_llm, monkeypatch):
    def raise_timeout(prompt):
        raise httpx.ReadTimeout("took too long")

    monkeypatch.setattr(
        "src.rag.query_rewriter.with_resilience",
        lambda func: raise_timeout,
    )

    result = rewriter.rewrite(
        "How does it work?",
        [{"role": "user", "content": "What is RLHF?"}],
    )

    assert result == "How does it work?"


def test_rewrite_falls_back_to_original_query_on_unexpected_exception(rewriter, mock_query_rewriter_llm, monkeypatch):
    def raise_weird_error(prompt):
        raise RuntimeError("something totally unexpected")

    monkeypatch.setattr(
        "src.rag.query_rewriter.with_resilience",
        lambda func: raise_weird_error,
    )

    result = rewriter.rewrite(
        "How does it work?",
        [{"role": "user", "content": "What is RLHF?"}],
    )

    assert result == "How does it work?"


def test_rewrite_falls_back_to_original_query_on_empty_llm_response(rewriter, mock_query_rewriter_llm):
    mock_query_rewriter_llm.complete.return_value = _mock_response("")

    result = rewriter.rewrite(
        "How does it work?",
        [{"role": "user", "content": "What is RLHF?"}],
    )

    assert result == "How does it work?"


def test_rewrite_falls_back_to_original_query_on_whitespace_only_response(rewriter, mock_query_rewriter_llm):
    mock_query_rewriter_llm.complete.return_value = _mock_response("   \n  ")

    result = rewriter.rewrite(
        "How does it work?",
        [{"role": "user", "content": "What is RLHF?"}],
    )

    assert result == "How does it work?"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_rewriter_uses_query_rewrite_model_env_var(monkeypatch, mock_query_rewriter_llm):
    import src.rag.query_rewriter as query_rewriter_module

    ctor_spy = MagicMock(return_value=mock_query_rewriter_llm)
    monkeypatch.setattr(query_rewriter_module, "Ollama", ctor_spy)
    monkeypatch.setenv("QUERY_REWRITE_MODEL", "custom-tiny-model")

    QueryRewriter()

    _, kwargs = ctor_spy.call_args
    assert kwargs["model"] == "custom-tiny-model"


def test_rewriter_defaults_to_qwen_when_env_var_unset(monkeypatch, mock_query_rewriter_llm):
    import src.rag.query_rewriter as query_rewriter_module

    ctor_spy = MagicMock(return_value=mock_query_rewriter_llm)
    monkeypatch.setattr(query_rewriter_module, "Ollama", ctor_spy)
    monkeypatch.delenv("QUERY_REWRITE_MODEL", raising=False)

    QueryRewriter()

    _, kwargs = ctor_spy.call_args
    assert kwargs["model"] == "qwen2.5:0.5b"