"""
Unit tests for src/rag/providers.py

Since every provider SDK is an optional dependency, these tests mock
each provider's import location directly (patching sys.modules) so
the test suite can exercise ALL providers (local, openai, gemini,
claude, voyage, cohere) without any of those packages actually being
installed -- exactly the scenario a real user running `uv sync
--extra openai` only (no gemini/claude/voyage/cohere) would be in.

Coverage: each provider's env-var-driven configuration (model names,
API keys, defaults), the ValueError raised for an unrecognized
provider name, and the RuntimeError with an actionable `uv sync
--extra ...` message raised when a selected provider's package isn't
installed.
"""
import sys
from unittest.mock import MagicMock

import pytest

from src.rag.providers import (
    get_llm_provider,
    get_embedding_provider,
    get_query_rewrite_llm_provider,
    get_reranker_provider,
)

pytestmark = pytest.mark.unit


def _install_fake_module(monkeypatch, module_path: str, **attrs):
    """
    Installs a fake module into sys.modules so `from module_path
    import X` succeeds inside providers.py without the real package
    being installed. attrs become attributes (typically MagicMock
    classes) on the fake module.
    """
    fake_module = MagicMock()
    for name, value in attrs.items():
        setattr(fake_module, name, value)
    monkeypatch.setitem(sys.modules, module_path, fake_module)
    return fake_module


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


def test_get_llm_provider_local_uses_ollama_env_defaults(monkeypatch):
    fake_ollama_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.llms.ollama", Ollama=fake_ollama_cls)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)

    get_llm_provider("local")

    _, kwargs = fake_ollama_cls.call_args
    assert kwargs["model"] == "llama3"
    assert kwargs["base_url"] == "http://localhost:11434"
    assert kwargs["temperature"] == 0.0


def test_get_llm_provider_local_respects_env_overrides(monkeypatch):
    fake_ollama_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.llms.ollama", Ollama=fake_ollama_cls)
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")
    monkeypatch.setenv("OLLAMA_URL", "http://custom-host:1234")

    get_llm_provider("local")

    _, kwargs = fake_ollama_cls.call_args
    assert kwargs["model"] == "custom-model"
    assert kwargs["base_url"] == "http://custom-host:1234"


def test_get_llm_provider_openai_uses_env_defaults(monkeypatch):
    fake_openai_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.llms.openai", OpenAI=fake_openai_cls)
    monkeypatch.delenv("OPENAI_LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    get_llm_provider("openai")

    _, kwargs = fake_openai_cls.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["api_key"] == "sk-test-key"
    assert kwargs["temperature"] == 0.0


def test_get_llm_provider_gemini_uses_env_defaults(monkeypatch):
    fake_gemini_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.llms.google_genai", GoogleGenAI=fake_gemini_cls)
    monkeypatch.delenv("GEMINI_LLM_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    get_llm_provider("gemini")

    _, kwargs = fake_gemini_cls.call_args
    assert kwargs["model"] == "gemini-2.0-flash"
    assert kwargs["api_key"] == "gemini-test-key"


def test_get_llm_provider_claude_uses_env_defaults(monkeypatch):
    fake_anthropic_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.llms.anthropic", Anthropic=fake_anthropic_cls)
    monkeypatch.delenv("CLAUDE_LLM_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")

    get_llm_provider("claude")

    _, kwargs = fake_anthropic_cls.call_args
    assert kwargs["model"] == "claude-3-5-haiku-20241022"
    assert kwargs["api_key"] == "anthropic-test-key"


def test_get_llm_provider_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm_provider("not-a-real-provider")


def test_get_llm_provider_reads_env_var_when_no_argument_given(monkeypatch):
    fake_ollama_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.llms.ollama", Ollama=fake_ollama_cls)
    monkeypatch.setenv("LLM_PROVIDER", "local")

    get_llm_provider()  # no explicit provider argument

    fake_ollama_cls.assert_called_once()


def test_get_llm_provider_missing_openai_package_raises_actionable_runtime_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "llama_index.llms.openai", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _raise_import_error_for("llama_index.llms.openai"),
    )

    with pytest.raises(RuntimeError, match="uv sync --extra openai"):
        get_llm_provider("openai")


def _raise_import_error_for(blocked_module: str):
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == blocked_module or name.startswith(blocked_module + "."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    return _fake_import


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------


def test_get_embedding_provider_local_uses_env_defaults(monkeypatch):
    fake_embed_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.embeddings.ollama", OllamaEmbedding=fake_embed_cls)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    get_embedding_provider("local")

    _, kwargs = fake_embed_cls.call_args
    assert kwargs["model_name"] == "nomic-embed-text"


def test_get_embedding_provider_openai_uses_env_defaults(monkeypatch):
    fake_embed_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.embeddings.openai", OpenAIEmbedding=fake_embed_cls)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    get_embedding_provider("openai")

    _, kwargs = fake_embed_cls.call_args
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["api_key"] == "sk-test-key"


def test_get_embedding_provider_gemini_uses_env_defaults(monkeypatch):
    fake_embed_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.embeddings.google_genai", GoogleGenAIEmbedding=fake_embed_cls)
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    get_embedding_provider("gemini")

    _, kwargs = fake_embed_cls.call_args
    assert kwargs["model_name"] == "text-embedding-004"


def test_get_embedding_provider_voyage_uses_env_defaults(monkeypatch):
    fake_embed_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.embeddings.voyageai", VoyageEmbedding=fake_embed_cls)
    monkeypatch.delenv("VOYAGE_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-test-key")

    get_embedding_provider("voyage")

    _, kwargs = fake_embed_cls.call_args
    assert kwargs["model_name"] == "voyage-3"
    assert kwargs["voyage_api_key"] == "voyage-test-key"


def test_get_embedding_provider_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        get_embedding_provider("not-a-real-provider")


# ---------------------------------------------------------------------------
# Query rewrite LLM provider
# ---------------------------------------------------------------------------


def test_get_query_rewrite_llm_provider_local_uses_small_model_default(monkeypatch):
    fake_ollama_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.llms.ollama", Ollama=fake_ollama_cls)
    monkeypatch.delenv("QUERY_REWRITE_MODEL", raising=False)

    get_query_rewrite_llm_provider("local")

    _, kwargs = fake_ollama_cls.call_args
    assert kwargs["model"] == "qwen2.5:0.5b"


def test_get_query_rewrite_llm_provider_openai_uses_small_model_default(monkeypatch):
    fake_openai_cls = MagicMock()
    _install_fake_module(monkeypatch, "llama_index.llms.openai", OpenAI=fake_openai_cls)
    monkeypatch.delenv("OPENAI_QUERY_REWRITE_MODEL", raising=False)

    get_query_rewrite_llm_provider("openai")

    _, kwargs = fake_openai_cls.call_args
    assert kwargs["model"] == "gpt-4o-mini"


def test_get_query_rewrite_llm_provider_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown QUERY_REWRITE_PROVIDER"):
        get_query_rewrite_llm_provider("not-a-real-provider")


# ---------------------------------------------------------------------------
# Reranker provider
# ---------------------------------------------------------------------------


def test_get_reranker_provider_local_returns_reranker_instance(monkeypatch):
    fake_reranker_cls = MagicMock()
    monkeypatch.setattr("src.rag.reranker.Reranker", fake_reranker_cls)

    get_reranker_provider("local", top_n=4)

    fake_reranker_cls.assert_called_once_with(top_n=4)


def test_get_reranker_provider_cohere_uses_env_defaults(monkeypatch):
    fake_cohere_cls = MagicMock()
    _install_fake_module(
        monkeypatch, "llama_index.postprocessor.cohere_rerank", CohereRerank=fake_cohere_cls
    )
    monkeypatch.delenv("COHERE_RERANK_MODEL", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "cohere-test-key")

    get_reranker_provider("cohere", top_n=4)

    _, kwargs = fake_cohere_cls.call_args
    assert kwargs["model"] == "rerank-v3.5"
    assert kwargs["api_key"] == "cohere-test-key"
    assert kwargs["top_n"] == 4


def test_get_reranker_provider_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unknown RERANKER_PROVIDER"):
        get_reranker_provider("not-a-real-provider")


def test_get_reranker_provider_missing_cohere_package_raises_actionable_runtime_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "llama_index.postprocessor.cohere_rerank", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _raise_import_error_for("llama_index.postprocessor.cohere_rerank"),
    )

    with pytest.raises(RuntimeError, match="uv sync --extra cohere"):
        get_reranker_provider("cohere")