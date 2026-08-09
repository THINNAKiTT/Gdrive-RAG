import os
from typing import Optional
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()

logger = get_logger("Providers")

def _missing_dependency(provider: str, extra: str, package: str) -> RuntimeError:
    return RuntimeError(
        f"Provider '{provider}' is selected but its SDK ('{package}') "
        f"is not installed. Install it with:\n\n"
        f"  uv sync --extra {extra}\n\n"
        f"(or `pip install -e '.[{extra}]'`)"
    )

def get_llm_provider(provider: Optional[str] = None):
    provider = provider or os.getenv("LLM_PROVIDER", "local")

    if provider == "local":
        try:
            from llama_index.llms.ollama import Ollama # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("local", "ollama", "llama-index-llms-ollama")
        return Ollama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            temperature=0.0,
            request_timeout=120.0,
        )

    elif provider == "openai":
        try:
            from llama_index.llms.openai import OpenAI # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("openai", "openai", "llama-index-llms-openai")
        return OpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
        )

    elif provider == "gemini":
        try:
            from llama_index.llms.google_genai import GoogleGenAI #pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("gemini", "gemini", "llama-index-llms-google-genai")
        return GoogleGenAI(
            model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
        )

    elif provider == "claude":
        try:
            from llama_index.llms.anthropic import Anthropic # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("claude", "claude", "llama-index-llms-anthropic")
        return Anthropic(
            model=os.getenv("CLAUDE_LLM_MODEL", "claude-3-5-haiku-20241022"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Must be one of: "
            "local, openai, gemini, claude."
        )

def get_embedding_provider(provider: Optional[str] = None):
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "local")

    if provider == "local":
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("local", "ollama", "llama-index-embeddings-ollama")
        return OllamaEmbedding(
            model_name=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            request_timeout=60.0,
        )

    elif provider == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("openai", "openai", "llama-index-embeddings-openai")
        return OpenAIEmbedding(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
            embedding_config={"output_dimensionality": 768},
        )

    elif provider == "gemini":
        try:
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("gemini", "gemini", "llama-index-embeddings-google-genai")
        return GoogleGenAIEmbedding(
            model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
            api_key=os.getenv("GEMINI_API_KEY"),
            embedding_config={"output_dimensionality": 768},
        )

    elif provider == "voyage":
        try:
            from llama_index.embeddings.voyageai import VoyageEmbedding # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("voyage", "voyage", "llama-index-embeddings-voyageai")
        return VoyageEmbedding(
            model_name=os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3"),
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
            embedding_config={"output_dimensionality": 768},
        )

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider}'. Must be one of: "
            "local, openai, gemini, voyage."
        )

def get_query_rewrite_llm_provider(provider: Optional[str] = None):
    provider = provider or os.getenv("QUERY_REWRITE_PROVIDER", "local")

    if provider == "local":
        try:
            from llama_index.llms.ollama import Ollama # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("local", "ollama", "llama-index-llms-ollama")
        return Ollama(
            model=os.getenv("QUERY_REWRITE_MODEL", "qwen2.5:0.5b"),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            temperature=0.0,
            request_timeout=20.0,
        )

    elif provider == "openai":
        try:
            from llama_index.llms.openai import OpenAI # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("openai", "openai", "llama-index-llms-openai")
        return OpenAI(
            model=os.getenv("OPENAI_QUERY_REWRITE_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
        )

    elif provider == "gemini":
        try:
            from llama_index.llms.google_genai import GoogleGenAI # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("gemini", "gemini", "llama-index-llms-google-genai")
        return GoogleGenAI(
            model=os.getenv("GEMINI_QUERY_REWRITE_MODEL", "gemini-2.0-flash-lite"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0,
        )

    elif provider == "claude":
        try:
            from llama_index.llms.anthropic import Anthropic # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("claude", "claude", "llama-index-llms-anthropic")
        return Anthropic(
            model=os.getenv("CLAUDE_QUERY_REWRITE_MODEL", "claude-3-5-haiku-20241022"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0,
        )

    else:
        raise ValueError(
            f"Unknown QUERY_REWRITE_PROVIDER '{provider}'. Must be one of: "
            "local, openai, gemini, claude."
        )

def get_reranker_provider(provider: Optional[str] = None, top_n: int = 4):
    provider = provider or os.getenv("RERANKER_PROVIDER", "local")

    if provider == "local":
        try:
            from src.rag.reranker import Reranker # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("local", "reranker-local", "sentence-transformers")
        return Reranker(top_n=top_n)

    elif provider == "cohere":
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank # pyright: ignore[reportMissingImports]
        except ImportError:
            raise _missing_dependency("cohere", "cohere", "llama-index-postprocessor-cohere-rerank")
        return CohereRerank(
            api_key=os.getenv("COHERE_API_KEY"),
            model=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            top_n=top_n,
        )

    else:
        raise ValueError(
            f"Unknown RERANKER_PROVIDER '{provider}'. Must be one of: "
            "local, cohere."
        )