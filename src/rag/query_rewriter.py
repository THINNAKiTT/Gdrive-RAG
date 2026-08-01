import os

from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.utils.resilience import with_resilience, CircuitOpenError
from src.rag.prompt_templates import REWRITE_PROMPT_TEMPLATE

load_dotenv()

logger = get_logger("QueryRewriter")

class QueryRewriter:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        model_name = os.getenv("QUERY_REWRITE_MODEL", "qwen2.5:0.5b")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        self.llm = Ollama(
            model=model_name,
            base_url=ollama_url,
            temperature=0.0,
            request_timeout=20.0,
        )

    def _build_history_block(self, turns: list[dict]) -> str:
        if not turns:
            return "(no prior conversation)"

        lines = []
        for msg in turns:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    def rewrite(self, query: str, history: list[dict]) -> str:
        if not history:
            return query

        history_block = self._build_history_block(history)
        prompt = REWRITE_PROMPT_TEMPLATE.format(history_block=history_block, query=query)

        try:
            complete_with_resilience = with_resilience(self.llm.complete)
            response = complete_with_resilience(prompt)
            rewritten = str(response).strip()
        except CircuitOpenError as e:
            logger.warning(f"Query rewriter circuit breaker open, using original query: {e}")
            return query
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original query: {e}")
            return query

        if not rewritten:
            logger.warning("Query rewriter returned empty response, using original query.")
            return query

        if rewritten != query:
            logger.info(
                "Query rewritten for context.",
                extra={"original_query": query, "rewritten_query": rewritten},
            )

        return rewritten