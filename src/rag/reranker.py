import os
from typing import List, Optional

from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from src.utils.logger import get_logger

logger = get_logger("Reranker")

class Reranker(BaseNodePostprocessor):
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    top_n: int = Field(default=4)
    _cross_encoder: Optional[CrossEncoder] = PrivateAttr(default=None)

    def __init__(self, model_name: Optional[str] = None, top_n: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.top_n = top_n
        logger.info(
            f"Loading reranker model: {self.model_name} "
            "(first run downloads and caches it)"
        )
        self._cross_encoder = CrossEncoder(self.model_name)

    @classmethod
    def class_name(cls) -> str:
        return "Reranker"

    def _postprocess_nodes(
        self, 
        nodes: List[NodeWithScore], 
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes[: self.top_n]

        query_str = query_bundle.query_str
        pairs = [(query_str, node.get_content()) for node in nodes]
        scores = self._cross_encoder.predict(pairs) # type: ignore

        for node, score in zip(nodes, scores):
            node.score = float(score)

        reranked = sorted(
            nodes,
            key=lambda n: n.score if n.score is not None else float("-inf"),
            reverse=True,
        )
        return reranked[: self.top_n]