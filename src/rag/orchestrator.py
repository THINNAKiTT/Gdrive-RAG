import os
from dotenv import load_dotenv

from llama_index.core import PromptTemplate, Settings
from llama_index.core.node_parser import SentenceSplitter

from src.storage.vector_db import VectorDBManager
from src.rag.prompt_templates import STRICT_RAG_PROMPT
from src.rag.providers import get_llm_provider, get_embedding_provider, get_reranker_provider

load_dotenv()

class RAGOrchestrator:
    def __init__(self):
        Settings.embed_model = get_embedding_provider()
        Settings.llm = get_llm_provider()
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        self.db_manager = VectorDBManager()
        self.index = self.db_manager.load_index()
        self.reranker = get_reranker_provider(top_n=4)

    def get_query_engine(self):
        text_qa_template = PromptTemplate(STRICT_RAG_PROMPT)

        query_engine = self.index.as_query_engine(
            similarity_top_k=15,
            node_postprocessors=[self.reranker],
            text_qa_template=text_qa_template
        )
        return query_engine
        