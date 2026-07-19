import os
from dotenv import load_dotenv

from llama_index.core import PromptTemplate, Settings

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from src.storage.vector_db import VectorDBManager
from src.rag.prompt_templates import STRICT_RAG_PROMPT

# Load .env
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "../../"))

load_dotenv()

class RAGOrchestrator:
    def __init__(self):

        # For Ollama
        embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        Settings.embed_model = OllamaEmbedding(
            model_name=embed_model,
            base_url=ollama_url,
            request_timeout=60.0
        )
        Settings.llm = Ollama(
            model=ollama_model, 
            base_url=ollama_url,
            temperature=0.0,
            request_timeout=120.0
        )

        #======   If you want to use OPENAI   ======
        # openai_key = os.getenv("OPENAI_API_KEY")
        # if not openai_key:
        #     print("OPENAI_API_KEY not found")
        #     return

        self.db_manager = VectorDBManager()
        self.index = self.db_manager.load_index()

    def get_query_engine(self):
        text_qa_template = PromptTemplate(STRICT_RAG_PROMPT)

        query_engine = self.index.as_query_engine(
            similarity_top_k=4,
            text_qa_template=text_qa_template
        )
        return query_engine
        