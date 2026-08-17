from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.utils.logger import get_logger
import chromadb

logger = get_logger("VectorDatabase")

class VectorDBManager:
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        self.db_client = chromadb.PersistentClient(path=self.db_path)

        self.chroma_collection = self.db_client.get_or_create_collection("gdrive_knowledge")

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def load_index(self) -> VectorStoreIndex:
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )