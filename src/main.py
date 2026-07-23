import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.orchestrator import RAGOrchestrator
from src.ingestion.drive_client import GoogleDriveClient
from src.storage.metadata_store import DynamicSyncManager
from src.utils.logger import get_logger

logger = get_logger("GDrive-RAG")

def format_citations(source_nodes):
    """Extrating metadata from Chunks used to answer."""
    citations = []
    for node in source_nodes:
        meta = node.node.metadata
        file_name = meta.get("file_name", "Unknown File")
        score = node.score if node.score else 0.0

        citation_text = f"{file_name} (Score: {score:.2f})"
        if citation_text not in citations:
            citations.append(citation_text)
            
    return citations

def sync_knowledge_base():
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    orchestrator = RAGOrchestrator()
    client = GoogleDriveClient()

    if folder_id is None:
        logger.info("Environment variable GOOGLE_DRIVE_FOLDER_ID is not set.")
        return

    files = client.list_files_in_folder(folder_id)

    sync_engine = DynamicSyncManager(orchestrator.index, orchestrator.db_manager)
    sync_engine.sync_with_drive(files, client)

def main():
    logger.info("Loading the RAG Engine system...")
    orchestrator = RAGOrchestrator()

    logger.info("Checking status of database...")
    sync_knowledge_base()

    query_engine = orchestrator.get_query_engine()

    logger.info("\n The system is now Ready (type 'exit' to exit the program)")
    logger.info("="*60)

    while True:
        query = input("\n User: ")
        if query.strip().lower() == 'exit':
            break

        if not query.strip():
            continue

        logger.info("Searching for documents and processing responses...")

        response = query_engine.query(query)

        answer_text = getattr(response, "response", response)
        logger.info(f"\nAI : {answer_text}")

        citations = format_citations(response.source_nodes)
        if citations:
            logger.info("\n Source")
            for citation in citations:
                logger.info(f"   {citation}")
        logger.info("-" * 60)

if __name__ == "__main__":
    main()