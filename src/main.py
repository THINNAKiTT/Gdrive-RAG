import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.orchestrator import RAGOrchestrator
from src.ingestion.drive_client import GoogleDriveClient
from src.ingestion.document_parser import DocumentParser
from src.storage.vector_db import VectorDBManager

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

def init_knowledge_base():
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    db_manager = VectorDBManager()

    if folder_id is None:
        print("Environment variable GOOGLE_DRIVE_FOLDER_ID is not set.")
        return
    
    db_files = os.listdir(db_manager.db_path) if os.path.exists(db_manager.db_path) else []
    has_vector_dir = any(os.path.isdir(os.path.join(db_manager.db_path, f)) for f in db_files)
    
    if not has_vector_dir:
        print("Fetching Document from Google Drive...")
        
        client = GoogleDriveClient()
        files = client.list_files_in_folder(folder_id)
        
        parsed_documents = []

        for file in files:
            if "application/pdf" in file['mimeType']:
                print(f"Downloading and extracting : {file['name']}")
                file_bytes = client.download_files(file['id'])
                doc = DocumentParser.parse_pdf(file_bytes, file['name'], file['id'])
                parsed_documents.append(doc)
                
        if parsed_documents:
            db_manager.index_document(parsed_documents)
            print("The vector database has been successfully created.")
        else:
            print("PDF not found")

def main():
    print("Loading the RAG Engine system...")
    orchestrator = RAGOrchestrator()

    print("Checking status of database...")
    init_knowledge_base()

    query_engine = orchestrator.get_query_engine()

    print("\n The system is now Ready (type 'exit' to exit the program)")
    print("="*60)

    while True:
        query = input("\n User: ")
        if query.strip().lower() == 'exit':
            break

        if not query.strip():
            continue

        print("Searching for documents and processing responses...")

        response = query_engine.query(query)

        answer_text = getattr(response, "response", response)
        print(f"\nAI : {answer_text}")

        citations = format_citations(response.source_nodes)
        if citations:
            print("\n Source")
            for citation in citations:
                print(f"   {citation}")
        print("-" * 60)

if __name__ == "__main__":
    main()