import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from src.ingestion.drive_client import GoogleDriveClient
from src.ingestion.document_parser import DocumentParser
from src.storage.vector_db import VectorDBManager

load_dotenv()

def main():
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        request_timeout=60.0
    )

    Settings.llm = Ollama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        request_timeout=120.0
    )

    #======   If you want to use OPENAI   ======
    # openai_key = os.getenv("OPENAI_API_KEY")
    # if not openai_key:
    #     print("OPENAI_API_KEY not found")
    #     return
    
    if folder_id is None:
        print("Environment variable GOOGLE_DRIVE_FOLDER_ID is not set.")
        return
    
    print("Step 1: Fetching Document from Google Drive")
    client = GoogleDriveClient()
    files = client.list_files_in_folder(folder_id)

    parsed_documents = []

    for file in files:
        if "application/pdf" in file['mimeType']:
            print(f"downloading and extracting : {file['name']}...")
            file_bytes = client.download_files(file['id'])
            doc = DocumentParser.parse_pdf(file_bytes, file['name'], file['id'])
            parsed_documents.append(doc)

    if not parsed_documents:
        print("PDF not found")
        return
    
    print("\nStep 2: Start the chunking process and save to Vector DB. ")
    db_manager = VectorDBManager()
    index = db_manager.index_document(parsed_documents)

    print("\nStep 3: Searching for related content")
    query_engine = index.as_query_engine(similarity_top_k=2)

    test_query = "What is In-Context Reinforcement Learning?"
    print(f"Question: '{test_query}'")

    response = query_engine.query(test_query)
    print("\n📬 Results by AI:")
    print("-" * 50)
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    main()