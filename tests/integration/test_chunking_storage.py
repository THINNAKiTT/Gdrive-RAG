"""
Integration test: real Google Drive + real Ollama + real ChromaDB
(ephemeral, tmp_path-backed) end-to-end pipeline.

Requires:
  - .env with GCP_CREDENTIALS_PATH / GOOGLE_DRIVE_folder_id pointing at
    a live folder with at least one PDF
  - a running Ollama server (OLLAMA_URL) with the embedding + LLM
    models pulled

Run explicitly with:  pytest -m integration
"""
import os

import pytest
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from src.ingestion.drive_client import GoogleDriveClient
from src.ingestion.document_parser import DocumentParser
from src.storage.vector_db import VectorDBManager

load_dotenv()

pytestmark = pytest.mark.integration

folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

requires_real_drive_folder = pytest.mark.skipif(
    not folder_id, reason="GOOGLE_DRIVE_FOLDER_ID not set in environment"
)


@requires_real_drive_folder
def test_end_to_end_ingest_chunk_store_and_query(tmp_path):
    Settings.embed_model = OllamaEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        request_timeout=60.0,
    )
    Settings.llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        request_timeout=120.0,
    )

    client = GoogleDriveClient()
    assert folder_id is not None
    files = client.list_files_in_folder(folder_id)
    pdf_files = [f for f in files if f["mimeType"] == "application/pdf"]

    if not pdf_files:
        pytest.skip("No PDF files found in the configured Drive folder")

    all_docs = []
    for f in pdf_files:
        file_bytes = client.download_files(f["id"])
        # NOTE: original manual script called parse_pdf(bytes, name, id)
        # -- missing the required web_view_link arg -- and then
        # appended the returned *list* into another list, producing a
        # nested list that broke SentenceSplitter downstream. Fixed here:
        # flatten with extend(), and pass web_view_link.
        page_docs = DocumentParser.parse_file(
            file_bytes=file_bytes,
            file_name=f["name"],
            file_id=f["id"],
            web_view_link=f["webViewLink"],
            mimetype=f["mimeType"],
        )
        # Stamp modified_time here (index_document() itself does not --
        # see the docstring on VectorDBManager.index_document), so a
        # later DynamicSyncManager.sync_with_drive() call can correctly
        # recognize these as already-synced rather than re-ingesting
        # them on every cycle.
        for doc in page_docs:
            doc.metadata["modified_time"] = f["modifiedTime"]
        all_docs.extend(page_docs)

    if not all_docs:
        pytest.skip("PDFs found but none contained extractable text")

    db_manager = VectorDBManager(db_path=str(tmp_path / "chroma_db"))
    index = db_manager.index_document(all_docs)

    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query("What is this document about?")

    assert response is not None
    assert str(response).strip() != ""
