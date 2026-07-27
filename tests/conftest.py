"""
Shared pytest fixtures.

IMPORTANT: this file (tests/conftest.py) applies to BOTH tests/unit/
and tests/integration/, because pytest conftest.py fixtures cascade
into all subdirectories. Anything placed here with autouse=True runs
for integration tests too -- which is exactly the bug that used to
live here: an autouse env-faking fixture was overriding the real
GCP_CREDENTIALS_PATH from .env, breaking integration tests that need
real credentials.

Rule for this file: fixtures here must be opt-in (no autouse=True),
because they are shared by both unit and integration tests, and
integration tests intentionally want the real environment.

The env-isolation fixture that unit tests need lives in
tests/unit/conftest.py instead, scoped so it only affects that
directory.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

# Make `src` importable the same way src/main.py does at runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Google Drive mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_drive_files():
    """A representative page of Drive `files.list` results."""
    return [
        {
            "id": "file-001",
            "name": "handbook.pdf",
            "mimeType": "application/pdf",
            "modifiedTime": "2026-01-10T12:00:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/file-001/view",
        },
        {
            "id": "file-002",
            "name": "notes.txt",
            "mimeType": "text/plain",
            "modifiedTime": "2026-01-11T09:30:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/file-002/view",
        },
    ]


@pytest.fixture
def mock_drive_service(sample_drive_files):
    """
    A MagicMock standing in for the `googleapiclient.discovery.build(...)`
    service object, wired for the calls GoogleDriveClient makes:
    `.files().list(...).execute()`, `.files().get_media(...)`,
    `.changes().getStartPageToken().execute()`, and
    `.changes().list(...).execute()`.

    Changes API responses default to "no changes, single page" so
    tests that don't care about the Changes API still get a sane
    default; tests that do care override
    `service.changes().list.return_value.execute.side_effect`.
    """
    service = MagicMock()
    service.files.return_value.list.return_value.execute.return_value = {
        "files": sample_drive_files
    }
    # get_media() result is only used by MediaIoBaseDownload, which we
    # patch separately in the drive_client tests rather than here.
    service.files.return_value.get_media.return_value = MagicMock()

    service.changes.return_value.getStartPageToken.return_value.execute.return_value = {
        "startPageToken": "start-token-000"
    }
    service.changes.return_value.list.return_value.execute.return_value = {
        "newStartPageToken": "start-token-000",
        "changes": [],
    }
    return service


@pytest.fixture
def mock_service_account_credentials(monkeypatch):
    """Patch service_account.Credentials.from_service_account_file so
    GoogleDriveClient.__init__ never touches a real key file."""
    fake_creds = MagicMock()
    monkeypatch.setattr(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        MagicMock(return_value=fake_creds),
    )
    return fake_creds


@pytest.fixture
def drive_client(monkeypatch, mock_service_account_credentials, mock_drive_service):
    """A GoogleDriveClient wired to mocks end to end."""
    from src.ingestion.drive_client import GoogleDriveClient

    monkeypatch.setattr(
        "src.ingestion.drive_client.build",
        MagicMock(return_value=mock_drive_service),
    )
    return GoogleDriveClient()


# ---------------------------------------------------------------------------
# PyMuPDF (fitz) mocks
# ---------------------------------------------------------------------------

def _make_fake_page(text: str):
    page = MagicMock()
    page.get_text.return_value = text
    return page


@pytest.fixture
def fake_pdf_page_factory():
    """Factory so a test can build a fake fitz page with arbitrary text."""
    return _make_fake_page


@pytest.fixture
def mock_fitz_document(monkeypatch):
    """
    Patches fitz.open(...) to return a fake multi-page document without
    needing pymupdf installed or real PDF bytes. Tests can set
    `mock_fitz_document.pages = [...]` before calling the parser.
    """
    state = {"pages": [_make_fake_page("Default page text.")]}

    class _FakeReader:
        def __len__(self):
            return len(state["pages"])

        def load_page(self, idx):
            return state["pages"][idx]

    fake_module = MagicMock()
    fake_module.open.return_value = _FakeReader()
    monkeypatch.setattr("src.ingestion.document_parser.fitz", fake_module)

    def _set_pages(texts):
        state["pages"] = [_make_fake_page(t) for t in texts]

    fake_module.set_pages = _set_pages
    return fake_module


# ---------------------------------------------------------------------------
# Ollama / LlamaIndex Settings mocks
#
# llama_index.core.Settings.llm / .embed_model setters assert
# isinstance(value, LLM) / isinstance(value, BaseEmbedding). A bare
# MagicMock() fails that check, so the constructor mock returns a real
# MockLLM / MockEmbedding instance (llama_index's own test doubles,
# which ARE real subclasses) instead of a MagicMock instance. The
# constructor call itself (Ollama(...), OllamaEmbedding(...)) is still
# a MagicMock, so call_args/call_count assertions on *that* still work
# -- only the returned object is swapped for something Settings accepts.
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ollama_llm(monkeypatch):
    from llama_index.core.llms import MockLLM

    real_llm = MockLLM()
    ollama_ctor = MagicMock(name="Ollama", return_value=real_llm)
    monkeypatch.setattr("src.rag.orchestrator.Ollama", ollama_ctor)
    return real_llm


@pytest.fixture
def mock_ollama_embedding(monkeypatch):
    from llama_index.core.embeddings import MockEmbedding

    real_embed = MockEmbedding(embed_dim=8)
    ollama_embed_ctor = MagicMock(name="OllamaEmbedding", return_value=real_embed)
    monkeypatch.setattr("src.rag.orchestrator.OllamaEmbedding", ollama_embed_ctor)
    return real_embed


@pytest.fixture
def mock_vector_db_manager(monkeypatch):
    """Patches VectorDBManager so RAGOrchestrator never touches a real
    ChromaDB path on disk."""
    fake_manager = MagicMock(name="VectorDBManager")
    fake_manager.load_index.return_value = MagicMock(name="VectorStoreIndex")
    monkeypatch.setattr(
        "src.rag.orchestrator.VectorDBManager",
        MagicMock(return_value=fake_manager),
    )
    return fake_manager
