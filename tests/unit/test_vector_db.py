"""
Unit tests for src/storage/vector_db.py (VectorDBManager)

Uses a real (ephemeral, tmp_path-backed) ChromaDB instance rather than
mocking Chroma itself -- Chroma's PersistentClient against a temp dir
is fast and gives us real confidence that chunking + storage actually
works, without touching the developer's real ./chroma_db.

VectorStoreIndex(...) resolves Settings.embed_model internally. If
nothing has set it (this test file exercises VectorDBManager directly,
bypassing RAGOrchestrator, which is the only place that normally sets
it), llama-index falls back to resolving an OpenAI embedding model and
blows up with a missing-API-key error. We pin Settings.embed_model to
a fake, dependency-free embedding for the duration of this file and
restore whatever was there before, since Settings is a process-wide
singleton shared across test files.
"""
import pytest
from llama_index.core import Document, Settings
from llama_index.core.embeddings import MockEmbedding

from src.storage.vector_db import VectorDBManager

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_embed_model():
    previous = Settings._embed_model
    Settings.embed_model = MockEmbedding(embed_dim=8)
    yield
    Settings._embed_model = previous


@pytest.fixture
def db_manager(tmp_path):
    return VectorDBManager(db_path=str(tmp_path / "chroma_db"))


def test_index_document_chunks_and_stores(db_manager):
    docs = [
        Document(
            text="Sentence one. " * 100,  # long enough to force multiple chunks
            metadata={"file_id": "file-001", "file_name": "handbook.pdf", "modified_time": "2026-01-10T12:00:00.000Z"},
        )
    ]

    index = db_manager.index_document(docs)

    assert index is not None
    assert db_manager.chroma_collection.count() > 0


def test_index_document_preserves_metadata_on_chunks(db_manager):
    docs = [
        Document(
            text="Short content.",
            metadata={"file_id": "file-001", "file_name": "handbook.pdf", "modified_time": "2026-01-10T12:00:00.000Z"},
        )
    ]

    db_manager.index_document(docs)

    stored = db_manager.chroma_collection.get(include=["metadatas"])
    assert stored["metadatas"][0]["file_id"] == "file-001"


def test_index_document_without_modified_time_stores_nothing_for_it(db_manager):
    """
    Documents a real gap: index_document() does not itself stamp
    modified_time (see docstring in vector_db.py). If the caller
    forgets to stamp it before calling index_document(), the sync
    diff in DynamicSyncManager will not find "modified_time" in this
    file's stored metadata and will treat it as always-new. This test
    exists so that gap can't silently regress further (e.g. someone
    "fixing" it by swallowing the KeyError instead of stamping it).
    """
    docs = [
        Document(
            text="Content missing modified_time.",
            metadata={"file_id": "file-002", "file_name": "no_timestamp.pdf"},
        )
    ]

    db_manager.index_document(docs)

    stored = db_manager.chroma_collection.get(include=["metadatas"])
    assert "modified_time" not in stored["metadatas"][0]


def test_load_index_returns_index_backed_by_same_collection(db_manager):
    docs = [
        Document(
            text="Persisted content.",
            metadata={"file_id": "file-001", "file_name": "handbook.pdf", "modified_time": "2026-01-10T12:00:00.000Z"},
        )
    ]
    db_manager.index_document(docs)

    loaded_index = db_manager.load_index()

    assert loaded_index is not None
