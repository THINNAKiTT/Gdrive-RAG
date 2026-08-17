"""
Unit tests for src/storage/vector_db.py (VectorDBManager)

Uses a real (ephemeral, tmp_path-backed) ChromaDB instance rather than
mocking Chroma itself -- Chroma's PersistentClient against a temp dir
is fast and gives us real confidence that chunking + storage actually
works, without touching the developer's real ./chroma_db.

Chunking is configured via the global Settings.node_parser (set by
RAGOrchestrator in production; see orchestrator.py), NOT by
VectorDBManager itself -- llama-index's index.insert(), which is the
method actually used by DynamicSyncManager, only consults the global
setting. These tests set Settings.node_parser directly (bypassing
RAGOrchestrator) to pin down that VectorDBManager's storage layer
behaves correctly under the same chunking config production uses.

VectorStoreIndex(...) also resolves Settings.embed_model internally.
If nothing has set it, llama-index falls back to resolving an OpenAI
embedding model and blows up with a missing-API-key error. We pin
both Settings.embed_model and Settings.node_parser to test doubles
for the duration of this file and restore whatever was there before,
since Settings is a process-wide singleton shared across test files.
"""
import pytest
from llama_index.core import Document, Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter

from src.storage.vector_db import VectorDBManager

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_settings():
    previous_embed_model = Settings._embed_model
    previous_node_parser = Settings._node_parser
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    yield
    Settings._embed_model = previous_embed_model
    Settings._node_parser = previous_node_parser


@pytest.fixture
def db_manager(tmp_path):
    return VectorDBManager(db_path=str(tmp_path / "chroma_db"))


def _insert(db_manager, doc):
    """Mirrors the real ingestion path: DynamicSyncManager inserts
    one Document at a time into the index returned by load_index()."""
    index = db_manager.load_index()
    index.insert(doc)


def test_insert_chunks_and_stores(db_manager):
    doc = Document(
        # chunk_size=512 is measured in tokens, not characters -- use
        # enough repeats to comfortably clear that regardless of
        # tokenizer, rather than eyeballing a character count.
        text="Sentence one. " * 500,
        metadata={"file_id": "file-001", "file_name": "handbook.pdf", "modified_time": "2026-01-10T12:00:00.000Z"},
    )

    _insert(db_manager, doc)

    assert db_manager.chroma_collection.count() > 1


def test_insert_preserves_metadata_on_chunks(db_manager):
    doc = Document(
        text="Short content.",
        metadata={"file_id": "file-001", "file_name": "handbook.pdf", "modified_time": "2026-01-10T12:00:00.000Z"},
    )

    _insert(db_manager, doc)

    stored = db_manager.chroma_collection.get(include=["metadatas"])
    assert stored["metadatas"][0]["file_id"] == "file-001"


def test_insert_without_modified_time_stores_nothing_for_it(db_manager):
    doc = Document(
        text="Content missing modified_time.",
        metadata={"file_id": "file-002", "file_name": "no_timestamp.pdf"},
    )

    _insert(db_manager, doc)

    stored = db_manager.chroma_collection.get(include=["metadatas"])
    assert "modified_time" not in stored["metadatas"][0]


def test_load_index_returns_index_backed_by_same_collection(db_manager):
    doc = Document(
        text="Persisted content.",
        metadata={"file_id": "file-001", "file_name": "handbook.pdf", "modified_time": "2026-01-10T12:00:00.000Z"},
    )
    _insert(db_manager, doc)

    loaded_index = db_manager.load_index()

    assert loaded_index is not None
    assert db_manager.chroma_collection.count() > 0


def test_insert_respects_configured_chunk_size(db_manager):
    short_doc = Document(
        text="One short sentence.",
        metadata={"file_id": "file-short", "modified_time": "2026-01-10T12:00:00.000Z"},
    )
    _insert(db_manager, short_doc)
    assert db_manager.chroma_collection.count() == 1

    long_doc = Document(
        # chunk_size=512 is in tokens; use a wide margin over that
        # rather than a char-count guess (see test_insert_chunks_and_stores).
        text="Sentence. " * 1500,
        metadata={"file_id": "file-long", "modified_time": "2026-01-10T12:00:00.000Z"},
    )
    _insert(db_manager, long_doc)

    chunks_for_long_doc = db_manager.chroma_collection.count() - 1  # minus the short doc's 1 chunk
    assert chunks_for_long_doc > 1
