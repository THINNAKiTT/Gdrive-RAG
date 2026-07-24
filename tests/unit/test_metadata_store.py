"""
Unit tests for src/storage/metadata_store.py (DynamicSyncManager)

These tests pin down the sync/diff logic against a fake Chroma
collection and a fake drive client, with document_parser's fitz
dependency mocked out.
"""
from unittest.mock import MagicMock

import pytest

from src.storage.metadata_store import DynamicSyncManager

pytestmark = pytest.mark.unit


def make_db_manager(existing_metadatas):
    """Build a fake db_manager exposing a .chroma_collection with
    .get/.delete, matching what DynamicSyncManager expects."""
    db_manager = MagicMock()
    db_manager.chroma_collection.get.return_value = {"metadatas": existing_metadatas}
    return db_manager


def make_drive_client(file_bytes=b"%PDF-fake-bytes"):
    client = MagicMock()
    client.download_files.return_value = file_bytes
    return client


def test_new_file_is_ingested_and_indexed(mock_fitz_document):
    mock_fitz_document.set_pages(["New file content."])
    index = MagicMock()
    db_manager = make_db_manager(existing_metadatas=[])
    sync = DynamicSyncManager(index=index, db_manager=db_manager)

    drive_files = [{
        "id": "file-001",
        "name": "handbook.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-10T12:00:00.000Z",
        "webViewLink": "https://drive.google.com/file/d/file-001/view",
    }]

    sync.sync_with_drive(drive_files, make_drive_client())

    index.insert.assert_called_once()
    inserted_doc = index.insert.call_args[0][0]
    assert inserted_doc.metadata["file_id"] == "file-001"


def test_new_file_gets_modified_time_stamped_on_insert(mock_fitz_document):
    """
    Regression test: index_document() (used for the *initial* bulk
    ingest) never stamps modified_time, so freshly-synced docs must at
    least get it stamped here, or every subsequent sync will treat
    them as "new" forever.
    """
    mock_fitz_document.set_pages(["Content."])
    index = MagicMock()
    db_manager = make_db_manager(existing_metadatas=[])
    sync = DynamicSyncManager(index=index, db_manager=db_manager)

    drive_files = [{
        "id": "file-001",
        "name": "handbook.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-10T12:00:00.000Z",
        "webViewLink": "https://drive.google.com/file/d/file-001/view",
    }]

    sync.sync_with_drive(drive_files, make_drive_client())

    inserted_doc = index.insert.call_args[0][0]
    assert inserted_doc.metadata["modified_time"] == "2026-01-10T12:00:00.000Z"


def test_unchanged_file_is_not_reingested(mock_fitz_document):
    index = MagicMock()
    db_manager = make_db_manager(existing_metadatas=[
        {"file_id": "file-001", "modified_time": "2026-01-10T12:00:00.000Z"}
    ])
    sync = DynamicSyncManager(index=index, db_manager=db_manager)

    drive_files = [{
        "id": "file-001",
        "name": "handbook.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-10T12:00:00.000Z",  # same as stored
        "webViewLink": "https://drive.google.com/file/d/file-001/view",
    }]

    sync.sync_with_drive(drive_files, make_drive_client())

    index.insert.assert_not_called()
    db_manager.chroma_collection.delete.assert_not_called()


def test_modified_file_is_deleted_then_reingested(mock_fitz_document):
    mock_fitz_document.set_pages(["Updated content."])
    index = MagicMock()
    db_manager = make_db_manager(existing_metadatas=[
        {"file_id": "file-001", "modified_time": "2026-01-01T00:00:00.000Z"}
    ])
    sync = DynamicSyncManager(index=index, db_manager=db_manager)

    drive_files = [{
        "id": "file-001",
        "name": "handbook.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-10T12:00:00.000Z",  # newer than stored
        "webViewLink": "https://drive.google.com/file/d/file-001/view",
    }]

    sync.sync_with_drive(drive_files, make_drive_client())

    db_manager.chroma_collection.delete.assert_called_once_with(
        where={"file_id": "file-001"}
    )
    index.insert.assert_called_once()


def test_file_removed_from_drive_is_deleted_from_index(mock_fitz_document):
    index = MagicMock()
    db_manager = make_db_manager(existing_metadatas=[
        {"file_id": "file-gone", "modified_time": "2026-01-01T00:00:00.000Z"}
    ])
    sync = DynamicSyncManager(index=index, db_manager=db_manager)

    sync.sync_with_drive(current_drive_files=[], drive_client=make_drive_client())

    db_manager.chroma_collection.delete.assert_called_once_with(
        where={"file_id": "file-gone"}
    )
    index.insert.assert_not_called()


def test_unsupported_mimetype_is_skipped(mock_fitz_document):
    index = MagicMock()
    db_manager = make_db_manager(existing_metadatas=[])
    sync = DynamicSyncManager(index=index, db_manager=db_manager)

    drive_files = [{
        "id": "file-005",
        "name": "spreadsheet.xlsx",
        "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "modifiedTime": "2026-01-10T12:00:00.000Z",
        "webViewLink": "https://drive.google.com/file/d/file-005/view",
    }]

    sync.sync_with_drive(drive_files, make_drive_client())

    index.insert.assert_not_called()


def test_chroma_get_failure_falls_back_to_empty_local_state(mock_fitz_document):
    """If chroma_collection.get() raises (e.g. empty/uninitialized
    collection), sync should treat everything as new rather than crash."""
    mock_fitz_document.set_pages(["Content."])
    index = MagicMock()
    db_manager = MagicMock()
    db_manager.chroma_collection.get.side_effect = Exception("collection not found")
    sync = DynamicSyncManager(index=index, db_manager=db_manager)

    drive_files = [{
        "id": "file-001",
        "name": "handbook.pdf",
        "mimeType": "application/pdf",
        "modifiedTime": "2026-01-10T12:00:00.000Z",
        "webViewLink": "https://drive.google.com/file/d/file-001/view",
    }]

    sync.sync_with_drive(drive_files, make_drive_client())

    index.insert.assert_called_once()
