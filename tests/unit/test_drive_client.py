"""
Unit tests for src/ingestion/drive_client.py

All Google API calls are mocked -- no real service account, no network.
"""
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def test_list_files_in_folder_returns_files(drive_client, sample_drive_files):
    files = drive_client.list_files_in_folder("some-folder-id")

    assert files == sample_drive_files


def test_list_files_in_folder_builds_correct_query(drive_client, mock_drive_service):
    drive_client.list_files_in_folder("folder-xyz")

    _, kwargs = mock_drive_service.files.return_value.list.call_args
    assert kwargs["q"] == "'folder-xyz' in parents and trashed = false"


def test_list_files_in_folder_empty_result(monkeypatch, drive_client, mock_drive_service):
    mock_drive_service.files.return_value.list.return_value.execute.return_value = {}

    files = drive_client.list_files_in_folder("empty-folder")

    assert files == []


def test_download_files_returns_bytes_on_success(drive_client):
    fake_chunk_status = MagicMock()

    with patch("src.ingestion.drive_client.MediaIoBaseDownload") as mock_downloader_cls:
        instance = mock_downloader_cls.return_value

        def fake_next_chunk():
            return (fake_chunk_status, True)

        instance.next_chunk.side_effect = fake_next_chunk

        # Simulate bytes being written into the BytesIO stream passed to
        # MediaIoBaseDownload by writing through the constructor args.
        def fake_downloader_ctor(file_stream, request):
            file_stream.write(b"pdf-file-contents")
            return instance

        mock_downloader_cls.side_effect = fake_downloader_ctor

        result = drive_client.download_files("file-001")

    assert result == b"pdf-file-contents"


def test_download_files_returns_empty_bytes_on_error(drive_client):
    with patch("src.ingestion.drive_client.MediaIoBaseDownload") as mock_downloader_cls:
        mock_downloader_cls.side_effect = RuntimeError("network exploded")

        result = drive_client.download_files("file-broken")

    assert result == b""


# ---------------------------------------------------------------------------
# Changes API (get_start_page_token / list_changes)
# ---------------------------------------------------------------------------


def test_get_start_page_token_returns_token(drive_client):
    token = drive_client.get_start_page_token()

    assert token == "start-token-000"


def test_list_changes_no_changes_returns_empty_and_advances_token(
    drive_client, mock_drive_service
):
    mock_drive_service.changes.return_value.list.return_value.execute.return_value = {
        "newStartPageToken": "token-001",
        "changes": [],
    }

    changed, removed, new_token = drive_client.list_changes("token-000", "folder-xyz")

    assert changed == []
    assert removed == set()
    assert new_token == "token-001"


def test_list_changes_detects_modified_file_in_watched_folder(
    drive_client, mock_drive_service
):
    mock_drive_service.changes.return_value.list.return_value.execute.return_value = {
        "newStartPageToken": "token-002",
        "changes": [
            {
                "fileId": "file-001",
                "removed": False,
                "file": {
                    "id": "file-001",
                    "name": "handbook.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "2026-01-12T00:00:00.000Z",
                    "webViewLink": "https://drive.google.com/file/d/file-001/view",
                    "parents": ["folder-xyz"],
                    "trashed": False,
                },
            }
        ],
    }

    changed, removed, new_token = drive_client.list_changes("token-001", "folder-xyz")

    assert len(changed) == 1
    assert changed[0]["id"] == "file-001"
    assert removed == set()
    assert new_token == "token-002"


def test_list_changes_treats_trashed_file_as_removed(drive_client, mock_drive_service):
    mock_drive_service.changes.return_value.list.return_value.execute.return_value = {
        "newStartPageToken": "token-003",
        "changes": [
            {
                "fileId": "file-001",
                "removed": False,
                "file": {
                    "id": "file-001",
                    "name": "handbook.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "2026-01-12T00:00:00.000Z",
                    "webViewLink": "",
                    "parents": ["folder-xyz"],
                    "trashed": True,
                },
            }
        ],
    }

    changed, removed, new_token = drive_client.list_changes("token-002", "folder-xyz")

    assert changed == []
    assert removed == {"file-001"}


def test_list_changes_treats_hard_delete_as_removed(drive_client, mock_drive_service):
    """A hard delete has removed=True and no `file` payload at all."""
    mock_drive_service.changes.return_value.list.return_value.execute.return_value = {
        "newStartPageToken": "token-004",
        "changes": [
            {"fileId": "file-gone", "removed": True, "file": None},
        ],
    }

    changed, removed, new_token = drive_client.list_changes("token-003", "folder-xyz")

    assert changed == []
    assert removed == {"file-gone"}


def test_list_changes_ignores_file_moved_out_of_watched_folder(
    drive_client, mock_drive_service
):
    mock_drive_service.changes.return_value.list.return_value.execute.return_value = {
        "newStartPageToken": "token-005",
        "changes": [
            {
                "fileId": "file-001",
                "removed": False,
                "file": {
                    "id": "file-001",
                    "name": "handbook.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "2026-01-12T00:00:00.000Z",
                    "webViewLink": "",
                    "parents": ["some-other-folder"],
                    "trashed": False,
                },
            }
        ],
    }

    changed, removed, new_token = drive_client.list_changes("token-004", "folder-xyz")

    assert changed == []
    assert removed == {"file-001"}, (
        "a file moved out of the watched folder must be treated as "
        "removed from this knowledge base, even though it still "
        "exists on Drive elsewhere"
    )


def test_list_changes_paginates_across_multiple_pages(drive_client, mock_drive_service):
    page_one = {
        "nextPageToken": "page-2",
        "changes": [
            {
                "fileId": "file-001",
                "removed": False,
                "file": {
                    "id": "file-001",
                    "name": "a.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "2026-01-12T00:00:00.000Z",
                    "webViewLink": "",
                    "parents": ["folder-xyz"],
                    "trashed": False,
                },
            }
        ],
    }
    page_two = {
        "newStartPageToken": "token-final",
        "changes": [
            {
                "fileId": "file-002",
                "removed": False,
                "file": {
                    "id": "file-002",
                    "name": "b.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "2026-01-12T00:00:00.000Z",
                    "webViewLink": "",
                    "parents": ["folder-xyz"],
                    "trashed": False,
                },
            }
        ],
    }
    mock_drive_service.changes.return_value.list.return_value.execute.side_effect = [
        page_one,
        page_two,
    ]

    changed, removed, new_token = drive_client.list_changes("token-000", "folder-xyz")

    assert {f["id"] for f in changed} == {"file-001", "file-002"}
    assert new_token == "token-final"


def test_list_changes_never_reports_a_removed_file_as_changed(
    drive_client, mock_drive_service
):
    """
    Regression guard for the 'modified then trashed in the same
    batch' case: the file must end up ONLY in removed_file_ids, never
    in both lists.
    """
    mock_drive_service.changes.return_value.list.return_value.execute.return_value = {
        "newStartPageToken": "token-006",
        "changes": [
            {
                "fileId": "file-001",
                "removed": True,
                "file": {
                    "id": "file-001",
                    "name": "handbook.pdf",
                    "mimeType": "application/pdf",
                    "modifiedTime": "2026-01-12T00:00:00.000Z",
                    "webViewLink": "",
                    "parents": ["folder-xyz"],
                    "trashed": False,
                },
            }
        ],
    }

    changed, removed, new_token = drive_client.list_changes("token-005", "folder-xyz")

    changed_ids = {f["id"] for f in changed}
    assert changed_ids.isdisjoint(removed)
    assert removed == {"file-001"}


def test_list_changes_falls_back_to_input_token_if_api_never_returns_one(
    drive_client, mock_drive_service
):
    """
    Regression test: if Drive's response never includes
    newStartPageToken (shouldn't happen per the API docs, but the
    daemon runs unattended and must not crash on a malformed
    response), list_changes must not raise NameError -- it should
    fall back to something usable rather than crashing the sync loop.
    """
    mock_drive_service.changes.return_value.list.return_value.execute.return_value = {
        "changes": []
    }

    changed, removed, new_token = drive_client.list_changes("token-fallback", "folder-xyz")

    assert changed == []
    assert removed == set()
    assert new_token == "token-fallback"
