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
