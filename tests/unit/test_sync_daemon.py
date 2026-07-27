"""
Unit tests for src/sync_daemon.py

Covers run_once()'s two paths (initial full sync when no page token
is saved yet, incremental sync via the Changes API afterward) and the
page-token persistence contract that main()'s crash-recovery relies
on. Google Drive, the lock, and disk I/O for the page token file are
all faked so these tests need no real Drive account, no real lock
file races, and no real Ollama.
"""
import os
from unittest.mock import MagicMock

import pytest

import src.sync_daemon as sync_daemon

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_page_token_path(monkeypatch, tmp_path):
    """Point the module's on-disk page-token file at a tmp path so
    tests never touch a real ./chroma_db/.drive_page_token."""
    fake_path = str(tmp_path / "chroma_db" / ".drive_page_token")
    monkeypatch.setattr(sync_daemon, "PAGE_TOKEN_PATH", fake_path)
    return fake_path


@pytest.fixture
def fake_client():
    client = MagicMock()
    client.list_files_in_folder.return_value = [
        {"id": "file-001", "name": "a.pdf", "mimeType": "application/pdf",
         "modifiedTime": "2026-01-01T00:00:00.000Z", "webViewLink": ""}
    ]
    client.get_start_page_token.return_value = "start-token-000"
    client.list_changes.return_value = ([], set(), "token-001")
    return client


@pytest.fixture
def fake_sync_engine():
    return MagicMock()


@pytest.fixture
def fake_lock():
    lock = MagicMock()
    return lock


def test_run_once_does_initial_full_sync_when_no_page_token_saved(
    fake_client, fake_sync_engine, fake_lock
):
    sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    fake_client.list_files_in_folder.assert_called_once_with("folder-xyz")
    fake_sync_engine.sync_with_drive.assert_called_once()
    fake_sync_engine.sync_from_changes.assert_not_called()


def test_run_once_saves_start_page_token_after_initial_sync(
    fake_client, fake_sync_engine, fake_lock, _isolated_page_token_path
):
    sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    with open(_isolated_page_token_path) as f:
        saved = f.read().strip()
    assert saved == "start-token-000"


def test_run_once_initial_sync_acquires_and_releases_lock(
    fake_client, fake_sync_engine, fake_lock
):
    sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    fake_lock.acquire.assert_called_once()
    fake_lock.release.assert_called_once()


def test_run_once_initial_sync_passes_lock_refresh_as_on_progress(
    fake_client, fake_sync_engine, fake_lock
):
    sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    _, kwargs = fake_sync_engine.sync_with_drive.call_args
    assert kwargs["on_progress"] is fake_lock.refresh


def test_run_once_uses_incremental_sync_when_page_token_exists(
    fake_client, fake_sync_engine, fake_lock, _isolated_page_token_path
):
    os.makedirs(os.path.dirname(_isolated_page_token_path), exist_ok=True)
    with open(_isolated_page_token_path, "w") as f:
        f.write("existing-token")
    fake_client.list_changes.return_value = (
        [{"id": "file-002", "name": "b.pdf", "mimeType": "application/pdf",
          "modifiedTime": "2026-01-02T00:00:00.000Z", "webViewLink": ""}],
        set(),
        "new-token",
    )

    sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    fake_client.list_changes.assert_called_once_with("existing-token", "folder-xyz")
    fake_sync_engine.sync_from_changes.assert_called_once()
    fake_client.list_files_in_folder.assert_not_called()


def test_run_once_incremental_sync_saves_new_page_token(
    fake_client, fake_sync_engine, fake_lock, _isolated_page_token_path
):
    os.makedirs(os.path.dirname(_isolated_page_token_path), exist_ok=True)
    with open(_isolated_page_token_path, "w") as f:
        f.write("existing-token")
    fake_client.list_changes.return_value = ([], set(), "advanced-token")

    sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    with open(_isolated_page_token_path) as f:
        saved = f.read().strip()
    assert saved == "advanced-token"


def test_run_once_skips_lock_and_sync_when_no_changes(
    fake_client, fake_sync_engine, fake_lock, _isolated_page_token_path
):
    os.makedirs(os.path.dirname(_isolated_page_token_path), exist_ok=True)
    with open(_isolated_page_token_path, "w") as f:
        f.write("existing-token")
    fake_client.list_changes.return_value = ([], set(), "same-or-next-token")

    sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    fake_lock.acquire.assert_not_called()
    fake_lock.release.assert_not_called()
    fake_sync_engine.sync_from_changes.assert_not_called()


def test_run_once_incremental_sync_passes_lock_refresh_as_on_progress(
    fake_client, fake_sync_engine, fake_lock, _isolated_page_token_path
):
    os.makedirs(os.path.dirname(_isolated_page_token_path), exist_ok=True)
    with open(_isolated_page_token_path, "w") as f:
        f.write("existing-token")
    fake_client.list_changes.return_value = (
        [{"id": "file-002", "name": "b.pdf", "mimeType": "application/pdf",
          "modifiedTime": "2026-01-02T00:00:00.000Z", "webViewLink": ""}],
        set(),
        "new-token",
    )

    sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    _, kwargs = fake_sync_engine.sync_from_changes.call_args
    assert kwargs["on_progress"] is fake_lock.refresh


def test_run_once_does_not_advance_page_token_if_sync_raises(
    fake_client, fake_sync_engine, fake_lock, _isolated_page_token_path
):
    """
    If sync_from_changes() blows up partway through, the page token
    must NOT be advanced -- otherwise the next cycle would skip past
    changes we never actually finished applying. The caller (main())
    is responsible for catching the exception and retrying next
    interval with the same (un-advanced) token.
    """
    os.makedirs(os.path.dirname(_isolated_page_token_path), exist_ok=True)
    with open(_isolated_page_token_path, "w") as f:
        f.write("existing-token")
    fake_client.list_changes.return_value = (
        [{"id": "file-002", "name": "b.pdf", "mimeType": "application/pdf",
          "modifiedTime": "2026-01-02T00:00:00.000Z", "webViewLink": ""}],
        set(),
        "new-token",
    )
    fake_sync_engine.sync_from_changes.side_effect = RuntimeError("Ollama unreachable")

    with pytest.raises(RuntimeError):
        sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    with open(_isolated_page_token_path) as f:
        saved = f.read().strip()
    assert saved == "existing-token", "page token must not advance past a failed sync"


def test_run_once_releases_lock_even_if_sync_raises(
    fake_client, fake_sync_engine, fake_lock, _isolated_page_token_path
):
    os.makedirs(os.path.dirname(_isolated_page_token_path), exist_ok=True)
    with open(_isolated_page_token_path, "w") as f:
        f.write("existing-token")
    fake_client.list_changes.return_value = (
        [{"id": "file-002", "name": "b.pdf", "mimeType": "application/pdf",
          "modifiedTime": "2026-01-02T00:00:00.000Z", "webViewLink": ""}],
        set(),
        "new-token",
    )
    fake_sync_engine.sync_from_changes.side_effect = RuntimeError("Ollama unreachable")

    with pytest.raises(RuntimeError):
        sync_daemon.run_once(fake_client, fake_sync_engine, fake_lock, "folder-xyz")

    fake_lock.release.assert_called_once()
