"""
Unit tests for src/storage/sync_lock.py (SyncLock)

Uses a real temp file (tmp_path) rather than mocking the filesystem --
this is a genuinely filesystem-based mechanism, so exercising it
against real files is the only way to trust it.
"""
import time

import pytest

from src.storage.sync_lock import SyncLock

pytestmark = pytest.mark.unit


@pytest.fixture
def lock(tmp_path):
    return SyncLock(lock_path=str(tmp_path / "test.lock"), stale_after_seconds=1.0)


def test_not_locked_when_no_lock_file_exists(lock):
    assert lock.is_locked() is False


def test_locked_after_acquire(lock):
    lock.acquire()

    assert lock.is_locked() is True


def test_not_locked_after_release(lock):
    lock.acquire()
    lock.release()

    assert lock.is_locked() is False


def test_release_without_acquire_does_not_raise(lock):
    lock.release()  # should be a no-op, not FileNotFoundError


def test_acquire_writes_pid_to_lock_file(lock, tmp_path):
    import os

    lock.acquire()

    with open(tmp_path / "test.lock") as f:
        content = f.read()
    assert content == str(os.getpid())


def test_stale_lock_is_treated_as_unlocked(lock):
    lock.acquire()
    time.sleep(1.2)  # exceed stale_after_seconds=1.0

    assert lock.is_locked() is False


def test_stale_lock_is_removed_when_detected(lock, tmp_path):
    lock.acquire()
    time.sleep(1.2)

    lock.is_locked()  # triggers the stale cleanup

    assert not (tmp_path / "test.lock").exists()


def test_refresh_prevents_lock_from_going_stale(lock):
    lock.acquire()
    time.sleep(0.6)
    lock.refresh()
    time.sleep(0.6)  # total elapsed 1.2s, but refreshed at 0.6s so age is only 0.6s

    assert lock.is_locked() is True


def test_refresh_without_acquire_does_not_raise(lock):
    lock.refresh()  # should be a no-op if no lock file exists


def test_wait_until_free_returns_true_immediately_when_unlocked(lock):
    result = lock.wait_until_free(timeout_seconds=5.0)

    assert result is True


def test_wait_until_free_returns_true_once_lock_released(lock):
    import threading

    lock.acquire()

    def release_soon():
        time.sleep(0.1)
        lock.release()

    threading.Thread(target=release_soon).start()

    result = lock.wait_until_free(timeout_seconds=2.0, poll_interval=0.05)

    assert result is True


def test_wait_until_free_times_out_if_never_released(lock):
    lock.acquire()

    result = lock.wait_until_free(timeout_seconds=0.3, poll_interval=0.05)

    assert result is False
