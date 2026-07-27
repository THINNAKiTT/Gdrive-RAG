"""
Standalone sync daemon: polls Google Drive's Changes API and keeps
the vector store up to date, independent of the query-serving UI.

Run this as its own process, separate from `streamlit run ui/app.py`:

    python -m src.sync_daemon

It never lists the whole Drive folder repeatedly (unlike the old
sync-on-every-query approach in ui/app.py) -- after the first run it
only asks Drive "what changed since my last checkpoint", which is
cheap even with thousands of files in the watched folder.

State kept on disk (both under the ChromaDB directory by default, so
`rm -rf ./chroma_db` resets everything together):
  - <db_path>/.drive_page_token  -- Changes API checkpoint
  - ./.sync.lock                 -- see src/storage/sync_lock.py

This is the local-phase building block for what becomes a proper
background worker (Celery/RQ/Cloud Run job) in the Enterprise Cloud
version -- the polling loop here maps directly onto a scheduled task
later; only the lock and state storage need to change.
"""
import os
import time

from dotenv import load_dotenv

from src.ingestion.drive_client import GoogleDriveClient
from src.rag.orchestrator import RAGOrchestrator
from src.storage.metadata_store import DynamicSyncManager
from src.storage.sync_lock import SyncLock
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger("SyncDaemon")

POLL_INTERVAL_SECONDS = int(os.getenv("SYNC_POLL_INTERVAL_SECONDS", "30"))
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
PAGE_TOKEN_PATH = os.path.join(DB_PATH, ".drive_page_token")


def _load_page_token() -> str | None:
    if os.path.exists(PAGE_TOKEN_PATH):
        with open(PAGE_TOKEN_PATH, "r") as f:
            token = f.read().strip()
            return token or None
    return None


def _save_page_token(token: str):
    os.makedirs(os.path.dirname(PAGE_TOKEN_PATH), exist_ok=True)
    with open(PAGE_TOKEN_PATH, "w") as f:
        f.write(token)


def run_once(client: GoogleDriveClient, sync_engine: DynamicSyncManager, lock: SyncLock, folder_id: str):
    page_token = _load_page_token()

    if page_token is None:
        logger.info("No saved page token found -- running initial full sync.")
        lock.acquire()
        try:
            files = client.list_files_in_folder(folder_id)
            sync_engine.sync_with_drive(files, client, on_progress=lock.refresh)
        finally:
            lock.release()

        page_token = client.get_start_page_token()
        _save_page_token(page_token)
        logger.info("Initial full sync complete. Switching to incremental polling.")
        return

    changed_files, removed_file_ids, new_page_token = client.list_changes(page_token, folder_id)

    if changed_files or removed_file_ids:
        lock.acquire()
        try:
            sync_engine.sync_from_changes(
                changed_files, removed_file_ids, client, on_progress=lock.refresh
            )
        finally:
            lock.release()
    else:
        logger.debug("No changes since last check.")

    _save_page_token(new_page_token)


def main():
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        logger.error("GOOGLE_DRIVE_FOLDER_ID is not set -- sync daemon cannot start.")
        return

    logger.info(f"Starting sync daemon. Polling every {POLL_INTERVAL_SECONDS}s.")

    client = GoogleDriveClient()
    orchestrator = RAGOrchestrator()
    sync_engine = DynamicSyncManager(orchestrator.index, orchestrator.db_manager)
    lock = SyncLock()

    # Safety net: if a previous run of this daemon crashed while
    # holding the lock, don't start this run still holding it.
    lock.release()

    while True:
        try:
            run_once(client, sync_engine, lock, folder_id)
        except Exception:
            logger.exception("Sync cycle failed; will retry next interval.")
            lock.release()  # never leave the lock held across a crashed cycle
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
