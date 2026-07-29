from llama_index.core import VectorStoreIndex
from src.utils.logger import get_logger
from src.ingestion.document_parser import DocumentParser, SUPPORTED_MIMETYPES
from src.utils.resilience import with_resilience, CircuitOpenError

logger = get_logger("MetadataSync")

class DynamicSyncManager:
    def __init__(self, index: VectorStoreIndex, db_manager):
        self.index = index
        self.db_manager = db_manager

    def sync_with_drive(self, current_drive_files: list, drive_client, on_progress=None):
        """Compares local index metadata with active Google Drive state."""

        # Fetch current vector storage state
        try:
            stored_data = self.db_manager.chroma_collection.get(include=["metadatas"])
            indexed_meta = stored_data.get("metadatas", [])
        except Exception:
            indexed_meta = []

        local_files = {}
        for meta in indexed_meta:
            if "file_id" in meta and "modified_time" in meta:
                local_files[meta["file_id"]] = meta["modified_time"]

        active_drive_ids = {file["id"] for file in current_drive_files}
        
        # Action A: Clean up 
        for local_id in list(local_files.keys()):
            if local_id not in active_drive_ids:
                logger.info(f"File removed from Drive. Deleting vectors for ID: {local_id}")
                self.db_manager.chroma_collection.delete(where={"file_id": local_id})
                if on_progress:
                    on_progress()

        # Action B: Process new or modified files
        for file in current_drive_files:
            file_id = file["id"]
            drive_mod_time = file["modifiedTime"]
            
            is_new = file_id not in local_files
            is_modified = file_id in local_files and local_files[file_id] != drive_mod_time

            if (is_new or is_modified) and file.get("mimeType") in SUPPORTED_MIMETYPES:
                if is_modified:
                    logger.info(f"File modified. Updating tracking vectors for: {file['name']}")
                    self.db_manager.chroma_collection.delete(where={"file_id": file_id})
                else:
                    logger.info(f"New file discovered. Syncing: {file['name']}")

                file_bytes = drive_client.download_files(file_id)
                try:
                    docs = DocumentParser.parse_file(
                        file_bytes, file["name"], file_id, file["webViewLink"],
                        mimetype=file.get("mimeType"),
                    )
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Skipping file {file['name']}: {e}")
                    continue

                insert_with_resilience = with_resilience(self.index.insert)
                try:
                    for doc in docs:
                        doc.metadata["modified_time"] = drive_mod_time
                        insert_with_resilience(doc)
                    logger.info(f"Ingested {len(docs)} chunk(s)/page(s) for file: {file['name']}")
                except CircuitOpenError as e:
                    logger.error(
                    f"Skipping '{file['name']}': Circuit breaker is open. {e}"
                    )
                    continue 
                if on_progress:
                    on_progress()
        
        logger.info("Dynamic synchronization cycle completed successfully.")

    def sync_from_changes(self, changed_files: list, removed_file_ids: set, drive_client, on_progress=None):
        for file_id in removed_file_ids:
            logger.info(f"File removed/trashed on Drive. Deleting vectors for ID: {file_id}")
            self.db_manager.chroma_collection.delete(where={"file_id": file_id})
            if on_progress:
                on_progress()

        for file in changed_files:
            if file.get("mimeType") not in SUPPORTED_MIMETYPES:
                continue

            file_id = file["id"]
            self.db_manager.chroma_collection.delete(where={"file_id": file_id})

            logger.info(f"Syncing changed file: {file['name']}")
            file_bytes = drive_client.download_files(file_id)
            try:
                docs = DocumentParser.parse_file(
                    file_bytes, file["name"], file_id, file["webViewLink"],
                    mimetype=file.get("mimeType"),
                )
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Skipping file '{file['name']}': {e}")
                if on_progress:
                    on_progress()
                continue

            insert_with_resilience = with_resilience(self.index.insert)
            try:
                for doc in docs:
                    doc.metadata["modified_time"] = file["modifiedTime"]
                    insert_with_resilience(doc)
                logger.info(f"Ingested {len(docs)} chunk(s)/page(s) for file: {file['name']}")
            except CircuitOpenError as e:
                logger.error(
                        f"Skipping '{file['name']}': Circuit breaker is open. {e}"
                    )
                continue
            if on_progress:
                on_progress()

        if changed_files or removed_file_ids:
            logger.info(
                f"Incremental sync completed: {len(changed_files)} changed, "
                f"{len(removed_file_ids)} removed."
        )   