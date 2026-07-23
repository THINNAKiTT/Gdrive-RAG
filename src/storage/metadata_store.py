from llama_index.core import VectorStoreIndex

class DynamicSyncManager:
    def __init__(self, index: VectorStoreIndex, db_manager):
        self.index = index
        self.db_manager = db_manager

    def sync_with_drive(self, current_drive_files: list, drive_client):
        """Compares local index metadata with active Google Drive state."""
        from src.ingestion.document_parser import DocumentParser

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
                print(f"File removed from Drive. Deleting vectors for ID: {local_id}")
                self.db_manager.chroma_collection.delete(where={"file_id": local_id})

        # Action B: Process new or modified files
        for file in current_drive_files:
            file_id = file["id"]
            drive_mod_time = file["modifiedTime"]
            
            is_new = file_id not in local_files
            is_modified = file_id in local_files and local_files[file_id] != drive_mod_time

            SUPPORTED_MIMETYPES = [
                "application/pdf",
                "application/epub+zip",
                "application/x-cbz",
                "image/png",
                "image/jpeg",
                "text/plain"
            ]

            if (is_new or is_modified) and file.get("mimeType") in SUPPORTED_MIMETYPES:
                if is_modified:
                    print(f"File modified. Updating tracking vectors for: {file['name']}")
                    self.db_manager.chroma_collection.delete(where={"file_id": file_id})
                else:
                    print(f"New file discovered. Syncing: {file['name']}")

                file_bytes = drive_client.download_files(file_id)
                docs = DocumentParser.parse_pdf(file_bytes, file["name"], file_id, file["webViewLink"])
                
                for doc in docs:
                    doc.metadata["modified_time"] = drive_mod_time
                
                for doc in docs:
                    self.index.insert(doc)
        
        print("Dynamic synchronization cycle completed successfully.")