import io
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

load_dotenv()

from src.utils.logger import get_logger
logger = get_logger("GDrive-RAG")

class GoogleDriveClient:
    def __init__(self):
        self.creds_path = os.getenv("GCP_CREDENTIALS_PATH", "config/secure_gcp_credentials.json")
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']

        self.credentials = service_account.Credentials.from_service_account_file(
            self.creds_path, scopes=self.scopes
        )

        self.service = build('drive', 'v3', credentials=self.credentials)

    def list_files_in_folder(self, folder_id: str):
        query = f"'{folder_id}' in parents and trashed = false"
        results = self.service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)"
        ).execute()

        return results.get('files', [])

    def get_start_page_token(self) -> str:
        response = self.service.changes().getStartPageToken().execute()
        return response["startPageToken"]

    def list_changes(self, page_token: str, folder_id: str):
        changed_files = []
        removed_file_ids = set()
        token = page_token
        new_page_token = page_token 

        while token is not None:
            response = self.service.changes().list(
                pageToken=token,
                fields=(
                    "nextPageToken, newStartPageToken, "
                    "changes(fileId, removed, "
                    "file(id, name, mimeType, modifiedTime, webViewLink, parents, trashed))"
                ),
            ).execute()

            for change in response.get("changes", []):
                file_id = change.get("fileId")
                file_meta = change.get("file")
                is_removed_or_trashed = (
                    change.get("removed")
                    or (file_meta is not None and file_meta.get("trashed"))

                )
                is_in_watched_folder = (
                    file_meta is not None
                    and folder_id in (file_meta.get("parents") or [])
                )

                if is_removed_or_trashed or not is_in_watched_folder:
                    removed_file_ids.add(file_id)
                elif is_in_watched_folder:
                    if file_meta is not None:
                        changed_files.append({
                            "id": file_meta["id"],
                            "name": file_meta["name"],
                            "mimeType": file_meta["mimeType"],
                            "modifiedTime": file_meta["modifiedTime"],
                            "webViewLink": file_meta.get("webViewLink", ""),
                    })

            if "newStartPageToken" in response:
                new_page_token = response["newStartPageToken"]
                token = None  # this was the last page
            else:
                token = response.get("nextPageToken")
        changed_files = [f for f in changed_files if f["id"] not in removed_file_ids]
        return changed_files, removed_file_ids, new_page_token

    
    def download_files(self, file_id: str) -> bytes:
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(file_stream, request)
            done = False

            while not done:
                status, done = downloader.next_chunk()
            return file_stream.getvalue()
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
        return b""