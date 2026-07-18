import io
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

load_dotenv()

class GoogleDriveClient:
    def __init__(self):
        self.creds_path = os.getenv("GCP_CREDENTIALS_PATH", "config/secure_gcp_credents.json")
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']

        self.credentials = service_account.Credentials.from_service_account_file(
            self.creds_path, scopes=self.scopes
        )

        self.service = build('drive', 'v3', credentials=self.credentials)

    def list_files_in_folder(self, folder_id: str):
        query = f"'{folder_id}' in parents and trashed = false"
        results = self.service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)"
        ).execute()

        return results.get('files', [])
    
    def download_files(self, file_id: str) -> bytes:
        request = self.service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        return file_stream.getvalue()