import os
from dotenv import load_dotenv
from src.ingestion.drive_client import GoogleDriveClient
from src.ingestion.document_parser import DocumentParser

load_dotenv()

def main():
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if folder_id is None:
        print("Environment variable GOOGLE_DRIVE_FOLDER_ID is not set.")
        return

    print("Connecting to Google Drive API...")
    client = GoogleDriveClient()

    print(f"scanning files in Folder ID: {folder_id}")
    files = client.list_files_in_folder(folder_id)

    if not files:
        print("The file is not found in this folder, or the Service Account permissions are not correct.")
        return
    
    for file in files:
        print(f"\nFound: {file['name']} (ID {file['id']}), Type: {file['mimeType']}")

        if "application/pdf" in file['mimeType']:
            print("downloading and extracting {file['name']}...")
            file_bytes = client.download_files(file['id'])
            doc = DocumentParser.parse_pdf(file_bytes, file['name'], file['id'])

            print("Data retrieval succesful! example for the first 200 chars")
            print("-" * 40)
            print(doc.text[:400])
            print("-" * 40)

if __name__ == "__main__":
    main()