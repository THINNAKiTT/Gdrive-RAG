"""
Integration test: real Google Drive API + real PDF parsing.

Requires a real .env with GCP_CREDENTIALS_PATH and
GOOGLE_DRIVE_folder_id pointing at a live folder with at least one PDF.

Run explicitly with:  pytest -m integration
(excluded from the default `pytest` run via pytest.ini's default
selection of the whole tests/ tree, but marked so CI can skip it with
`pytest -m "not integration"`).
"""
import os

import pytest
from dotenv import load_dotenv

from src.ingestion.drive_client import GoogleDriveClient
from src.ingestion.document_parser import DocumentParser

load_dotenv()

pytestmark = pytest.mark.integration

folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

requires_real_drive_folder = pytest.mark.skipif(
    not folder_id, reason="GOOGLE_DRIVE_FOLDER_ID not set in environment"
)

@requires_real_drive_folder
def test_can_list_files_in_configured_folder():
    client = GoogleDriveClient()

    assert folder_id is not None
    files = client.list_files_in_folder(folder_id)

    assert isinstance(files, list)


@requires_real_drive_folder
def test_can_download_and_parse_first_pdf_in_folder():
    client = GoogleDriveClient()
    assert folder_id is not None
    files = client.list_files_in_folder(folder_id)
    pdf_files = [f for f in files if f["mimeType"] == "application/pdf"]

    if not pdf_files:
        pytest.skip("No PDF files found in the configured Drive folder")

    target = pdf_files[0]
    file_bytes = client.download_files(target["id"])

    # NOTE: web_view_link is a required positional/keyword arg -- the
    # original manual script omitted it and crashed with a TypeError.
    docs = DocumentParser.parse_pdf(
        file_bytes=file_bytes,
        file_name=target["name"],
        file_id=target["id"],
        web_view_link=target["webViewLink"],
    )

    assert isinstance(docs, list)
    if docs:  # a scanned/no-text PDF may legitimately produce zero docs
        assert docs[0].text
        assert docs[0].metadata["file_id"] == target["id"]
