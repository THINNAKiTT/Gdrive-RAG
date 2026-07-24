"""
Unit tests for src/ingestion/document_parser.py

Covers the bug found in the original manual scripts: parse_pdf()
requires `web_view_link` as a positional argument and returns a
list[Document] (one per page), not a single Document.
"""
import pytest

from src.ingestion.document_parser import DocumentParser

pytestmark = pytest.mark.unit


def test_parse_pdf_returns_one_document_per_page(mock_fitz_document):
    mock_fitz_document.set_pages(["Page one text.", "Page two text."])

    docs = DocumentParser.parse_pdf(
        file_bytes=b"%PDF-fake-bytes",
        file_name="handbook.pdf",
        file_id="file-001",
        web_view_link="https://drive.google.com/file/d/file-001/view",
    )

    assert isinstance(docs, list)
    assert len(docs) == 2
    assert docs[0].text == "Page one text."
    assert docs[1].text == "Page two text."


def test_parse_pdf_attaches_expected_metadata(mock_fitz_document):
    mock_fitz_document.set_pages(["Only page."])

    docs = DocumentParser.parse_pdf(
        file_bytes=b"%PDF-fake-bytes",
        file_name="handbook.pdf",
        file_id="file-001",
        web_view_link="https://drive.google.com/file/d/file-001/view",
    )

    meta = docs[0].metadata
    assert meta["file_name"] == "handbook.pdf"
    assert meta["file_id"] == "file-001"
    assert meta["page_number"] == 1
    assert meta["web_view_link"] == "https://drive.google.com/file/d/file-001/view"
    assert meta["source"] == "google_drive"


def test_parse_pdf_skips_pages_with_no_extractable_text(mock_fitz_document):
    # Second page has empty text (e.g. scanned image page) -> should be dropped.
    mock_fitz_document.set_pages(["Has text.", ""])

    docs = DocumentParser.parse_pdf(
        file_bytes=b"%PDF-fake-bytes",
        file_name="scan.pdf",
        file_id="file-003",
        web_view_link="https://drive.google.com/file/d/file-003/view",
    )

    assert len(docs) == 1
    assert docs[0].metadata["page_number"] == 1


def test_parse_pdf_empty_document_returns_empty_list(mock_fitz_document):
    mock_fitz_document.set_pages([])

    docs = DocumentParser.parse_pdf(
        file_bytes=b"%PDF-fake-bytes",
        file_name="blank.pdf",
        file_id="file-004",
        web_view_link="https://drive.google.com/file/d/file-004/view",
    )

    assert docs == []


def test_parse_pdf_requires_web_view_link_positional_arg(mock_fitz_document):
    """
    Regression test for the bug in the original test scripts, which
    called parse_pdf(file_bytes, name, id) -- missing web_view_link.
    That call must fail loudly, not silently succeed with bad metadata.
    """
    with pytest.raises(TypeError):
        DocumentParser.parse_pdf(b"%PDF-fake-bytes", "handbook.pdf", "file-001")
