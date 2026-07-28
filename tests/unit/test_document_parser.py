"""
Unit tests for src/ingestion/document_parser.py

Covers the bug found in the original manual scripts: parse_pdf()
requires `web_view_link` as a positional argument and returns a
list[Document] (one per page), not a single Document.
"""
import pytest

from src.ingestion.document_parser import DocumentParser, SUPPORTED_MIMETYPES

pytestmark = pytest.mark.unit

#----------------------------
#-- PDF (application/pdf) --#
#----------------------------

def test_parse_file_pdf_returns_one_document_per_page(mock_fitz_document):
    mock_fitz_document.set_pages(["Page one text.", "Page two text."])

    docs = DocumentParser.parse_file(
        file_bytes=b"%PDF-fake-bytes",
        file_name="handbook.pdf",
        file_id="file-001",
        web_view_link="https://drive.google.com/file/d/file-001/view",
        mimetype="application/pdf"
    )

    assert isinstance(docs, list)
    assert len(docs) == 2
    assert docs[0].text == "Page one text."
    assert docs[1].text == "Page two text."


def test_parse_file_pdf_attaches_expected_metadata(mock_fitz_document):
    mock_fitz_document.set_pages(["Only page."])

    docs = DocumentParser.parse_file(
        file_bytes=b"%PDF-fake-bytes",
        file_name="handbook.pdf",
        file_id="file-001",
        web_view_link="https://drive.google.com/file/d/file-001/view",
        mimetype="application/pdf",
    )

    meta = docs[0].metadata
    assert meta["file_name"] == "handbook.pdf"
    assert meta["file_id"] == "file-001"
    assert meta["page_number"] == 1
    assert meta["web_view_link"] == "https://drive.google.com/file/d/file-001/view"
    assert meta["source"] == "google_drive"


def test_parse_file_pdf_skips_pages_with_no_extractable_text(mock_fitz_document):
    # Second page has empty text (e.g. scanned image page) -> should be dropped.
    mock_fitz_document.set_pages(["Has text.", ""])

    docs = DocumentParser.parse_file(
        file_bytes=b"%PDF-fake-bytes",
        file_name="scan.pdf",
        file_id="file-003",
        web_view_link="https://drive.google.com/file/d/file-003/view",
        mimetype="application/pdf",
    )

    assert len(docs) == 1
    assert docs[0].metadata["page_number"] == 1


def test_parse_file_pdf_empty_document_returns_empty_list(mock_fitz_document):
    mock_fitz_document.set_pages([])

    docs = DocumentParser.parse_file(
        file_bytes=b"%PDF-fake-bytes",
        file_name="blank.pdf",
        file_id="file-004",
        web_view_link="https://drive.google.com/file/d/file-004/view",
        mimetype="application/pdf",
    )

    assert docs == []


def test_parse_file_requires_mimetype_arg(mock_fitz_document):
    """
    Regression test for the bug in the original test scripts, which
    called parse_pdf(file_bytes, name, id) -- missing web_view_link.
    That call must fail loudly, not silently succeed with bad metadata.
    """
    with pytest.raises(TypeError):
        DocumentParser.parse_file(
            b"%PDF-fake-bytes", "handbook.pdf", "file-001",
            "https://drive.google.com/file/d/file-001/view",
        )

#----------------------------------
#-- EPUB (application/epub+zip) --#
#----------------------------------

def test_parse_file_epub_returns_one_document_per_page(mock_fitz_document):
    mock_fitz_document.set_pages(["Chapter 1 text.", "Chapter 2 text."])

    docs = DocumentParser.parse_file(
        file_bytes=b"epub-fake-bytes",
        file_name="pirates-of-venus.epub",
        file_id="file-010",
        web_view_link="https://drive.google.com/file/d/file-010/view",
        mimetype="application/epub+zip",
    )

    assert len(docs) == 2
    assert docs[0].text == "Chapter 1 text."

def test_parse_file_epub_calls_fitz_open_with_epub_filetype(monkeypatch, mock_fitz_document):
    """
    Regression guard for the original bug: the old code always called
    fitz.open(..., filetype="pdf") regardless of the real file type.
    EPUB must be opened with filetype="epub" explicitly.
    """
    docs = DocumentParser.parse_file(
        file_bytes=b"epub-fake-bytes",
        file_name="pirates-of-venus.epub",
        file_id="file-010",
        web_view_link="",
        mimetype="application/epub+zip",
    )

    _, kwargs = mock_fitz_document.open.call_args
    assert kwargs["filetype"] == "epub"


#------------------------------
#-- Plain text (text/plain) --#
#------------------------------


def test_parse_file_text_returns_single_document():
    docs = DocumentParser.parse_file(
        file_bytes="Hello, this is a plain text note.".encode("utf-8"),
        file_name="notes.txt",
        file_id="file-002",
        web_view_link="https://drive.google.com/file/d/file-002/view",
        mimetype="text/plain",
    )

    assert len(docs) == 1
    assert docs[0].text == "Hello, this is a plain text note."
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["file_id"] == "file-002"


def test_parse_file_text_empty_file_returns_empty_list():
    docs = DocumentParser.parse_file(
        file_bytes=b"",
        file_name="empty.txt",
        file_id="file-011",
        web_view_link="",
        mimetype="text/plain",
    )

    assert docs == []


def test_parse_file_text_whitespace_only_returns_empty_list():
    docs = DocumentParser.parse_file(
        file_bytes=b"   \n\n   ",
        file_name="whitespace.txt",
        file_id="file-012",
        web_view_link="",
        mimetype="text/plain",
    )

    assert docs == []


def test_parse_file_text_handles_invalid_utf8_without_crashing():
    docs = DocumentParser.parse_file(
        file_bytes=b"\xff\xfe not valid utf-8 \x80\x81",
        file_name="bad_encoding.txt",
        file_id="file-013",
        web_view_link="",
        mimetype="text/plain",
    )

    # Must not raise UnicodeDecodeError -- falls back to replacement
    # characters rather than crashing the whole sync cycle over one file.
    assert isinstance(docs, list)

#---------------------------------------------
#-- Images (image/png, image/jpeg) via OCR --#
#---------------------------------------------

@pytest.fixture
def mock_ocr(monkeypatch):
    """Mocks pytesseract + PIL so image tests need no real Tesseract
    binary and no real image bytes."""
    import src.ingestion.document_parser as parser_module

    state = {"text": "OCR extracted text."}

    fake_image = object()
    monkeypatch.setattr(parser_module.Image, "open", lambda _bytes_io: fake_image)
    monkeypatch.setattr(
        parser_module.pytesseract, "image_to_string", lambda img: state["text"]
    )
    monkeypatch.setattr(parser_module, "OCR_AVAILABLE", True)

    def _set_text(text):
        state["text"] = text

    class _Handle:
        set_text = staticmethod(_set_text)

    return _Handle()


def test_parse_file_image_returns_ocr_text(mock_ocr):
    mock_ocr.set_text("Scanned receipt: total $42.00")

    docs = DocumentParser.parse_file(
        file_bytes=b"fake-png-bytes",
        file_name="receipt.png",
        file_id="file-020",
        web_view_link="",
        mimetype="image/png",
    )

    assert len(docs) == 1
    assert docs[0].text == "Scanned receipt: total $42.00"


def test_parse_file_image_empty_ocr_result_returns_empty_list(mock_ocr):
    mock_ocr.set_text("")

    docs = DocumentParser.parse_file(
        file_bytes=b"fake-png-bytes",
        file_name="blank_image.png",
        file_id="file-021",
        web_view_link="",
        mimetype="image/png",
    )

    assert docs == []


def test_parse_file_image_raises_clear_error_when_ocr_unavailable(monkeypatch):
    import src.ingestion.document_parser as parser_module

    monkeypatch.setattr(parser_module, "OCR_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="OCR dependencies"):
        DocumentParser.parse_file(
            file_bytes=b"fake-png-bytes",
            file_name="receipt.png",
            file_id="file-020",
            web_view_link="",
            mimetype="image/png",
        )


def test_parse_file_image_raises_clear_error_when_tesseract_binary_missing(
    monkeypatch, mock_ocr
):
    import pytesseract as real_pytesseract
    import src.ingestion.document_parser as parser_module

    def _raise_not_found(img):
        raise real_pytesseract.pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(parser_module.pytesseract, "image_to_string", _raise_not_found)

    with pytest.raises(RuntimeError, match="Tesseract OCR engine not found"):
        DocumentParser.parse_file(
            file_bytes=b"fake-png-bytes",
            file_name="receipt.png",
            file_id="file-020",
            web_view_link="",
            mimetype="image/png",
        )


# ---------------------------------------------------------------------------
# Unsupported mimetypes
# ---------------------------------------------------------------------------


def test_parse_file_unsupported_mimetype_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported mimetype"):
        DocumentParser.parse_file(
            file_bytes=b"some bytes",
            file_name="comic.cbz",
            file_id="file-030",
            web_view_link="",
            mimetype="application/x-cbz",
        )


def test_supported_mimetypes_does_not_include_cbz():
    """Regression guard: .cbz was deliberately dropped -- it has no
    extractable text by nature, so it should never appear as
    'supported' again by accident."""
    assert "application/x-cbz" not in SUPPORTED_MIMETYPES


def test_supported_mimetypes_includes_expected_types():
    assert set(SUPPORTED_MIMETYPES) == {
        "application/pdf",
        "application/epub+zip",
        "image/png",
        "image/jpeg",
        "text/plain",
    }