import io
from llama_index.core import Document
import fitz

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

SUPPORTED_MIMETYPES = [
    "application/pdf",
    "application/epub+zip",
    "image/png",
    "image/jpeg",
    "text/plain",
]

class DocumentParser:
    @staticmethod
    def parse_file(
            file_bytes: bytes,
            file_name: str, 
            file_id: str, 
            web_view_link: str, 
            mimetype: str
        ) -> list[Document]:
        if mimetype == "application/pdf":
            return DocumentParser._parse_pdf(file_bytes, file_name, file_id, web_view_link)
        elif mimetype == "application/epub+zip":
            return DocumentParser._parse_epub(file_bytes, file_name, file_id, web_view_link)
        elif mimetype == "text/plain":
            return DocumentParser._parse_text(file_bytes, file_name, file_id, web_view_link)
        elif mimetype in ("image/png", "image/jpeg"):
            return DocumentParser._parse_image(file_bytes, file_name, file_id, web_view_link)
        else:
            raise ValueError(
                f"Unsupported mimetype '{mimetype}' for file '{file_name}'."
                f"Supported types: {SUPPORTED_MIMETYPES}."
            )
        
    @staticmethod
    def _parse_pdf(file_bytes: bytes, file_name: str, file_id: str, web_view_link: str) -> list[Document]:
        reader = fitz.open(stream=file_bytes, filetype="pdf")
        return DocumentParser._extract_pages(reader, file_name, file_id, web_view_link)

    @staticmethod
    def _parse_epub(file_bytes: bytes, file_name: str, file_id: str, web_view_link: str) -> list[Document]:
        reader = fitz.open(stream=file_bytes, filetype="epub")
        return DocumentParser._extract_pages(reader, file_name, file_id, web_view_link)

    @staticmethod
    def _extract_pages(reader, file_name: str, file_id: str, web_view_link: str) -> list[Document]:
        documents_per_page = []
        for page_num in range(len(reader)):
            page = reader.load_page(page_num)
            text = page.get_text()
            if text:
                doc = Document(
                    text=text,
                    metadata={
                        "file_name": file_name,
                        "file_id": file_id,
                        "page_number": page_num + 1,
                        "web_view_link": web_view_link,
                        "source": "google_drive",
                    },
                )
                documents_per_page.append(doc)
        return documents_per_page

    @staticmethod
    def _parse_text(file_bytes: bytes, file_name: str, file_id: str, web_view_link: str) -> list[Document]:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("utf-8", errors="replace")

        if not text.strip():
            return []

        return [
            Document(
                text=text,
                metadata={
                    "file_name": file_name,
                    "file_id": file_id,
                    "page_number": 1,
                    "web_view_link": web_view_link,
                    "source": "google_drive",
                },
            )
        ]

    @staticmethod
    def _parse_image(file_bytes: bytes, file_name: str, file_id: str, web_view_link: str) -> list[Document]:
        if not OCR_AVAILABLE:
            raise RuntimeError(
                "OCR dependencies (pytesseract, Pillow) are not installed. "
                "Install with: pip install pytesseract Pillow, and ensure "
            )

        try:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
        except pytesseract.pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract OCR engine not found on this system. "
                "Install it with: "
                "(e.g.)"
                "brew install tesseract #for macOS"
                "sudo apt install tesseract-ocr tesseract-ocr-eng #for Linux (Ubuntu/Debian)"
            )

        if not text.strip():
            return []

        return [
            Document(
                text=text,
                metadata={
                    "file_name": file_name,
                    "file_id": file_id,
                    "page_number": 1,
                    "web_view_link": web_view_link,
                    "source": "google_drive",
                },
            )
        ]