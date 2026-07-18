import io
from llama_index.core import Document
import pypdf #It will be changed to LlamaParse or PyMuPDF in the future.

class DocumentParser:
    @staticmethod
    def parse_pdf(file_bytes: bytes, file_name: str, file_id: str) -> Document:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)

        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        return Document(
            text=full_text,
            metadata={
                "file_name": file_name,
                "file_id": file_id,
                "source": "google_drive"
            }
        )