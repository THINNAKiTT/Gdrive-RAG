import io
from llama_index.core import Document
import pypdf #It will be changed to LlamaParse or PyMuPDF in the future.

class DocumentParser:
    @staticmethod
    def parse_pdf(file_bytes: bytes, file_name: str, file_id: str, web_view_link: str) -> list[Document]:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)

        documents_per_page = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                doc = Document(
                    text=text,
                    metadata={
                        "file_name": file_name,
                        "file_id": file_id,
                        "page_number": page_num + 1,
                        "web_view_link": web_view_link,
                        "source": "google_drive"
                    }
                )
                documents_per_page.append(doc)
        return documents_per_page