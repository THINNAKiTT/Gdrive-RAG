import io
from llama_index.core import Document
import fitz

class DocumentParser:
    @staticmethod
    def parse_pdf(file_bytes: bytes, file_name: str, file_id: str, web_view_link: str, file_type: str = "pdf") -> list[Document]:
        reader = fitz.open(stream=file_bytes, filetype=file_type)
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
                        "source": "google_drive"
                    }
                )
                documents_per_page.append(doc)
        return documents_per_page