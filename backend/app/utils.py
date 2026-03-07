import io
import os

def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from .txt / .pdf / .docx bytes."""
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".txt":
        return data.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    if ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Unsupported extension: {ext}")
