import pdfplumber
from pathlib import Path


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Returns a list of dicts, one per page:
        { 'page_num': int, 'text': str }
    Blank or near-blank pages are included with empty text so callers
    can decide whether to skip them.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            pages.append({"page_num": i + 1, "text": text})
    return pages
