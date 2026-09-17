import pymupdf
from pathlib import Path


def load_pdf(pdf_path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_number,
                "source": Path(pdf_path).name
            })

    document.close()

    return pages