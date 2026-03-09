import re
from pathlib import Path
import PyPDF2

PAGE_RE = re.compile(r"page[_\-]?(\d+)", re.IGNORECASE)


def extract_page(name):

    m = PAGE_RE.search(name)
    if not m:
        return None

    return int(m.group(1))


def validate_collection(directory):

    base = Path(directory)

    pdfs = list(base.rglob("*.pdf"))

    if len(pdfs) != 1:
        raise RuntimeError("Document must contain exactly one PDF")

    reader = PyPDF2.PdfReader(str(pdfs[0]))
    pdf_pages = len(reader.pages)

    images = list(base.rglob("*.png"))

    img_pages = sorted(
        p for p in (extract_page(f.name) for f in images) if p
    )

    if img_pages and max(img_pages) != pdf_pages:
        raise RuntimeError(
            f"Page mismatch: PDF={pdf_pages}, images={len(img_pages)}"
        )

    return pdf_pages
