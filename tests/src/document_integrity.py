import re
from pathlib import Path
import PyPDF2
from abstract_utilities import *
PAGE_RE = re.compile(r"page[_\-]?(\d+)", re.IGNORECASE)


def extract_page(name):

    m = PAGE_RE.search(name)
    if not m:
        return None

    return int(m.group(1))


def validate_collection(pdf_path):
    file_parts = get_file_parts(pdf_path)
    dirname = file_parts.get('dirname')
    filename = file_parts.get('filename')
    dirbase = file_parts.get('dirbase')
    basename = file_parts.get('basename')
    pdf_path = file_parts.get("file_path")
    
    if not os.path.isfile(pdf_path):
        raise RuntimeError("Document must contain exactly one PDF")

    reader = PyPDF2.PdfReader(pdf_path)
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
