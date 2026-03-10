from pathlib import Path
from .document_fs import normalize_pdf_path
from .document_slug import slugify
from .document_integrity import validate_collection
from .document_rename import rename_collection
from .SliceManager import SliceManager


class DocumentPipeline:

    def __init__(self, pdf_path):
        self.file_parts = normalize_pdf_path(pdf_path)
        self.pdf_path = self.file_parts.get("file_path")
        self.base_dir = self.file_parts.get("dirname")

    def run(self):

        print("📂 normalized directory:", self.base_dir)

        # OCR
        slice_mgr = SliceManager(self.pdf_path)
        self.file_parts = slice_mgr.process_pdf()
        self.pdf_path = self.file_parts.get("file_path")
        self.base_dir = self.file_parts.get("dirname")
        self.dirbase = self.file_parts.get("dirbase")
        # validate completeness
##        pages = validate_collection(self.pdf_path)



        # slug rename
        slug = slugify(self.dirbase)

        new_dir = rename_collection(self.base_dir, slug)

        print("📦 renamed collection:", new_dir)

        return new_dir
