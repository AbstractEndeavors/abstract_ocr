from .document_fs import normalize_pdf_directory
from .document_slug import slugify
from .document_integrity import validate_collection
from .document_rename import rename_collection
from .SliceManager import SliceManager



class DocumentPipeline:

    def __init__(self, pdf_path):

        self.base_dir = normalize_pdf_directory(pdf_path)

    def run(self):

        print("📂 normalized directory:", self.base_dir)

        # OCR
        slice_mgr = SliceManager(self.base_dir)
        slice_mgr.process_pdf()

        # validate completeness
        pages = validate_collection(self.base_dir)

        print(f"✅ validated {pages} pages")

        # slug rename
        slug = slugify(self.base_dir.name)

        new_dir = rename_collection(self.base_dir, slug)

        print("📦 renamed collection:", new_dir)

        return new_dir
