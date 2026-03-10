from src import DocumentPipeline
from pathlib import Path
pdf_path = "/var/www/presites/thedailydialectics/react/main/public/pdfs/cancer/cannabis/test-cancer/test-cancer.pdf"
slice_mgr = DocumentPipeline(pdf_path)
slice_mgr.run()
