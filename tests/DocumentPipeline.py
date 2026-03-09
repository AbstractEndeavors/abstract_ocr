from src import DocumentPipeline
pdf_path = "/var/www/presites/thedailydialectics/react/main/public/pdfs/cancer/cannabis/test_cancer/test_cancer.pdf"
slice_mgr = DocumentPipeline(pdf_path)
slice_mgr.run()
