from abstract_ocr  import *
preprocess_for_ocr()
pdf_path = "/home/op/Documents/python/scripts/pdftest/hi.pdf"
slice_mgr = SliceManager(pdf_path)
slice_mgr.process_pdf()
