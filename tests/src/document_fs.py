from pathlib import Path
import shutil
from abstract_pdfs import *

def normalize_pdf_path(pdf_path):

    file_parts = get_file_parts(pdf_path)
    dirname = file_parts.get('dirname')
    name = file_parts.get('filename')
    dirbase = file_parts.get('dirbase')
    basename = file_parts.get('basename')
    if dirbase == name:
        pdf_path = os.path.join(dirname,f"{dirbase}.pdf")
        return get_file_parts(pdf_path)
    target_dir = os.path.join(dirname,name)
    new_pdf = os.path.join(target_dir,basename)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir,exist_ok=True)
    shutil.move(str(pdf), str(new_pdf))
    return get_file_parts(str(new_pdf))


