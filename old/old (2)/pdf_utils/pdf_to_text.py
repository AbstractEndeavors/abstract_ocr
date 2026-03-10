import os,shutil,hashlib,re,PyPDF2,logging
from pathlib import Path
from pdf2image import convert_from_path
from abstract_ocr import extract_text_from_image as image_to_text
from abstract_utilities.path_utils import (is_file, mkdirs, get_directory,  get_base_name, split_text, get_ext, get_file_name)
from abstract_utilities.type_utils import is_str
from abstract_utilities.read_write_utils import write_to_file
from typing import *
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Assuming abstract_images provides image_to_text


def get_file_hash(file_path, hash_algorithm='sha256', chunk_size=8192):
    """Calculate the hash of a file's contents."""
    hash_obj = hashlib.new(hash_algorithm)
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def is_pdf_file(file_path):
    """Check if a file is a PDF based on its extension."""
    return file_path.lower().endswith('.pdf')

def get_preferred_filename(filenames):
    """Determine the most appropriate filename from a list of duplicates."""
    name_info = []
    for fname in filenames:
        base_name, ext = os.path.splitext(fname)
        match = re.match(r'^(.*?)(?:_(\d+))?$', base_name)
        if match:
            core_name, suffix = match.groups()
            suffix = int(suffix) if suffix else None
            name_info.append((core_name, suffix, ext, fname))
    name_info.sort(key=lambda x: (x[1] is not None, x[1] or 0))
    return name_info[0][3]

def get_pdf_pages(pdf_file):
    """Get the total number of pages in the PDF."""
    pdf_file = get_pdf_obj(pdf_file)
    try:
        return len(pdf_file.pages)
    except:
        return False

def get_pdf_obj(pdf_obj):
    """Processes and returns a PDF object."""
    if is_str(pdf_obj) and is_pdf_file(pdf_obj):
        try:
            return PyPDF2.PdfReader(pdf_obj)
        except Exception as e:
            logging.error(f"Failed to read PDF {pdf_obj}: {str(e)}")
            return None
    return pdf_obj

def save_pdf(output_file_path, pdf_writer):
    """Save a PDF writer object to a file."""
    with open(output_file_path, 'wb') as output_file:
        pdf_writer.write(output_file)

def split_pdf(input_path, pdf_pages_dir, file_name=None):
    """Split a PDF into separate files for each page in a specified directory."""
    pdf_pages = []
    file_name = get_file_name(input_path) if file_name is None else file_name
    mkdirs(pdf_pages_dir)

    try:
        with open(input_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[page_num])
                output_file_path = os.path.join(pdf_pages_dir, f'{file_name}_page_{page_num + 1}.pdf')
                pdf_pages.append(output_file_path)
                save_pdf(output_file_path, pdf_writer)
    except Exception as e:
        logging.warning(f"Skipping {input_path} due to error: {str(e)}")
        return []
    return pdf_pages

def pdf_to_text_in_folders(src_dir, dest_base_dir):
    """
    Copy unique PDFs to their own folders with subfolders for pages, images, and text.
    """
    src_path = Path(src_dir)
    dest_base_path = Path(dest_base_dir)
    dest_base_path.mkdir(parents=True, exist_ok=True)
    
    hash_registry = {}
    copied_count = 0
    skipped_count = 0
    
    # First pass: Identify unique PDFs
    for filename in os.listdir(src_path):
        if is_pdf_file(filename):
            source_file = src_path / filename
            file_hash = get_file_hash(source_file)
            
            if file_hash in hash_registry:
                hash_registry[file_hash]['duplicates'].append(str(source_file))
                skipped_count += 1
                logging.info(f"Skipped duplicate: {source_file} (Hash: {file_hash[:8]}...)")
            else:
                hash_registry[file_hash] = {
                    'filename': filename,
                    'duplicates': [str(source_file)]
                }
    
    # Second pass: Process unique PDFs
    for file_hash, info in hash_registry.items():
        all_names = [Path(f).name for f in info['duplicates']]
        preferred_name = get_preferred_filename(all_names)
        source_file = Path(info['duplicates'][0])
        
        # Create folder structure
        base_name = os.path.splitext(preferred_name)[0]
        pdf_dir = dest_base_path / base_name
        pdf_dir.mkdir(exist_ok=True)
        
        pdf_pages_dir = pdf_dir / 'pdf_pages'
        images_dir = pdf_dir / 'images'
        text_dir = pdf_dir / 'text'
        pdf_pages_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
        
        # Copy original PDF to parent folder
        dest_file = pdf_dir / preferred_name
        try:
            shutil.copy2(source_file, dest_file)
            copied_count += 1
            logging.info(f"Copied: {source_file} to {dest_file} (Hash: {file_hash[:8]}...)")
        except Exception as e:
            logging.error(f"Error copying {source_file}: {str(e)}")
            continue
        
        # Split PDF into pages
        pdf_pages = split_pdf(dest_file, pdf_pages_dir)
        if not pdf_pages:
            continue
        
        # Convert each page to image and text
        for page_path in pdf_pages:
            page_name = os.path.basename(page_path)
            try:
                images = convert_from_path(page_path)
                if images:
                    img_path = os.path.join(images_dir, page_name.replace('.pdf', '.png'))
                    images[0].save(img_path, 'PNG')
                    
                    text = image_to_text(img_path)
                    txt_path = os.path.join(text_dir, page_name.replace('.pdf', '.txt'))
                    write_to_file(file_path=txt_path, contents=text)
                    logging.info(f"Converted: {page_path} -> {img_path} -> {txt_path}")
            except Exception as e:
                logging.error(f"Error converting {page_path}: {str(e)}")
        
        # Log duplicates
        if len(info['duplicates']) > 1:
            logging.info(f"Hash {file_hash[:8]}...: Kept {preferred_name}, Skipped: {', '.join(info['duplicates'][1:])}")
    
    logging.info(f"\nTotal PDFs processed: {copied_count + skipped_count}")
    logging.info(f"Unique PDFs copied and converted: {copied_count}")
    logging.info(f"Duplicates skipped: {skipped_count}")
def convrt_pdf(pdf_dir):
        pdf_convert_dir = os.path.join(pdf_dir, 'pdf_convert')
        os.makedirs(pdf_convert_dir, exist_ok=True)
        pdf_convert_dir = os.path.join(pdf_convert_dir, os.path.basename(pdf_dir))
        os.makedirs(pdf_convert_dir, exist_ok=True)
        all_paths = [os.path.join(pdf_dir,dirname) for dirname in os.listdir(pdf_dir) if dirname]
        directories = [direct for direct in all_paths if os.path.isdir(direct)]+[pdf_dir]
        for pdf_dir in directories:
            pdf_convert_dir = os.path.join(pdf_convert_dir, os.path.basename(pdf_dir))
            os.makedirs(pdf_convert_dir, exist_ok=True)
            pdf_to_text_in_folders(pdf_dir, pdf_convert_dir)
def main():
    pdf_dirs = """/mnt/24T/media/thedailydialectics/pdfs/wipow/US_2006_0185726_A1""".split('\n')
    for pdf_dir in pdf_dirs:
        convrt_pdf(pdf_dir)

if __name__ == "__main__":
    main()
