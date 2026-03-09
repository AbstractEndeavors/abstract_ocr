from abstract_pdfs import *
from typing import Optional, Tuple, Dict
import os
import shutil
import traceback
from .document_fs import normalize_pdf_path


# ---------------------------------------------------------
# PDF Discovery
# ---------------------------------------------------------

def find_closest_pdf(directory: str) -> Optional[str]:
    """
    Locate the most likely PDF inside a directory tree.

    Preference:
    1. PDF matching directory name
    2. Shallowest PDF in directory
    """
    directory = os.path.abspath(directory)
    target = os.path.basename(directory).lower().replace(" ", "_")

    closest_pdf = None
    closest_depth = float("inf")

    for root, _, files in os.walk(directory):

        depth = root[len(directory):].count(os.sep)
        pdfs = [f for f in files if f.lower().endswith(".pdf")]

        for pdf in pdfs:

            base = os.path.splitext(pdf)[0].lower().replace(" ", "_")

            if base == target:
                return os.path.join(root, pdf)

            if depth < closest_depth:
                closest_depth = depth
                closest_pdf = os.path.join(root, pdf)

    return closest_pdf

# ---------------------------------------------------------
# Directory Normalization
# ---------------------------------------------------------

def ensure_pdf_directory(pdf_item: str, out_root: Optional[str] = None) -> Tuple[str, str]:
    """
    Guarantee that a PDF lives inside a directory with the same name.
    """
    file_parts = normalize_pdf_path(pdf_item)
    dirname = file_parts.get('dirname')
    filename = file_parts.get('filename')
    dirbase = file_parts.get('dirbase')
    basename = file_parts.get('basename')
    pdf_item = file_parts.get("file_path")
    if os.path.isdir(pdf_item):
        input(pdf_item)
        pdf_item = find_closest_pdf(pdf_item)

        if not pdf_item:
            raise RuntimeError(f"No PDF found in directory: {pdf_item}")

        return get_file_parts(pdf_item)

    if not os.path.isfile(pdf_item) or not pdf_item.lower().endswith(".pdf"):
        raise RuntimeError(f"Invalid PDF path: {pdf_item}")
    file_parts = normalize_pdf_path(pdf_item)
    dirname = file_parts.get('dirname')
    filename = file_parts.get('filename')
    dirbase = file_parts.get('dirbase')
    basename = file_parts.get('basename')
    pdf_item = file_parts.get("file_path")

    base_dir = out_root or dirname

    os.makedirs(base_dir, exist_ok=True)

    new_pdf = os.path.join(base_dir, os.path.basename(pdf_item))

    if pdf_item != new_pdf:
        shutil.copy(pdf_item, new_pdf)

    return get_file_parts(new_pdf)

# ---------------------------------------------------------
# Slice Manager
# ---------------------------------------------------------

class SliceManager:
    """
    Column-aware OCR processor.
    """

    def __init__(
        self,
        pdf_path: str,
        out_root: Optional[str] = None,
        engines="paddle",
        engine_directory=False,
        visualize=False,
    ):
        self.file_parts = ensure_pdf_directory(pdf_path, out_root)
        self.pdf_path = self.file_parts.get("file_path")
        self.base_dir = self.file_parts.get("dirname")
        self.filename = self.file_parts.get("filename")

        self.visualize = visualize
        self.imgs = []

        self.engines = make_list(engines or Config.OCR_ENGINES)
        self.engine_directory = engine_directory or len(self.engines) > 1

        # Base directories
        self.pages = make_dir(self.base_dir, "pages")
        self.images = make_dir(self.base_dir, "images")
        self.cols = make_dir(self.base_dir, "columns")

        # Engine directories
        self.engine_dirs = {}

        for engine in self.engines:

            root = make_dir(self.base_dir, engine) if self.engine_directory else self.base_dir

            self.engine_dirs[engine] = {

                "root": root,
                "raw_tx": make_dir(root, "text"),
                "clean_tx": make_dir(root, "text", "cleaned"),
                "pre_img": make_dir(root, "preprocessed_images"),
                "pre_txt": make_dir(root, "preprocessed_text"),
                "pre_cln": make_dir(root, "preprocessed_text", "cleaned"),

                "final_raw": os.path.join(root, f"{self.filename}_{engine}_FULL.txt"),
                "final_clean": os.path.join(root, f"{self.filename}_{engine}_FULL_cleaned.txt"),
            }

    # ---------------------------------------------------------
    # Page Extraction
    # ---------------------------------------------------------

    def extract_page_image(self, page, page_num: int) -> Optional[str]:

        try:

            page_pdf = os.path.join(self.pages, f"page_{page_num}.pdf")

            writer = PyPDF2.PdfWriter()
            writer.add_page(page)

            with open(page_pdf, "wb") as f:
                writer.write(f)

            images = convert_from_path(page_pdf)

            if not images:
                logger.warning(f"No image extracted for page {page_num}")
                return None

            img_path = os.path.join(self.images, f"page_{page_num}.png")

            images[0].save(img_path, "PNG")

            return img_path

        except Exception as e:

            logger.error(f"Page extraction failed on {page_num}: {e}")
            return None

    # ---------------------------------------------------------
    # OCR Column Processing
    # ---------------------------------------------------------

    def process_single_column(
        self,
        img_path: str,
        page_num: int,
        engine: str,
        side_label: str = "",
    ) -> Tuple[str, str]:

        dirs = self.engine_dirs[engine]

        suffix = f"_{side_label}" if side_label else ""
        png_name = f"page_{page_num}{suffix}.png"
        txt_name = f"page_{page_num}{suffix}.txt"

        proc_img = os.path.join(dirs["pre_img"], png_name)

        preprocess_image(img_path, proc_img)

        image_array = cv2.imread(proc_img)

        df = layered_ocr_img(image_array, engine=engine)

        txt = "\n".join(df["text"].tolist())
        txt_path = os.path.join(dirs["raw_tx"], txt_name)
        write_to_file(contents=txt,file_path=txt_path)
        
        cln = clean_text(txt)
        cln_path = os.path.join(dirs["clean_tx"], txt_name)
        write_to_file(contents=cln,file_path=cln_path)

        logger.info(f"[{engine}] OCR complete page {page_num}{suffix}")

        return txt, cln

    # ---------------------------------------------------------
    # Page Processing
    # ---------------------------------------------------------

    def process_page(self, page, page_num: int, engine: str):

        result = {
            "left": {"raw": {"text": None}, "clean": {"text": None}},
            "right": {"raw": {"text": None}, "clean": {"text": None}},
        }

        try:

            img_path = self.extract_page_image(page, page_num)

            if not img_path:
                return result

            divider, _ = detect_columns(img_path)

            validate_reading_order(img_path, divider, visualize=self.visualize)

            columns = slice_columns(img_path, divider, self.base_dir, {})
            input(columns)
            for side, meta in columns.items():
                input(meta)
                if side not in ("left", "right"):
                    continue
                input(meta)
                txt, cln = self.process_single_column(
                    meta["image"]["path"],
                    page_num,
                    engine,
                    side,
                )

                result[side]["raw"]["text"] = txt
                result[side]["clean"]["text"] = cln

            return result

        except Exception as e:

            logger.error(f"[{engine}] page {page_num} failed: {e}")
            traceback.print_exc()

            return result

    # ---------------------------------------------------------
    # Engine Processing
    # ---------------------------------------------------------

    def process_pdf_for_engine(self, engine: str):

        logger.info(f"[{engine}] starting OCR for {self.filename}")

        reader = PyPDF2.PdfReader(self.pdf_path)
        dirs = self.engine_dirs[engine]

        all_left = []
        all_right = []

        for i, page in enumerate(reader.pages, start=1):

            res = self.process_page(page, i, engine)

            if res["left"]["raw"]["text"]:
                all_left.append(res["left"]["raw"]["text"])

            if res["right"]["raw"]["text"]:
                all_right.append(res["right"]["raw"]["text"])
        left_text = "\n\n".join(all_right)
        left_text_path = os.path.join(dirs["raw_tx"], f"{self.filename}_{engine}_LEFT.txt")
        write_to_file(contents=left_text,file_path=left_text_path)
        right_text = "\n\n".join(all_left)
        right_text_path = os.path.join(dirs["raw_tx"], f"{self.filename}_{engine}_RIGHT.txt")
        write_to_file(contents=right_text,file_path=right_text_path)

        logger.info(f"[{engine}] finished OCR")

    # ---------------------------------------------------------
    # Multi Engine
    # ---------------------------------------------------------

    def process_pdf(self):

        logger.info(f"Starting OCR pipeline for {self.filename}")

        for engine in self.engines:
            self.process_pdf_for_engine(engine)

        logger.info("OCR pipeline complete")
