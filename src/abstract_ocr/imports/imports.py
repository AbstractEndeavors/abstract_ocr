import numpy as np
import pandas as pd
from typing import *
from PIL import Image
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import io, os, cv2, PyPDF2, traceback, cv2, re, cv2,logging,os
import PyPDF2, easyocr, pytest, pytesseract, glob,difflib
from abstract_utilities import *
from functools import lru_cache
from pathlib import Path
from abstract_utilities import (
    make_dirs,
    get_all_file_types,
    pytesseract,
    path_join,
    get_logFile,
    get_file_parts,
    write_to_file,
    make_list,
    is_number,
    make_dirs,
    get_lazy_attr,
    lazy_import,
    lru_cache
    )
logger = get_logFile(__name__)
