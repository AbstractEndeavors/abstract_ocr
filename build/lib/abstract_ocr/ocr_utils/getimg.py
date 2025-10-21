import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

# Assuming you have these from your modules (adjust imports as needed)
from abstract_ocr.ocr_utils.column_utils import detect_columns, validate_reading_order, slice_columns
from abstract_ocr.ocr_utils.layered_ocr import layered_ocr_img

def process_image(input_data, engine='paddle', debug=True, visualize=False, out_dir=None):
    """
    Processes an image (path or np.array): detects columns, validates layout, slices if needed, performs OCR.
    
    Args:
        input_data: str, Path, or np.ndarray of the image.
        engine: OCR engine ('tesseract', 'easy', 'paddle'). Defaults to 'paddle'.
        debug: Log detailed validation info.
        visualize: Save visualization images for column detection.
        out_dir: Optional Path/str for saving sliced images (defaults to cwd).
    
    Returns:
        dict with 'ocr_results' (pd.DataFrame or list[pd.DataFrame]), 'is_two_col' (bool),
        'divider' (int), 'width' (int).
    
    Raises:
        ValueError: If input is invalid or image can't be read.
    """
    # Handle input conversion
    if isinstance(input_data, (str, Path)):
        image_path = Path(input_data)
        if not image_path.exists():
            raise ValueError(f"Image path does not exist: {image_path}")
    elif isinstance(input_data, np.ndarray):
        image_path = Path('/tmp/temp_img.png')
        if not cv2.imwrite(str(image_path), input_data):
            raise ValueError("Failed to write NumPy array to temp image")
    else:
        raise ValueError("Input must be str, Path, or np.ndarray")
    
    # Detect and validate columns
    divider, width = detect_columns(image_path)
    is_two_col = validate_reading_order(image_path, divider, debug=debug, visualize=visualize)
    
    # Perform OCR (slice if two columns)
    if is_two_col:
        left_path, right_path = slice_columns(image_path, divider, out_dir=out_dir)
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        if left_img is None or right_img is None:
            raise ValueError("Failed to read sliced images")
        ocr_left = layered_ocr_img(left_img, engine=engine)
        ocr_right = layered_ocr_img(right_img, engine=engine)
        ocr_results = [ocr_left, ocr_right]  # List for left/right
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        ocr_results = layered_ocr_img(img, engine=engine)  # Single DataFrame
    
    return {
        'ocr_results': ocr_results,
        'is_two_col': is_two_col,
        'divider': divider,
        'width': width
    }
