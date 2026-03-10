from .imports import *
from .column_utils import (
    detect_columns,
    validate_reading_order,
    slice_columns
    )
from .layered_ocr  import layered_ocr_img

def process_image(
    input_data,
    engine: str    = "paddle",
    debug: bool    = True,
    visualize: bool = False,
    out_dir: str   = None,
) -> dict:
    """
    Full single-image OCR pipeline:
      1. Detect columns.
      2. Validate whether the page is truly two-column.
      3. Slice if two-column; OCR each half independently.
      4. Return results.

    Returns
    -------
    {
        "ocr_results": pd.DataFrame | [pd.DataFrame, pd.DataFrame],
        "is_two_col":  bool,
        "divider":     int,
        "width":       int,
    }
    """
    # --- normalise input ---
    if isinstance(input_data, (str, Path)):
        image_path = Path(input_data)
        if not image_path.exists():
            raise ValueError(f"Image path does not exist: {image_path}")
    elif isinstance(input_data, np.ndarray):
        image_path = Path("/tmp/ocr_tmp_input.png")
        if not cv2.imwrite(str(image_path), input_data):
            raise ValueError("Failed to write ndarray to temp file")
    else:
        raise ValueError("input_data must be str, Path, or np.ndarray")

    # --- column detection ---
    divider, width = detect_columns(image_path)
    is_two_col     = validate_reading_order(
        image_path, divider, debug=debug, visualize=visualize
    )

    # --- OCR ---
    if is_two_col:
        # BUG FIX: slice_columns returns Dict, not (left_path, right_path) tuple.
        columns = slice_columns(image_path, divider, out_dir=out_dir)

        results = []
        for side in ("left", "right"):
            side_path = columns.get(side, {}).get("image", {}).get("path")
            if not side_path:
                logger.warning(f"Missing {side} column path after slicing")
                continue
            side_img = cv2.imread(str(side_path))
            if side_img is None:
                raise ValueError(f"Failed to read sliced {side} image: {side_path}")
            results.append(layered_ocr_img(side_img, engine=engine))

        ocr_results = results   # list: [left_df, right_df]
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        ocr_results = layered_ocr_img(img, engine=engine)

    return {
        "ocr_results": ocr_results,
        "is_two_col":  is_two_col,
        "divider":     divider,
        "width":       width,
    }
