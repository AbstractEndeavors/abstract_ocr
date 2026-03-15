"""
abstract_cv.column_utils
------------------------
Tools for detecting and validating multi-column layouts in scanned pages or PDF images.
Includes:
- detect_columns(): finds vertical divider between text columns
- validate_reading_order(): decides if a page is truly two-column
- visualize_columns(): saves quick visual check of divider placement
- slice_columns(): extracts left/right column image crops
"""
"""
abstract_cv.column_utils
"""

from .imports import *
from .layered_ocr import *
# -------------------------------------------------------
# Column divider detection
# -------------------------------------------------------

def detect_columns(image_path: Path):
    """
    Find the x-coordinate of the whitespace valley between two text columns.
    Returns (divider_x, image_width).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"⚠️ Could not read image: {image_path}")
        return None, None

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    vert = np.sum(binary, axis=0)
    m0   = w // 4
    m1   = 3 * w // 4
    divider = int(np.argmin(vert[m0:m1]) + m0)

    logger.info(f"📏 Detected divider at x={divider} (image width={w})")
    return divider, w


# -------------------------------------------------------
# Visualization helper
# -------------------------------------------------------

def visualize_columns(image_path: Path, divider: int):
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"⚠️ Could not read image {image_path}")
        return None

    h, w, _ = img.shape
    divider  = max(0, min(divider, w - 1))
    cv2.line(img, (divider, 0), (divider, h), (0, 0, 255), 2)

    out_path = str(image_path).replace(".png", "_divider_vis.png")
    cv2.imwrite(out_path, img)
    logger.info(f"🧩 Divider visualization saved: {out_path}")
    return Path(out_path)


# -------------------------------------------------------
# Two-column layout validation
# -------------------------------------------------------

def validate_reading_order(
    image_path: Path,
    divider: int,
    debug: bool = True,
    visualize: bool = False,
) -> bool:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"⚠️ Could not read image {image_path}")
        return False

    _, bin_img  = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w        = bin_img.shape
    divider     = max(int(w * 0.2), min(divider, int(w * 0.8)))

    left        = bin_img[:, :divider]
    right       = bin_img[:, divider:]
    left_density  = np.mean(left  > 0)
    right_density = np.mean(right > 0)

    band_width  = max(5, min(40, w // 60))
    band        = bin_img[:, divider - band_width : divider + band_width]
    band_density = np.mean(band > 0)

    max_density = max(left_density, right_density)
    balance = (
        min(left_density, right_density) / max_density
        if max_density else 0
    )
    median_density = np.median([left_density, right_density])

    is_two_col = (
        left_density  > 0.02
        and right_density > 0.02
        and band_density  < (median_density * 0.3)
        and balance       > 0.25
    )

    if debug:
        logger.info(
            f"[validate_reading_order] {Path(str(image_path)).name}: "
            f"L={left_density:.3f} R={right_density:.3f} "
            f"Band={band_density:.3f} Balance={balance:.3f} "
            f"→ {'TWO-COLUMN ✅' if is_two_col else 'SINGLE-COLUMN ❌'}"
        )

    if visualize:
        visualize_columns(image_path, divider)

    return is_two_col


# -------------------------------------------------------
# Save sliced column images to disk
# -------------------------------------------------------

def save_column_img(out_dir: str, columns_js: Dict) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    filename = columns_js.get("page", {}).get("filename", "page")
    base_dir = os.path.join(out_dir, filename)
    os.makedirs(base_dir, exist_ok=True)

    for side in ("left", "right"):
        entry = columns_js.get(side, {})
        img   = entry.get("image", {}).get("img")
        if img is None:
            continue

        col_filename = f"{filename}_{side}"
        img_path     = os.path.join(base_dir, f"{col_filename}.png")
        cv2.imwrite(img_path, img)

        columns_js[side]["filename"]       = col_filename
        columns_js[side]["image"]["path"]  = img_path
        logger.info(f"✂️  {side} column → {img_path}")

    return columns_js


# -------------------------------------------------------
# Column slicing
# -------------------------------------------------------

def slice_columns(
    image_path: str,
    divider: int  = None,
    out_dir: str  = None,
    columns_js: Dict = None,
    left_overlap: float = 0.02,   # left column extends 2% past divider
    right_gap:    float = 0.005,  # right column starts 0.5% after divider
) -> Dict[str, Dict]:
    """
    Crop an image into left and right halves around `divider`.
    Returns a dict with keys 'left' and 'right', each containing
    {'image': {'img': <ndarray>, 'path': <str>}, 'filename': <str>}.
    """
    columns_js = columns_js or {}
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"⚠️ Could not read image {image_path}")
        return columns_js

    h, w, _ = img.shape

    if divider is None:
        logger.info("Single column page detected")
        return columns_js
    out_dir = out_dir or os.getcwd()

    left_end    = max(0, int(divider + w * left_overlap))
    right_start = max(0, int(divider + w * right_gap))

    # BUG FIX: was `columns_js.get("image")` (wrong key) — clobbered itself to {}.
    columns_js.setdefault("left",  {})
    columns_js.setdefault("right", {})
    columns_js["left"].setdefault("image",  {})
    columns_js["right"].setdefault("image", {})

    columns_js["left"]["image"]["img"]  = img[:, :left_end,    :]
    columns_js["right"]["image"]["img"] = img[:, right_start:, :]

    return save_column_img(out_dir, columns_js)
