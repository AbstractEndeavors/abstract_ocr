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
from ..imports import *
from .layered_ocr import *
# -------------------------------------------------------
# 1️⃣  Column divider detection
# -------------------------------------------------------

def detect_columns(image_path: Path) -> Tuple[int, int]:
    """
    Detects approximate column divider by analyzing vertical ink density.
    Returns (divider_x_position, image_width).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"⚠️ Could not read image: {image_path}")
        return 0, 0

    # Binary inverse (ink = white)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    w = binary.shape[1]

    # Sum vertical ink to detect whitespace valleys
    vert = np.sum(binary, axis=0)
    m0, m1 = w // 4, 3 * w // 4
    divider = int(np.argmin(vert[m0:m1]) + m0)-1

    logger.info(f"📏 Detected divider at x={divider} of width={w}")
    return divider, w
# -------------------------------------------------------
# 3️⃣  Visualization helper
# -------------------------------------------------------

def visualize_columns(
    image_path: Path,
    divider: int
    ) -> Path:
    """
    Draws a visible divider line to confirm column separation visually.
    Returns path to saved visualization.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"⚠️ Could not read image {image_path}")
        return None

    h, w, _ = img.shape
    cv2.line(img, (divider, 0), (divider, h), (0, 0, 255), 2)
    out_path = str(image_path).replace(".png", "_divider_vis.png")
    cv2.imwrite(out_path, img)
    logger.info(f"🧩 Divider visualization saved to: {out_path}")
    return Path(out_path)


# -------------------------------------------------------
# 2️⃣  Column layout validation
# -------------------------------------------------------

def validate_reading_order(
    image_path: Path,
    divider: int,
    debug: bool = True,
    visualize: bool = False,
) -> bool:
    """
    Adaptive detector to confirm two-column layout.
    Uses ink density & whitespace balance near divider.
    Returns True if layout likely contains two text columns.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"⚠️ Could not read image {image_path}")
        return False

    # Threshold ink pixels
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bin_img.shape
    divider = min(max(divider, int(w * 0.2)), int(w * 0.8))

    # Split halves
    left, right = bin_img[:, :divider], bin_img[:, divider:]
    left_density, right_density = np.mean(left > 0), np.mean(right > 0)

    # White band density (the gutter)
    band_width = max(5, min(40, w // 60))
    band = bin_img[:, divider - band_width : divider + band_width]
    band_density = np.mean(band > 0)

    # Ink density ratio between halves
    balance = (
        min(left_density, right_density) / max(left_density, right_density)
        if max(left_density, right_density)
        else 0
    )

    median_density = np.median([left_density, right_density])
    is_two_col = (
        left_density > 0.02
        and right_density > 0.02
        and band_density < (median_density * 0.3)
        and balance > 0.25
    )

    if debug:
        logger.info(
            f"[validate_reading_order] {os.path.basename(str(image_path))}: "
            f"L={left_density:.3f}, R={right_density:.3f}, Band={band_density:.3f}, "
            f"Balance={balance:.3f} → {'TWO COLUMNS ✅' if is_two_col else 'ONE COLUMN ❌'}"
        )

    if visualize:
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(overlay, (divider, 0), (divider, h), (0, 0, 255), 2)
        cv2.rectangle(
            overlay,
            (divider - band_width, 0),
            (divider + band_width, h),
            (0, 255, 255),
            1,
        )
        out_vis = str(image_path).replace(".png", "_validate_vis.png")
        cv2.imwrite(out_vis, overlay)
        logger.info(f"📸 Visualization saved to {out_vis}")

    return is_two_col



def save_column_img(out_dir: str, columns_js: Dict[str, Dict]) -> Dict[str, Dict]:
    make_dirs(out_dir)
    filename = columns_js.get("page", {}).get('filename', "page")
    base_dir = os.path.join(out_dir, filename)
    make_dirs(base_dir)

    for column, values in columns_js.items():
        if column not in ("left", "right"):
            continue
        img = values["image"].get("img")
        if img is None:
            continue
        col_filename = f"{filename}_{column}"
        columns_js[column]["filename"] = col_filename
        img_path = os.path.join(base_dir, f"{col_filename}.png")
        cv2.imwrite(img_path, img)
        columns_js[column]["image"]["path"] = img_path
        logger.info(f"✂️  Columns sliced → {column}: {img_path}")
    return columns_js

    
# -------------------------------------------------------
# 4️⃣  Column slicing
# -------------------------------------------------------

def slice_columns(
    image_path: str,
    divider: int = None,
    out_dir: str = None,
    columns_js: Dict = None,
    left_overlap: float = 0.02,   # left column reaches 2% *past* divider
    right_gap: float = 0.005      # right column starts 0.5% *after* divider
) -> Dict[str, Dict]:
    """
    Split an image into left/right halves, giving the left column priority
    at the divider. The left extends slightly *past* the divider, while
    the right begins a bit *after* it.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"⚠️ Could not read image {image_path}")
        return columns_js or {}

    h, w, _ = img.shape
    divider = divider or detect_columns(image_path)[0]
    out_dir = out_dir or os.getcwd()

    # compute offsets
    left_end = max(0, int(divider + w * left_overlap))
    right_start = max(0, int(divider + w * right_gap))

    # crop halves
    left = img[:, :left_end, :]
    right = img[:, right_start:, :]

    # assign results
    if columns_js:
        columns_js["left"]["image"]["img"] = left
        columns_js["right"]["image"]["img"] = right
        return save_column_img(out_dir, columns_js)
    else:
        return {"left": left, "right": right}
