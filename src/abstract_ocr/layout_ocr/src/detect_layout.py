"""
Step: detect_layout

PreprocessedImage → (PreprocessedImage, LayoutDetection)

Column detection via vertical projection profile.
Block classification via connected-component analysis.
"""
from ..imports import (
    cv2,
    np,
    Tuple,
    annotations
    ) 
from ..registry import registry
from ..schemas import (
    BBox,
    BlockKind,
    ColumnLayout,
    LayoutDetection,
    LayoutRegion,
    PipelineConfig,
    PreprocessedImage,
)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _vertical_projection(binary: np.ndarray, blur_k: int) -> np.ndarray:
    projection = binary.astype(np.float64).sum(axis=0)
    if blur_k > 1:
        blur_k = blur_k | 1
        projection = cv2.GaussianBlur(
            projection.reshape(1, -1), (blur_k, 1), 0
        ).flatten()
    return projection


def _find_valleys(
    projection: np.ndarray,
    page_width: int,
    min_gap_frac: float,
    valley_depth: float,
    max_columns: int,
) -> list[int]:
    """
    Find valleys (column dividers) in the vertical projection.
    A valley = contiguous region below valley_depth * peak, at least
    min_gap_frac * width wide, outside page margins.
    """
    if projection.max() == 0:
        return []

    threshold = projection.max() * valley_depth
    min_gap_px = int(page_width * min_gap_frac)

    below = projection < threshold
    valleys: list[tuple[int, int]] = []
    start = None

    for i, b in enumerate(below):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if (i - start) >= min_gap_px:
                valleys.append((start, i))
            start = None
    if start is not None and (len(below) - start) >= min_gap_px:
        valleys.append((start, len(below)))

    # Exclude page margins (first/last 8%)
    margin = int(page_width * 0.08)
    valleys = [
        (s, e) for s, e in valleys
        if s > margin and e < page_width - margin
    ]

    # Keep deepest valleys, up to max_columns - 1
    valleys.sort(key=lambda v: projection[v[0]:v[1]].sum())
    valleys = valleys[: max_columns - 1]
    valleys.sort(key=lambda v: v[0])

    return [(s + e) // 2 for s, e in valleys]


def _detect_header_cutoff(binary: np.ndarray, scan_fraction: float) -> int:
    h, w = binary.shape
    scan_h = int(h * scan_fraction)
    if scan_h < 10:
        return 0

    row_density = binary[:scan_h].astype(np.float64).sum(axis=1) / w
    threshold = row_density.max() * 0.05
    gap_start = None
    best_gap_end = 0

    for y in range(scan_h):
        if row_density[y] < threshold:
            if gap_start is None:
                gap_start = y
        else:
            if gap_start is not None and (y - gap_start) > 5:
                best_gap_end = y
            gap_start = None

    return best_gap_end


# ---------------------------------------------------------------------------
# Block detection
# ---------------------------------------------------------------------------

def _classify_components(
    binary: np.ndarray,
    config: PipelineConfig,
    dividers: list[int],
    page_width: int,
    page_height: int,
) -> list[LayoutRegion]:
    """
    Conservative dilation to avoid merging across columns.
    Density-based classification: text > 0.08 density, figures below.
    """
    # Merge chars → words (horizontal, conservative)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
    dilated = cv2.dilate(binary, kernel_h, iterations=2)

    # Merge lines → paragraph blocks (vertical, conservative)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 6))
    dilated = cv2.dilate(dilated, kernel_v, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )

    regions: list[LayoutRegion] = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        bbox = BBox(x=x, y=y, w=w, h=h)

        if area < 200:
            continue

        # Skip full-page background component
        if w > page_width * 0.9 and h > page_height * 0.9:
            continue

        # Density of original binary pixels within this bbox
        roi = binary[y:y+h, x:x+w]
        density = roi.sum() / (255 * max(bbox.area, 1))

        # Classify
        if density < 0.08 and bbox.area > config.figure_min_area:
            kind = BlockKind.FIGURE
            confidence = 0.8
        elif h < config.caption_max_height and w > page_width * 0.15:
            kind = BlockKind.CAPTION if density >= 0.08 else BlockKind.FIGURE
            confidence = 0.6
        else:
            kind = BlockKind.TEXT
            confidence = 0.7

        # Column assignment
        cx = x + w // 2
        col_idx = 0
        for di, div in enumerate(dividers):
            if cx > div:
                col_idx = di + 1

        regions.append(LayoutRegion(
            bbox=bbox, kind=kind, confidence=confidence, column_index=col_idx,
        ))

    return regions


# ---------------------------------------------------------------------------
# Registered step
# ---------------------------------------------------------------------------

@registry.register(
    "detect_layout",
    input_type=PreprocessedImage,
    output_type=tuple,
    description="Detect columns, headers, and classify page regions.",
)
def detect_layout(
    data: PreprocessedImage,
    config: PipelineConfig,
) -> Tuple[PreprocessedImage, LayoutDetection]:

    binary = data.binary
    h, w = binary.shape

    proj = _vertical_projection(binary, config.column_projection_blur)
    dividers = _find_valleys(
        proj, w,
        min_gap_frac=config.column_gap_min_fraction,
        valley_depth=config.column_valley_depth,
        max_columns=config.max_columns,
    )

    n_cols = len(dividers) + 1
    layout_map = {
        1: ColumnLayout.SINGLE,
        2: ColumnLayout.TWO_COLUMN,
        3: ColumnLayout.THREE_COLUMN,
    }
    layout = layout_map.get(n_cols, ColumnLayout.MIXED)

    header_y = _detect_header_cutoff(binary, config.header_scan_fraction)

    regions = _classify_components(binary, config, dividers, w, h)

    for r in regions:
        if r.kind == BlockKind.TEXT and r.bbox.y + r.bbox.h < header_y + 20:
            object.__setattr__(r, "kind", BlockKind.HEADER)

    detection = LayoutDetection(
        layout=layout,
        dividers=dividers,
        regions=regions,
        header_cutoff_y=header_y,
        page_width=w,
        page_height=h,
    )

    return (data, detection)
