"""
Step: ocr_regions

(PreprocessedImage, LayoutDetection) → OCRResult

Runs Tesseract on each text region, assigns reading order, assembles output.
"""
from ..imports import (
    annotations,
    pytesseract,
    cv2,
    np,
    Tuple
    )

from ..registry import registry
from ..schemas import (
    BlockKind,
    LayoutDetection,
    OCRBlock,
    OCRResult,
    PipelineConfig,
    PreprocessedImage,
)


def _crop_region(gray: np.ndarray, bbox) -> np.ndarray:
    """Extract and pad a region for better OCR."""
    pad = 8
    y1 = max(0, bbox.y - pad)
    y2 = min(gray.shape[0], bbox.y2 + pad)
    x1 = max(0, bbox.x - pad)
    x2 = min(gray.shape[1], bbox.x2 + pad)
    crop = gray[y1:y2, x1:x2]

    # Add white border for tesseract
    bordered = cv2.copyMakeBorder(
        crop, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT, value=255,
    )
    return bordered


def _reading_order_key(block: OCRBlock, direction: str) -> tuple:
    """
    Sort key for reading order:
      1. Column index (left→right or right→left)
      2. Y position (top→bottom)
    """
    col = block.region.column_index
    if direction == "rtl":
        col = -col
    return (col, block.region.bbox.y)


@registry.register(
    "ocr_regions",
    input_type=tuple,        # (PreprocessedImage, LayoutDetection)
    output_type=OCRResult,
    description="OCR each text region and assemble in reading order.",
)
def ocr_regions(
    data: Tuple[PreprocessedImage, LayoutDetection],
    config: PipelineConfig,
) -> OCRResult:

    preprocessed, layout = data
    gray = preprocessed.gray

    tess_config = (
        f"--psm {config.tesseract_psm} "
        f"-l {config.tesseract_lang} "
        f"{config.tesseract_extra}"
    ).strip()

    blocks: list[OCRBlock] = []

    # If no regions detected, fall back to full-page OCR per column
    if not layout.regions:
        blocks = _fallback_column_ocr(gray, layout, tess_config)
    else:
        for region in layout.regions:
            if region.kind == BlockKind.FIGURE:
                continue  # skip figures

            crop = _crop_region(gray, region.bbox)

            # For captions/annotations, use PSM 6 (single block)
            rc = tess_config
            if region.kind in (BlockKind.CAPTION, BlockKind.ANNOTATION):
                rc = tess_config.replace(
                    f"--psm {config.tesseract_psm}", "--psm 6"
                )

            text = pytesseract.image_to_string(crop, config=rc).strip()
            if text:
                blocks.append(OCRBlock(text=text, region=region))

    # Assign reading order
    blocks.sort(key=lambda b: _reading_order_key(b, config.column_reading_direction))
    for i, block in enumerate(blocks):
        block.reading_order = i

    # Separate headers, body text, captions
    headers = [b for b in blocks if b.region.kind == BlockKind.HEADER]
    body = [b for b in blocks if b.region.kind == BlockKind.TEXT]
    captions = [b for b in blocks if b.region.kind in (BlockKind.CAPTION, BlockKind.ANNOTATION)]

    ordered = headers + body + captions
    raw_text = "\n\n".join(b.text for b in ordered)

    return OCRResult(
        blocks=ordered,
        raw_text=raw_text,
        layout=layout,
        source=preprocessed.original,
    )


def _fallback_column_ocr(
    gray: np.ndarray,
    layout: LayoutDetection,
    tess_config: str,
) -> list[OCRBlock]:
    """
    Fallback: slice by detected dividers and OCR each column strip.
    Used when block detection yields no regions.
    """
    from ..schemas import BBox, BlockKind, LayoutRegion

    h, w = gray.shape
    boundaries = [0] + layout.dividers + [w]
    blocks: list[OCRBlock] = []

    for col_idx in range(len(boundaries) - 1):
        x1 = boundaries[col_idx]
        x2 = boundaries[col_idx + 1]

        # Slight inward margin to avoid gutter noise
        margin = max(5, int((x2 - x1) * 0.02))
        x1_m = x1 + margin
        x2_m = x2 - margin

        col_img = gray[:, x1_m:x2_m]
        bordered = cv2.copyMakeBorder(
            col_img, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, value=255,
        )

        text = pytesseract.image_to_string(bordered, config=tess_config).strip()
        if text:
            bbox = BBox(x=x1, y=0, w=x2 - x1, h=h)
            region = LayoutRegion(
                bbox=bbox,
                kind=BlockKind.TEXT,
                confidence=0.5,
                column_index=col_idx,
            )
            blocks.append(OCRBlock(text=text, region=region))

    return blocks
