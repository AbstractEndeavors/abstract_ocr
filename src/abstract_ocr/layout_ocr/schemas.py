"""
Schemas for the layout-aware OCR pipeline.

Every intermediate result is a dataclass. No ad-hoc dicts.
"""
from .imports import (
    annotations,
    dataclass,
    field,
    Enum,
    auto,
    Optional,
    np
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BlockKind(Enum):
    """What a detected region contains."""
    TEXT = auto()
    FIGURE = auto()
    CAPTION = auto()
    HEADER = auto()
    ANNOTATION = auto()
    UNKNOWN = auto()


class ColumnLayout(Enum):
    """Detected page layout."""
    SINGLE = auto()
    TWO_COLUMN = auto()
    THREE_COLUMN = auto()
    MIXED = auto()          # e.g. full-width header + two-column body


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box (x, y, w, h) in pixel coordinates."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / max(self.h, 1)

    def overlaps_x(self, other: BBox, threshold: float = 0.3) -> bool:
        """True if horizontal overlap exceeds threshold of the smaller width."""
        overlap = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        smaller = min(self.w, other.w)
        return (overlap / max(smaller, 1)) > threshold


# ---------------------------------------------------------------------------
# Pipeline data
# ---------------------------------------------------------------------------

@dataclass
class PageImage:
    """Raw page image with its provenance."""
    pixels: np.ndarray          # BGR or grayscale
    source_path: str
    page_number: int = 0
    dpi: int = 300


@dataclass
class PreprocessedImage:
    """Image after denoising + binarisation."""
    gray: np.ndarray            # single-channel denoised
    binary: np.ndarray          # thresholded for component analysis
    original: PageImage


@dataclass
class LayoutRegion:
    """One detected region on the page."""
    bbox: BBox
    kind: BlockKind
    confidence: float = 0.0     # 0-1, how sure we are about `kind`
    column_index: int = -1      # assigned during column assignment


@dataclass
class LayoutDetection:
    """Full layout analysis for one page."""
    layout: ColumnLayout
    dividers: list[int]                     # x-coordinates of column dividers
    regions: list[LayoutRegion] = field(default_factory=list)
    header_cutoff_y: int = 0               # y below which body starts
    page_width: int = 0
    page_height: int = 0


@dataclass
class OCRBlock:
    """OCR result for a single region."""
    text: str
    region: LayoutRegion
    reading_order: int = 0                 # assigned during reassembly


@dataclass
class OCRResult:
    """Final assembled output for one page."""
    blocks: list[OCRBlock]
    raw_text: str                          # blocks joined in reading order
    layout: LayoutDetection
    source: PageImage


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Explicit knobs — no 'smart defaults' hiding behind the scenes.

    Every threshold is here so you can tune without reading source.
    """
    # Preprocessing
    denoise_h: int = 10                         # fastNlMeansDenoising strength
    adaptive_block_size: int = 35               # must be odd
    adaptive_c: int = 11

    # Column detection
    column_projection_blur: int = 51            # smoothing for vertical projection
    column_gap_min_fraction: float = 0.02       # min gap width as fraction of page width
    column_valley_depth: float = 0.15           # how deep a valley must be (0-1)
    max_columns: int = 3

    # Block detection
    text_component_max_area: int = 4000         # larger = figure
    figure_min_area: int = 5000
    caption_max_height: int = 60                # pixels
    header_scan_fraction: float = 0.12          # top % of page to scan for headers

    # OCR
    tesseract_psm: int = 4                      # page segmentation mode
    tesseract_lang: str = "eng"
    tesseract_extra: str = ""                   # any extra --oem / --tessdata flags

    # Reading order
    column_reading_direction: str = "ltr"       # or "rtl"
    sort_within_column: str = "top_to_bottom"
