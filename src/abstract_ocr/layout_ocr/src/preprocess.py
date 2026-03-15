"""
Step: preprocess

PageImage → PreprocessedImage
Denoising + adaptive thresholding.
"""
from ..imports import (
    annotations,
    cv2,
    np
    )
from ..registry import registry
from ..schemas import PageImage, PreprocessedImage, PipelineConfig


@registry.register(
    "preprocess",
    input_type=PageImage,
    output_type=PreprocessedImage,
    description="Denoise and binarise the page image.",
)
def preprocess(data: PageImage, config: PipelineConfig) -> PreprocessedImage:
    img = data.pixels
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    denoised = cv2.fastNlMeansDenoising(gray, h=config.denoise_h)

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,          # invert so text = white
        config.adaptive_block_size,
        config.adaptive_c,
    )

    return PreprocessedImage(gray=denoised, binary=binary, original=data)
