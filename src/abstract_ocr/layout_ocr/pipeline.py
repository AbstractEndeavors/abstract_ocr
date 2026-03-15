"""
Pipeline orchestrator.

Chains registered steps, validates type compatibility at build time.
No magic — you declare the chain, it runs it.
"""
from .imports import (
    annotations,
    logging,
    time,
    dataclass,
    field,
    Any,
    cv2
    )
from .registry import registry
from .schemas import OCRResult, PageImage, PipelineConfig

# Ensure all steps are registered by importing the package

logger = logging.getLogger(__name__)


@dataclass
class StepTiming:
    name: str
    elapsed_s: float


@dataclass
class PipelineReport:
    result: OCRResult
    timings: list[StepTiming] = field(default_factory=list)
    total_s: float = 0.0


# The default chain — explicit, no auto-discovery.
DEFAULT_CHAIN = ["preprocess", "detect_layout", "ocr_regions"]


class OCRPipeline:
    """
    Wires steps from the registry into an executable chain.

    Usage:
        config = PipelineConfig()
        pipeline = OCRPipeline(config=config)
        # optionally: pipeline = OCRPipeline(config=config, chain=["preprocess", "ocr_regions"])
        result = pipeline.run(page_image)
    """

    def __init__(
        self,
        config: PipelineConfig,
        chain: list[str] | None = None,
    ) -> None:
        self.config = config
        self.chain = chain or DEFAULT_CHAIN

        # Validate at construction, not at runtime
        problems = registry.validate_chain(self.chain)
        if problems:
            raise ValueError(
                "Pipeline chain type mismatch:\n" +
                "\n".join(f"  - {p}" for p in problems)
            )

    def run(self, page: PageImage) -> PipelineReport:
        """Execute the full chain on one page."""
        t0 = time.perf_counter()
        timings: list[StepTiming] = []
        data: Any = page

        for step_name in self.chain:
            step = registry.get(step_name)
            ts = time.perf_counter()

            logger.info("Running step: %s", step_name)
            data = step.callable(data, self.config)

            elapsed = time.perf_counter() - ts
            timings.append(StepTiming(name=step_name, elapsed_s=elapsed))
            logger.info("  %s took %.3fs", step_name, elapsed)

        total = time.perf_counter() - t0
        return PipelineReport(result=data, timings=timings, total_s=total)


def run_on_image(
    image_path: str,
    config: PipelineConfig | None = None,
    chain: list[str] | None = None,
) -> PipelineReport:
    """
    Convenience: load image → run pipeline → return report.
    """
    config = config or PipelineConfig()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    page = PageImage(pixels=img, source_path=image_path)
    pipeline = OCRPipeline(config=config, chain=chain)
    return pipeline.run(page)
