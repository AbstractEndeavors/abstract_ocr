from .imports import *
from .image_utils import *
def make_dir(*paths):
    paths = [str(path) for path in paths]
    return make_dirs(*paths)
# conftest.py
import os,pytest,pytesseract

@pytest.fixture(autouse=True)
def tesseract_env(monkeypatch):
    """
    Ensure that in every test:
     - TESSDATA_PREFIX points at conda’s share/tessdata
     - pytesseract.pytesseract.tesseract_cmd points at the conda binary
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        pytest.skip("CONDA_PREFIX not set; skipping Tesseract-dependent tests")

    tessdata_dir = os.path.join(conda_prefix, "share", "tessdata")
    tesseract_bin = os.path.join(conda_prefix, "bin", "tesseract")

    # 1) env var so tesseract finds its language files
    monkeypatch.setenv("TESSDATA_PREFIX", os.path.dirname(tessdata_dir))
    # 2) point pytesseract at the correct binary
    monkeypatch.setenv("PATH", f"{os.path.join(conda_prefix, 'bin')}:{os.environ.get('PATH', '')}")
    pytesseract.pytesseract.tesseract_cmd = tesseract_bin

    # verify we can actually call it
    try:
        langs = pytesseract.get_languages()
    except pytesseract.TesseractError as e:
        pytest.skip(f"Tesseract not runnable in this env: {e}")

    yield
    # nothing to clean up
def clean_text(text: str) -> str:
    """Clean extracted text using DeepCoder-generated regex patterns."""
    prompt = """
    Write a Python function to clean OCR-extracted text. The function should:
    - Remove excessive whitespace
    - Strip non-alphanumeric characters except spaces, colons, commas, periods, and hyphens
    - Trim leading/trailing spaces
    Return the cleaned text.
    """

    txt = re.sub(r'\s+', ' ', text)
    return re.sub(r'[^A-Za-z0-9\s:.,-]', '', txt).strip()
