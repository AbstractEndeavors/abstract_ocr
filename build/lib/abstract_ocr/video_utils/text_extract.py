from ..imports import *
from .preprocess import *

def convert_image_to_text(image_path: str,preprocess=True) -> str:
    try:
        if preprocess:
            img = preprocess_for_ocr(image_path)
        else:
            img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


def determine_remove_text(text,remove_phrases=None):
    remove_phrases=remove_phrases or []
    found = False
    for remove_phrase in remove_phrases:
        if remove_phrase in text:
            found = True
            break
    if found == False:
        return text
