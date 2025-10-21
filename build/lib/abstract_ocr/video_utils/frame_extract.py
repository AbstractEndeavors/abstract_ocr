from ..imports import *
from .preprocess import *
from .text_extract import *
def extract_text_from_image(image_path: str) -> str:
    try:
        processed_img = preprocess_for_ocr(image_path)
        pil_img = Image.fromarray(cv2.bitwise_not(processed_img))  # invert for OCR
        text = pytesseract.image_to_string(pil_img, lang='eng')
        return text
    except Exception as e:
        print(f"[OCR Error] {e}")
        return ""
def is_frame_analyzed(frame_file,video_text_data):
    for values in video_text_data:
        frame = values
        if isinstance(values,dict):
            frame = values.get("frame")
        if frame_file == frame:
            return True
def extract_image_text(image_path,remove_phrases=None):
    remove_phrases = remove_phrases or []
    if is_media_type(image_path,media_types=['image']):
        raw_text = extract_text_from_image(image_path)
        cleaned_text = clean_text(raw_text)
        text = determine_remove_text(cleaned_text,remove_phrases=remove_phrases)
        return text
def extract_text_from_frame(image_path,image_texts,remove_phrases=None):
    basename = os.path.basename(image_path)
    if not is_frame_analyzed(basename,image_texts):
        text = extract_image_text(image_path,remove_phrases)
        if text:
            image_texts.append( {"frame": basename, "text": text})
    return image_texts
def extract_image_texts_from_directory(directory,image_texts=None,remove_phrases=None):
    image_texts = image_texts or []
    image_files = get_all_file_types(types=['image'],directory=directory)
    for i,image_path in enumerate(image_files):
        image_texts = extract_text_from_frame(image_path,image_texts)
        
    image_texts = sort_frames(image_texts)
    return image_texts
