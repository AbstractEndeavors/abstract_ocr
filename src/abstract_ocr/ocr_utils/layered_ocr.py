from ..imports import *
def preprocess_image(input_path: Path, output_path: Path) -> None:
        img = cv2.imread(str(input_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        cv2.imwrite(str(output_path), thresh)

# OCR Backends
def tesseract_ocr_img(img: np.ndarray) -> pd.DataFrame:
    pil = Image.fromarray(img)
    df = pytesseract.image_to_data(
        pil, config=Ocr_Config.TESS_PSM,
        output_type=pytesseract.Output.DATAFRAME
    )
    return df[df.text.notnull()]

def easyocr_ocr(path: Path) -> pd.DataFrame:
    reader = easyocr.Reader(Ocr_Config.EASY_LANGS, gpu=True)
    recs = []
    for bbox, text, conf in reader.readtext(str(path)):
        xs, ys = zip(*bbox)
        recs.append({
            'text': text,
            'conf': conf * 100,
            'left': min(xs), 'top': min(ys),
            'width': max(xs)-min(xs), 'height': max(ys)-min(ys)
        })
    return pd.DataFrame(recs)

def paddleocr_ocr(path: Path) -> pd.DataFrame:
    ocr = PaddleOCR(use_angle_cls=Ocr_Config.PADDLE_USE_ANGLE_CLS, lang=Ocr_Config.PADDLE_LANG)
    recs = []
    for page in ocr.ocr(str(path), cls=True):
        if page:    
            for bbox, (text, conf) in page:
                xs, ys = zip(*bbox)
                recs.append({
                    'text': text,
                    'conf': conf,
                    'left': min(xs), 'top': min(ys),
                    'width': max(xs)-min(xs), 'height': max(ys)-min(ys)
                })
    return pd.DataFrame(recs)

def layered_ocr_img(img: np.ndarray, engine='tesseract') -> pd.DataFrame:
    tmp = Path('/tmp/ocr_tmp.png')
    cv2.imwrite(str(tmp), img)

    dfs = []

    if engine == 'tesseract':
        df = tesseract_ocr_img(img)
    elif engine == 'easy':
        df = easyocr_ocr(tmp)
    elif engine == 'paddle':
        df = paddleocr_ocr(tmp)
    else:
        logger.warning(f"Unknown engine '{engine}', skipping")
        return pd.DataFrame(columns=['text', 'left', 'top', 'width', 'height', 'conf'])

    if df is None or df.empty:
        return pd.DataFrame(columns=['text', 'left', 'top', 'width', 'height', 'conf'])

    # Ensure consistent structure
    required_cols = ['text', 'left', 'top', 'width', 'height']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Sort robustly
    try:
        return df.sort_values(['top', 'left']).reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Could not sort dataframe for engine {engine}: {e}")
        return df.reset_index(drop=True)
