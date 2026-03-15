from .init_imports import *
import spacy,pytesseract,PyPDF2,whisper,easyocr
import torch
import speech_recognition as sr
from pydub.silence import detect_nonsilent, split_on_silence
from pydub import AudioSegment

from keybert import KeyBERT
from transformers import LEDTokenizer,LEDForConditionalGeneration, pipeline
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
logger = get_logFile(__name__)


# --------------------------------------------------
# internal cache
# --------------------------------------------------

_MODELS = {}

# --------------------------------------------------
# summarizer
# --------------------------------------------------

def get_summarizer():
    if "summarizer" not in _MODELS:
        _MODELS["summarizer"] = pipeline(
            "text-generation",
            model="Falconsai/text_summarization"
        )
    return _MODELS["summarizer"]


# --------------------------------------------------
# keyword extractor
# --------------------------------------------------

def get_keyword_extractor():
    if "keyword_extractor" not in _MODELS:
        _MODELS["keyword_extractor"] = pipeline(
            "feature-extraction",
            model="distilbert-base-uncased"
        )
    return _MODELS["keyword_extractor"]


# --------------------------------------------------
# generator
# --------------------------------------------------

def get_generator():
    if "generator" not in _MODELS:
        _MODELS["generator"] = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    return _MODELS["generator"]


# --------------------------------------------------
# keybert
# --------------------------------------------------

def get_kw_model():
    if "kw_model" not in _MODELS:
        extractor = get_keyword_extractor()
        _MODELS["kw_model"] = KeyBERT(model=extractor.model)
    return _MODELS["kw_model"]


# --------------------------------------------------
# LED summarizer (long docs)
# --------------------------------------------------

def get_led():
    if "led" not in _MODELS:
        tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
        model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
        _MODELS["led"] = (tokenizer, model)

    return _MODELS["led"]
