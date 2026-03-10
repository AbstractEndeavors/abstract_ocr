import numpy as np
import pandas as pd
from typing import *
from PIL import Image
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import io, os, cv2, PyPDF2, traceback, cv2, re, cv2,logging,os
import PyPDF2, easyocr, pytest, pytesseract, glob,difflib
from abstract_utilities import *
from functools import lru_cache
from pathlib import Path
import spacy
from abstract_utilities import (
    make_dirs,
    get_all_file_types,
    pytesseract,
    path_join,
    get_logFile,
    get_file_parts,
    write_to_file,
    make_list,
    is_number,
    make_dirs,
    get_lazy_attr,
    lazy_import,
    lru_cache
    )
logger = get_logFile(__name__)
from keybert import KeyBERT
from transformers import pipeline
import torch,os,json,unicodedata,hashlib
from transformers import LEDTokenizer,LEDForConditionalGeneration

import spacy,pytesseract,cv2,PyPDF2,argparse,whisper,shutil,os,sys,json,logging,glob,hashlib
from datetime import datetime
from  datetime import timedelta 
from PIL import Image
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
import speech_recognition as sr
from pydub.silence import detect_nonsilent
from pydub.silence import split_on_silence
from pydub import AudioSegment
from abstract_math import (divide_it,
                           multiply_it)
from typing import *
from urllib.parse import quote
from abstract_utilities import (timestamp_to_milliseconds,
                                format_timestamp,
                                get_time_now_iso,
                                parse_timestamp,
                                get_logFile,
                                url_join,
                                make_dirs,
                                safe_dump_to_file,
                                safe_read_from_json,
                                read_from_file,
                                write_to_file,
                                path_join,
                                confirm_type,
                                get_media_types,
                                get_all_file_types,
                                eatInner,
                                eatOuter,
                                eatAll)
                                
import torch
from transformers import pipeline
from keybert import KeyBERT
from transformers import LEDTokenizer, LEDForConditionalGeneration

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
