from abstract_hugpy import *
import os,json,unicodedata,hashlib,PyPDF2,argparse,shutil,os,json,shutil,whisper
import sys,json,logging,glob,hashlib,re,math,pytesseract,cv2,os,json,shutil
from datetime import datetime,timedelta 
from PIL import Image
import numpy as np
from pathlib import Path
import moviepy.editor as mp
from pdf2image import convert_from_path
import speech_recognition as sr
from pydub.silence import detect_nonsilent,split_on_silence
from pydub import AudioSegment
from abstract_math import (divide_it, multiply_it)
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
                                eatAll,
                                get_all_file_types,
                                is_media_type,
                                safe_load_from_json,
                                prune_inputs)
# conftest.py

logger = get_logFile(__name__)
