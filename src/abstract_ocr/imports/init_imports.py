from __future__ import annotations
import numpy as np
import pandas as pd
from typing import *
from PIL import Image
from functools import lru_cache
from pathlib import Path
import shutil,os,sys,glob,argparse,io,traceback,re,time
import difflib,json,unicodedata,hashlib,pytest,logging,cv2
from datetime import datetime, timedelta 

from abstract_math import (
    divide_it,
    multiply_it
    )
from urllib.parse import quote
from abstract_utilities import (is_media_type,
                                SingletonMeta,
                                timestamp_to_milliseconds,
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
                                lru_cache)
                                


from enum import Enum, auto
from dataclasses import (
    dataclass,
    field
    )

