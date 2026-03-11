from .imports import *
from .constants import *
from .splitToChunk import *

def get_file_size(file_path):
    """
    Get the file size in KB.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        str: File size in KB (e.g., "100KB").
    """
    try:
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024  # Convert bytes to KB
        return f"{int(size_kb)}KB"
    except Exception as e:
        print(f"Error getting file size for {file_path}: {e}")
        return "Unknown"

class importManager():
    def __init__(self):
        self.imports = {}
    def get_spacy(self):
        import spacy
        
def create_key_value(json_obj, key, value):
    json_obj[key] = json_obj.get(key, value) or value
    return json_obj

def getPercent(i):
    return divide_it(i, 100)

def getPercentage(num, i):
    percent = getPercent(i)
    percentage = multiply_it(num, percent)
    return percentage

def if_none_get_def(value, default):
    if value is None:
        value = default
    return value

def if_not_dir_return_None(directory):
    str_directory = str(directory)
    if os.path.isdir(str_directory):
        return str_directory
    return None

def determine_remove_text(text,remove_phrases=None):
    remove_phrases=remove_phrases or []
    found = False
    for remove_phrase in remove_phrases:
        if remove_phrase in text:
            found = True
            break
    if found == False:
        return text

def generate_file_id(path: str, max_length: int = 50) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = unicodedata.normalize('NFKD', base).encode('ascii', 'ignore').decode('ascii')
    base = base.lower()
    base = re.sub(r'[^a-z0-9]+', '-', base).strip('-')
    base = re.sub(r'-{2,}', '-', base)
    if len(base) > max_length:
        h = hashlib.sha1(base.encode()).hexdigest()[:8]
        base = base[: max_length - len(h) - 1].rstrip('-') + '-' + h
    return base
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s:.,-]', '', text)
    text = text.strip()
    return text
def get_from_list(list_obj=None,length=1):
    list_obj = list_obj or []
    if len(list_obj) >= length:
        list_obj = list_obj[:length]
    return list_obj
    

def get_image_metadata(file_path):
    """Extract image metadata (dimensions, file size)."""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            file_size = get_file_size(file_path)
        return {
            "dimensions": {"width": width, "height": height},
            "file_size": round(file_size, 3)
        }
    except Exception as e:
        return {"dimensions": {"width": 0, "height": 0}, "file_size": 0}
def update_json_data(json_data,update_data,keys=None):
    if keys == True:
        values_string = ''
        for key,value in update_data.items():
            values_string+= f"{key} == {value}\n"
        logger.info(f"new_datas:\n{values_string}")
        keys = valid_keys
    
    for key,value in update_data.items():
        if keys:
            if key in keys:
                json_data[key] = json_data.get(key) or value 
        else:
            json_data[key] = json_data.get(key) or value 
    return json_data

def update_sitemap(video_data,
                   sitemap_path):
    with open(sitemap_path, 'a') as f:
        f.write(f"""
<url>
    <loc>{video_data.get('canonical_url')}</loc>
    <video:video>
        <video:title>{video_data.get('seo_title')}</video:title>
        <video:description>{video_data.get('seo_description')}</video:description>
        <video:thumbnail_loc>{video_data.get('thumbnail',{}).get('file_path',{})}</video:thumbnail_loc>
        <video:content_loc>{video_data.get('video_path')}</video:content_loc>
    </video:video>
</url>
""")

def execute_if_bool(bool_key,function,keys,req=None,info_data=None):
    new_data,info_data = get_key_vars(keys,req,info_data)
    bool_response = bool_key
    if not isinstance(bool_response,bool):
        bool_response = info_data.get(bool_key) in [None,'',[],"",{}]
    logger.info(f"{bool_key} == {bool_response}")
    if bool_response:
        args, kwargs = prune_inputs(function, **new_data, flag=True)
        info = function(*args, **kwargs)

        info_data = update_json_data(info_data,info,keys=True)
    safe_dump_to_file(data=info_data,file_path=get_video_info_path(**info_data))
    return info_data



def get_video_id(**kwargs):
    info_data = kwargs.get('info_data',kwargs) or kwargs or {}
    info_dir = info_data.get('info_dir') or info_data.get('info_directory')
    video_id = info_data.get('video_id')
    video_path = info_data.get('video_path')
    if info_dir:
        video_id = os.path.basename(info_dir)
    if video_path:
        video_id = generate_file_id(video_path)
    if video_id:
        return video_id
def get_videos_path(directory = None, info_data = None):
    info_data = info_data or {}
    if info_data and directory == None:
        directory = info_data['output_dir']
    directory = directory or TEXT_DIR
    return directory
def get_video_basenames(directory = None, info_data = None):
    directory = get_videos_path(directory = None, info_data = None)
    directory_items = os.listdir(directory)
    return directory_items

def get_videos_paths(directory = None, info_data = None):
    directory = get_videos_path(directory = directory, info_data = info_data)
    video_basenames = get_video_basenames(directory = directory, info_data = directory)
    directory_items = [os.path.join(directory,basename) for basename in video_basenames]
    return directory_items

def get_videos_infos(directory = None, info_data = None):
    directory_items = get_videos_paths(directory = directory, info_data = info_data)
    directory_infos = [get_video_info_data(item_path) for item_path in directory_items]
    return directory_infos

def get_thumbnails_dir(info_dir=None,**kwargs):
    video_info_dir = info_dir or get_video_info_dir(**kwargs)
    thumbnails_directory=os.path.join(video_info_dir,'thumbnails')
    os.makedirs(thumbnails_directory,exist_ok=True)
    return thumbnails_directory

def get_video_info_dir(**kwargs):
    video_id = get_video_id(**kwargs)
    info_dir = make_dirs(TEXT_DIR,video_id)
    os.makedirs(info_dir,exist_ok=True)
    get_thumbnails_dir(info_dir)
    return info_dir

def get_video_info_path(**kwargs):
    info_dir = get_video_info_dir(**kwargs)
    info_path = os.path.join(info_dir,'info.json')
    return info_path

def get_video_info_data(**kwargs):
    info_data=kwargs.get('info_data',kwargs) or kwargs  or {}
    info_file_path = None
    if info_data and isinstance(info_data,str) and os.path.isdir(info_data):
        info_dir = info_data
        info_file_path = os.path.join(info_dir,'info.json')
    elif info_data and isinstance(info_data,str) and os.path.isfile(info_data):
        info_file_path = info_data
    else:
        info_file_path = get_video_info_path(**info_data)
    if os.path.isfile(info_file_path):
        info_data = safe_load_from_json(info_file_path)
        return info_data

def get_audio_path(**kwargs):
    info_dir = get_video_info_dir(**kwargs)
    audio_path = os.path.join(info_dir,'audio.wav')
    return audio_path

def get_audio_bool(**kwargs):
    audio_path = get_audio_path(**kwargs)
    if audio_path:  
        return os.path.isfile(audio_path)
    return False
def get_video_basename(**kwargs):
    video_path = kwargs.get('video_path')
    if not video_path:
        info_data = get_video_info_data(**kwargs)
        video_path = info_data.get('video_path')
    if video_path:
        basename= os.path.basename(video_path)
        return basename
def get_video_filename(**kwargs):
    basename = get_video_basename(**kwargs)
    filename,ext = os.path.splitext(basename)
    return filename
def get_video_ext(**kwargs):
    basename = get_video_basename(**kwargs)
    filename,ext = os.path.splitext(basename)
    return ext
def get_canonical_url(**kwargs):
    video_id = get_video_id(**kwargs)
    videos_url = kwargs.get('videos_url') or kwargs.get('video_url') or VIDEO_URL
    canonical_url = f"{videos_url}/{video_id}"
    return canonical_url
def get_key_vars(keys,req=None,data=None,info_data= None):
    new_data = {}
    if req:
        data,info_data = get_request_info_data(req)
    info_data = info_data or {}
    data = data or info_data
    all_data = data
    for key in keys:
        new_data[key] = all_data.get(key)
        if not new_data[key]:
            if key == 'audio_path':
                new_data[key] = get_audio_path(**all_data)
            elif key == 'video_path':
                new_data[key] = all_data.get('video_path')
            elif key == 'basename':
                new_data[key] = get_video_basename(**all_data)
            elif key == 'filename':
                new_data[key] = get_video_filename(**all_data)
            elif key == 'ext':
                new_data[key] = get_video_ext(**all_data)
            elif key == 'title':
                new_data[key] = get_video_filename(**all_data)
            elif key == 'video_id':
                new_data[key] = get_video_id(**all_data)
            elif key == 'video_path':
                new_data[key] = get_video_path(**all_data)
            elif key == 'thumbnails_directory':
                new_data[key] = get_thumbnails_dir(**all_data)
            elif key == 'model_size':
               new_data[key] = "tiny"
            elif key == 'use_silence':
               new_data[key] = True
            elif key == 'language':
               new_data[key] = "english"
            elif key == 'remove_phrases':
                new_data[key] = REMOVE_PHRASES
            elif key == 'uploader':
                new_data[key] = UPLOADER
            elif key == 'domain':
                new_data[key] = DOMAIN
            elif key == 'categories':
                new_data[key] = CATEGORIES
            elif key == 'videos_url':
                new_data[key] = VIDEOS_URL
            elif key == 'repository_dir':
                new_data[key] = REPO_DIR
            elif key == 'directory_links':
                new_data[key] = DIR_LINKS
            elif key == 'videos_dir':
                new_data[key] = VIDEOS_DIR
            elif key == 'infos_dir':
                new_data[key] = IMGS_DIR
            elif key == 'info_path':
                new_data[key] = get_video_info_path(**all_data)
            elif key in ['info_dir','info_directory']:
                new_data[key] = get_video_info_dir(**all_data)
            elif key == 'base_url':
                new_data[key] = DOMAIN
            elif key == 'generator':
                generator = get_generator()
                new_data[key] = generator
            elif key == 'LEDTokenizer':
                new_data[key] = LEDTokenizer
            elif key == 'LEDForConditionalGeneration':
                new_data[key] = LEDForConditionalGeneration
            elif key == 'full_text':
                new_data[key] = info_data.get('whisper_result',{}).get('text')
            elif key == 'parent_directory':
                new_data[key] = TEXT_DIR
        all_data = update_json_data(all_data,new_data)
    info_data = update_json_data(info_data,all_data,keys=True)
    if 'info_data' in keys:
        new_data['info_data'] =info_data
    if 'json_data' in keys:
        new_data['json_data'] =info_data
    return new_data,info_data

EXT_TO_PREFIX = {
    **dict.fromkeys(
        {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'},
        'infos'
    ),
    **dict.fromkeys(
        {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'},
        'videos'
    ),
    '.pdf': 'pdfs',
    **dict.fromkeys({'.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a'}, 'audios'),
    **dict.fromkeys({'.doc', '.docx', '.txt', '.rtf'}, 'docs'),
    **dict.fromkeys({'.ppt', '.pptx'}, 'slides'),
    **dict.fromkeys({'.xls', '.xlsx', '.csv'}, 'sheets'),
    **dict.fromkeys({'.srt'}, 'srts'),
}
whisper_model_path = DEFAULT_PATHS["whisper"]
