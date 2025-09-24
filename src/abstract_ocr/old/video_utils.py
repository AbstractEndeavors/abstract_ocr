import cv2,pytesseract,os,shutil,logging,spacy,json,re
logging.basicConfig(level=logging.INFO)  # Set level to INFO or higher
from .ocr_utils import *
from moviepy.editor import *
import numpy as np
from abstract_utilities import eatAll,safe_read_from_json,safe_dump_to_file,eatAll,make_dirs
from collections import Counter
# Load spaCy model (run `python -m spacy download en_core_web_sm` first if not installed)
nlp = spacy.load("en_core_web_sm")
def determine_remove_text(text,remove_phrases=None):
    remove_phrases=remove_phrases or []
    found = False
    for remove_phrase in remove_phrases:
        if remove_phrase in text:
            found = True
            break
    if found == False:
        return text
def split_it_out(obj1,obj2):
    obj_3=obj2
    if obj2 and obj1 and obj1.lower() in obj2.lower():
        start =0
        obj_3=''
        obj2_spl = obj2.lower().split(obj1.lower())
        len_obj1 = len(obj1)
        for each in obj2_spl:
            end = start+len(each)
            obj_3 += obj2[start:end]
            start +=len_obj1+len(each)
    return obj_3
def extract_hash_tags(strings):
    hash_tags = []
    hashtags = strings.split('#')
    for hashtag in hashtags:
        hashtag = f"#{hashtag.split(' ')[0]}"
        strings = split_it_out(hashtag,strings)
        hash_tags.append(hashtag[1:])
    return strings,hash_tags
def get_title_description(string,leng = 65):
    # If the string is 40 characters or less, return it as title with empty description
    if len(string) <= leng:
        return string, ""
    
    # Define separators to try splitting on
    separators = ['.', '!', ':', '|', ' ']
    
    # Try each separator
    for sep in separators:
        parts = string.split(sep)
        if len(parts) > 1:  # If the separator exists and splits the string
            title = ""
            for i, part in enumerate(parts):
                # Build the title incrementally
                potential_title = sep.join(parts[:i + 1]) if i > 0 else part
                if len(potential_title) <= leng:
                    title = potential_title
                else:
                    # If adding the next part exceeds 40, use the last valid title
                    if title:  # Ensure we have a title
                        description = sep.join(parts[i:]).strip()
                        # Include the separator in the title if it fits
                        if len(title + sep) <= leng:
                            title += sep
                        return title, description
                    else:
                        # No separator found within 40 yet, continue to next separator
                        break
    
    # If no separators are found within 40 characters, return full string as title
    # and empty description (since we don't want to cut mid-word)
    return string, ""
def is_frame_analyzed(frame_file,video_text_data):
    for values in video_text_data:
        frame = values.get("frame")
        if frame_file == frame:
            text = values.get("text")
            if text:
                return True
            return False
def analyze_video_text(video_path,output_dir=None,json_data=None,remove_phrases=None):
    remove_phrases=remove_phrases or []
    dirname = os.path.dirname(video_path)
    basename = os.path.basename(video_path)
    filename,ext = os.path.splitext(basename)
    json_data =json_data or {}
    output_dir = output_dir or os.path.join(dirname,'./frames')
    os.makedirs(output_dir, exist_ok=True)

    # Load the video
    video = VideoFileClip(video_path)
    duration = video.duration

    # Extract frames every 1 second
    frame_interval = 1  # Adjust as needed
    for t in range(0, int(duration), frame_interval):
        
        frame_path = f"{output_dir}/{filename}_frame_{t}.jpg"
        if not os.path.isfile(frame_path):
            frame = video.get_frame(t)
            cv2.imwrite(frame_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

    # Function to extract text from an image
    def extract_text_from_image_old(image_path,remove_phrases=None):
        remove_phrases=remove_phrases or []
        img = cv2.imread(image_path)
        # Preprocess: Convert to grayscale and apply thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Apply OCR
        text = pytesseract.image_to_string(thresh)
        if text:
            text = text.strip()
        text = determine_remove_text(text,remove_phrases=remove_phrases)
        if text:
            return text
    if 'video_text' not in json_data:
        json_data['video_text']=[]
    # Process all extracted frames
    for frame_file in os.listdir(output_dir):
        if frame_file.endswith(".jpg"):
            if not is_frame_analyzed(frame_file,json_data['video_text']):
                image_dir = f"{output_dir}/{frame_file}"
                text = extract_text_from_image(image_dir)
                text = determine_remove_text(text,remove_phrases=remove_phrases)

                if text:  # Only append non-empty text
                    json_data['video_text'].append({"frame": frame_file, "text": text})
                
    # Clean up (optional)
    return json_data
def get_file(filename,directory):
    file=None
    for item in os.listdir(directory):
        basename = os.path.splitext(item)[0]
        if basename and str(filename) == str(basename):
            file = item
            break
    return file
def get_thumbnail_texts(directory,remove_phrases=None):
    remove_phrases=remove_phrases or []
    texts = []
    if not directory.endswith('thumbnails'):
        directory = make_dirs(directory,'thumbnails')
    if os.path.isdir(directory):
        for thumbnail in os.listdir(directory):
            thumbnail_path = os.path.join(directory,thumbnail)
            text= convert_image_to_text(thumbnail_path)
            text = determine_remove_text(text,remove_phrases=remove_phrases)
            if text:
                texts.append(text)
    return texts
def get_constants(text,constants=[]):
    if constants == []:
        for line in text.split('\n'):
            line = eatAll(line,[' ','','\n','\t','\n'])
            constants.append(line)
        return constants
    constants = {const:False for const in constants}
    for line in text.split('\n'):
        line = eatAll(line,[' ','','\n','\t','\n'])
        for key,value in constants.items():
            if line in key and value == False:
                constants[key] = True
    constants_output=[]
    for key,value in constants.items():
        if value == True:
            constants_output.append(key)
    return constants_output
def get_text_constants(video_text,remove_phrases=None):
    remove_phrases=remove_phrases or []
    constants={}
    video_text_length = len(video_text)
    for j,frame in enumerate(video_text):
        text = frame.get('text')
        frame = int(os.path.splitext(frame.get('frame'))[0].split('_')[-1])
        text_spl = text.split('\n')
        text_spl_len = len(text_spl)
        for i,line in enumerate(text_spl):
            line = eatAll(line,[' ','','\n','\t','\n'])
            line = determine_remove_text(line,remove_phrases=remove_phrases)
            if line:
                if line not in constants:
                    constants[line]={"count":0,"positions":[]}
                constants[line]["count"]+=1
                constants[line]["positions"].append({"count":i+1,"of":text_spl_len,"frame":frame})
    return constants
def derive_video_info(data,keywords=[],description='',title=''):

    # Step 1: Preprocess the data
    def preprocess_text(text):
        # Remove special characters, normalize case, and fix common OCR errors
        text = re.sub(r'[^\w\s@]', '', text).strip().lower()
        return text

    # Extract phrase frequencies
    phrase_counts = {preprocess_text(phrase): info["count"] for phrase, info in data.items()}
    def extract_keywords_nlp(data, top_n=5):
        # Combine all phrases into a single text, weighted by count
        combined_text = " ".join([preprocess_text(phrase) * info["count"] for phrase, info in data.items()])
        
        # Process with spaCy
        doc = nlp(combined_text)
        
        # Extract nouns, proper nouns, and entities
        word_counts = Counter()
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 2:
                word_counts[token.text] += 1
        
        # Extract multi-word entities (e.g., "Elon Musk")
        entity_counts = Counter(ent.text.lower() for ent in doc.ents if len(ent.text.split()) > 1)
        
        # Combine and rank
        combined_counts = word_counts + entity_counts
        top_keywords = [word for word, count in combined_counts.most_common(top_n)]
        
        return top_keywords
    keywords_nlp = extract_keywords_nlp(data)
    keywords +=keywords_nlp
    # Step 2: Extract Title
    def get_title(phrase_counts, min_count=10):
        # Filter out noise and low-frequency phrases
        valid_phrases = {phrase: count for phrase, count in phrase_counts.items() 
                         if count > min_count and phrase and not phrase.startswith("~~~") and len(phrase.split()) > 1}
        try:
            # Sort by frequency and pick the top phrase
            top_phrase = max(valid_phrases.items(), key=lambda x: x[1])[0]
        except:
            top_phrase=''

        return top_phrase.capitalize()

    title+= ' '+ get_title(phrase_counts)

    # Step 3: Generate Description
    def get_description(phrase_counts, top_n=5):
        # Define keywords related to common video themes
        
        # Get top N frequent phrases
        top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Filter phrases with keywords
        relevant_phrases = [phrase for phrase, count in top_phrases 
                            if any(kw in phrase for kw in keywords) or count > 20]
        
        # Simple template-based description
        desc = description
        
        
        themes = [phrase for phrase in relevant_phrases if len(phrase.split()) > 1]
        desc += ", ".join(themes[:3])
        return desc

    description = get_description(phrase_counts)

    # Step 4: Determine Uploader
    def get_uploader(phrase_counts):
        # Look for phrases that look like handles or names
        candidates = [phrase for phrase in phrase_counts.keys() 
                      if phrase.startswith("@") or phrase.isupper()]
        
        if not candidates:
            return "Unknown"
        
        # Pick the most frequent candidate
        uploader = max(candidates, key=lambda x: phrase_counts[x])
        return uploader

    uploader = get_uploader(phrase_counts)

    return {"description":description,"uploader":uploader,"title":title,'keywords':keywords}
def extract_subtitles(text_contents, total_frames, frequency_threshold=0.2, min_subtitle_count=1, max_subtitle_count=10):

    
    # Step 1: Filter out static text
    subtitles = []
    static_texts = set()
    
    for text, data in text_contents.items():
        frame_count = data["count"]
        positions = data["positions"]
        
        # Skip empty strings or very high-frequency static text
        if not text.strip() or frame_count > total_frames * frequency_threshold:
            static_texts.add(text)
            continue
        
        # Check if the text appears at a fixed position (e.g., watermark)
        position_counts = set(pos["count"] for pos in positions)
        total_lines_per_frame = set(pos["of"] for pos in positions)
        if len(position_counts) == 1 and frame_count > total_frames * 0.1:  # Fixed position, frequent
            static_texts.add(text)
            continue
        
        # Likely a subtitle if count is within a reasonable range
        if min_subtitle_count <= frame_count <= max_subtitle_count:
            subtitles.append({
                "text": text,
                "frames": sorted([pos["frame"] for pos in positions]),
                "count": frame_count
            })
    
    # Step 2: Group subtitles by frame ranges
    subtitle_timeline = {}
    for subtitle in subtitles:
        frames = subtitle["frames"]
        text = subtitle["text"]
        # Group consecutive frames
        start_frame = frames[0]
        for i in range(1, len(frames)):
            if frames[i] > frames[i-1] + 1:  # Gap detected
                subtitle_timeline.setdefault((start_frame, frames[i-1]), []).append(text)
                start_frame = frames[i]
        subtitle_timeline.setdefault((start_frame, frames[-1]), []).append(text)
    
    # Step 3: Format output
    formatted_subtitles = []
    for (start, end), texts in sorted(subtitle_timeline.items()):
        formatted_subtitles.append({
            "start_frame": start,
            "end_frame": end,
            "text": " ".join(texts)  # Combine if multiple lines in same range
        })
    
    return formatted_subtitles, list(static_texts)
def derive_all_video_meta(video_path,output_dir=None,video_text_path=None,keywords=None,description=None,title=None,remove_phrases = None,summarizer=None):
    remove_phrases=remove_phrases or []
    video_dir = os.path.dirname(video_path)
    info_path = os.path.join(video_dir,'info.json')
    if os.path.isfile(info_path):
        info_data = safe_read_from_json(info_path)
        keywords = keywords or info_data.get('context',{}).get('keywords',[])
        description = description or info_data.get('context',{}).get('description','')
        title = title or info_data.get('context',{}).get('title','')
    if isinstance(keywords,str):
        keywords = [eatAll(keyword,[' ','#',',','\t']) for keyword in keywords.split(',') if keyword]
    if video_text_path == None:
        text_dir = os.path.join(video_dir,'video_text')
        os.makedirs(text_dir, exist_ok=True)
        video_text_path = os.path.join(text_dir,'video_text.json')
    basename = os.path.basename(video_path)
    filename,ext = os.path.splitext(basename)
    thumbnails_dir = make_dirs(output_dir,'thumbnails')
    video_text = analyze_video_text(video_path,output_dir=thumbnails_dir,remove_phrases=remove_phrases)
    text_constants = get_text_constants(video_text,remove_phrases=remove_phrases)
    thumbnail_texts = get_thumbnail_texts(thumbnails_dir,remove_phrases=remove_phrases)
    video_info = derive_video_info(text_constants,keywords,description,title)
    description = video_info.get('description','') or video_info.get('context',{}).get('description','')
    title = video_info.get('title','') or video_info.get('context',{}).get('title','')
    keywords = video_info.get('keywords','') or video_info.get('context',{}).get('keywords','') 
    if len(thumbnail_texts)>0:
        if not description:
            description = thumbnail_texts[0]
        if not title:
            title = thumbnail_texts[0]
    if isinstance(keywords,str):
        keywords  = keywords.split(',')
    subtitles = extract_subtitles(text_constants, len(video_text))
    keywords = [eatAll(keyword,[' ','\t','\n','#',',']) for keyword in keywords if eatAll(keyword,[' ','\t','\n','#',',']) and eatAll(keyword,[' ','\t','\n','#',',']) not in ["bolshevid","clownworld"]]
    title = title.replace(str(filename),'').split('reactions | ')[-1]
    title , keywords_spl = extract_hash_tags(title or '')
    keywords+=keywords_spl
    title,description_spl = get_title_description(title)
    description , keywords_spl = extract_hash_tags(description or '')
    keywords+=keywords_spl
    keywords = list(set(keywords))
    keywords_str = ','.join(keywords)
    description = description + ' '+description_spl
    summary=''
    
    if summarizer:
        all_text = ' '.join(thumbnail_texts)
        sum_text = filename+'\n'+description+'\n'+all_text
        try:
            summary = summarizer(sum_text, max_length=160, min_length=40)
        except Exception as e:
            logging.error(f"Error getting summary for {filename}: {e}")
        # Save JSON
    video_info['keywords'] = keywords
    video_info['title'] = title
    video_info['description'] = description
    video_json = {
        'summary':summary,
        "video_file_path": video_path,
        'thumbnail_texts': thumbnail_texts,
        "video_info": video_info,
        "video_text": video_text,
        "text_constants": text_constants,
        "subtitles": subtitles,
        "thumbnails_dir":thumbnails_dir,
    }
    safe_dump_to_file(data=video_json, file_path=video_text_path)
    return video_json
