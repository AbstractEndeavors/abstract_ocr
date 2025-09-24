#!/usr/bin/env python3
import os
import glob
import logging
import json
import os,sys,spacy
from multiprocessing import Process
import speech_recognition as sr
from abstract_utilities import *
from moviepy.editor import *
import moviepy.editor as mp
from pydub import AudioSegment
from datetime import timedelta
from .video_utils import derive_all_video_meta
from collections import Counter
r = sr.Recognizer()
logger = get_logFile('vid_to_aud')
nlp = spacy.load("en_core_web_sm")
def format_timestamp(ms):
    """Convert milliseconds to a formatted timestamp (HH:MM:SS.mmm)."""
    td = timedelta(milliseconds=ms)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from a video file using moviepy."""
    try:
        logging.info(f"Extracting audio from {video_path} to {audio_path}")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} was not created.")
        logging.info(f"Audio extracted successfully: {audio_path}")
        return audio_path
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {e}")
        return None
def create_key_value(json_obj,key,value):
    if key not in json_obj:
        json_obj[key]=value
    return json_obj
def check_if_in_time(json_data,start_time,end_time):
    audio_text = json_data.get('audio_text')
    for key,values in audio_text.items():
        value_start_time = values.get("start_time")
        value_end_time = values.get("end_time")
        if value_end_time - value_start_time
def transcribe_audio_file(audio_path, json_data, chunk_length_ms=60000,summarizer=None):
    """Transcribe audio file in chunks and save as text and time-blocked JSON."""
    try:
        logging.info(f"Transcribing audio: {audio_path}")
        audio = AudioSegment.from_wav(audio_path)
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        full_text = ""
        json_data = create_key_value(json_data,'audio_text',[])
        file_path = json_data['info_path']
        for i, chunk in enumerate(chunks):
            start_time = i * chunk_length_ms
            end_time = min((i + 1) * chunk_length_ms, len(audio))
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            logging.info(f"Processing chunk {i+1}/{len(chunks)} ({format_timestamp(start_time)} - {format_timestamp(end_time)})")
            
            with sr.AudioFile(chunk_path) as source:
                r.adjust_for_ambient_noise(source)
                audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data)
                    full_text += text + " "
                    json_data['audio_text'].append({
                        "start_time": format_timestamp(start_time),
                        "end_time": format_timestamp(end_time),
                        "text": text
                    })
                except sr.UnknownValueError:
                    logging.warning(f"Chunk {i+1} could not be transcribed.")
                    json_data['audio_text'].append({
                        "start_time": format_timestamp(start_time),
                        "end_time": format_timestamp(end_time),
                        "text": ""
                    })
                except sr.RequestError as e:
                    logging.error(f"API error for chunk {i+1}: {e}")
                    json_data['audio_text'].append({
                        "start_time": format_timestamp(start_time),
                        "end_time": format_timestamp(end_time),
                        "text": ""
                    })
                safe_dump_to_file(data=json_data,file_path=json_path) 
            os.remove(chunk_path)  # Clean up chunk file

        # Save plain text
        if full_text:
            full_text = full_text.strip()
        text_result= get_voice(full_text, text=full_text)
        json_data = create_key_value(json_data,'full_text',text_result)
        json_data = create_key_value(json_data,'summary',None)
        json_data = create_key_value(json_data,'keywords',extract_keywords_nlp(text_result))
        if summarizer:
            try:
                json_data['summary'] = summarizer(text_result, max_length=160, min_length=40)
            except Exception as e:
                logging.error(f"Error getting summary for {json_data['filename']}: {e}")
        
        return json_data
    except Exception as e:
        logging.error(f"Error transcribing {audio_path}: {e}")
        return json_data

def get_voice(voice, text=None):
    """Save transcribed text to a file."""
    text = text or ''
    if voice:
        text = text + '\n' + str(voice) if text else str(voice)
       
    return text
def extract_keywords_nlp(strings, top_n=5):

    # Process with spaCy
    doc = nlp(strings)
    
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
#!/usr/bin/env python3
import os
import glob
import logging
import json
import sys
import spacy
from datetime import timedelta, datetime
from collections import Counter
from multiprocessing import Process
import fcntl
from transformers import pipeline
from keybert import KeyBERT
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from abstract_utilities import *
from get_video import derive_all_video_meta, analyze_video_text
from transcribe_audio import *
summarizer = pipeline("summarization", model="Falconsai/text_summarization")
keyword_extractor = pipeline("feature-extraction", model="distilbert-base-uncased")
kw_model = KeyBERT(model=keyword_extractor.model)

def get_remove_phrases():
    return ['Video Converter','eeso','Auseesott','Aeseesott','esoft','eeso']
def transcribe_vids(video_path=None):
    video_path = video_path or get_video_dir()
    transcribe_all_video_paths(directory=video_path,output_dir=get_text_dir(),remove_phrases=get_remove_phrases(),summarizer=summarizer,get_vid_data=False)
def get_test_video():
    video_path = '/var/www/typicallyoutliers/frontend/public/repository/Video/Ty Bollinger - CancerTruth.net.mp4'
    transcribe_audio(video_path,output_dir=get_text_dir(),remove_phrases=get_remove_phrases(),summarizer=summarizer,get_vid_data=False)
def get_text_dir():
    text_dir = '/var/www/typicallyoutliers/frontend/public/repository/text_dir'
    return make_dirs(text_dir)
def get_video_dir():
    video_path ='/var/www/typicallyoutliers/frontend/public/repository/Video/'
    return video_path
def get_all_vids():
    transcribe_vids()
processed_files = set()
def check_if_in_time(json_data,start_time,end_time):
    audio_text = json_data.get('audio_text')
    for key,values in audio_text.items():
        value_start_time = values.get("start_time")
        value_end_time = values.get("end_time")
        if value_end_time - value_start_time
def transcribe_audio_file(audio_path, json_data, chunk_length_ms=60000,summarizer=None):
    """Transcribe audio file in chunks and save as text and time-blocked JSON."""
    try:
        logging.info(f"Transcribing audio: {audio_path}")
        audio = AudioSegment.from_wav(audio_path)
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        full_text = ""
        json_data = create_key_value(json_data,'audio_text',[])
        file_path = json_data['info_path']
        for i, chunk in enumerate(chunks):
            start_time = i * chunk_length_ms
            end_time = min((i + 1) * chunk_length_ms, len(audio))
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            logging.info(f"Processing chunk {i+1}/{len(chunks)} ({format_timestamp(start_time)} - {format_timestamp(end_time)})")
            
            with sr.AudioFile(chunk_path) as source:
                r.adjust_for_ambient_noise(source)
                audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data)
                    full_text += text + " "
                    json_data['audio_text'].append({
                        "start_time": format_timestamp(start_time),
                        "end_time": format_timestamp(end_time),
                        "text": text
                    })
                except sr.UnknownValueError:
                    logging.warning(f"Chunk {i+1} could not be transcribed.")
                    json_data['audio_text'].append({
                        "start_time": format_timestamp(start_time),
                        "end_time": format_timestamp(end_time),
                        "text": ""
                    })
                except sr.RequestError as e:
                    logging.error(f"API error for chunk {i+1}: {e}")
                    json_data['audio_text'].append({
                        "start_time": format_timestamp(start_time),
                        "end_time": format_timestamp(end_time),
                        "text": ""
                    })
                safe_dump_to_file(data=json_data,file_path=json_path) 
            os.remove(chunk_path)  # Clean up chunk file

        # Save plain text
        if full_text:
            full_text = full_text.strip()
        text_result= get_voice(full_text, text=full_text)
        json_data = create_key_value(json_data,'full_text',text_result)
        json_data = create_key_value(json_data,'summary',None)
        json_data = create_key_value(json_data,'keywords',extract_keywords_nlp(text_result))
        if summarizer:
            try:
                json_data['summary'] = summarizer(text_result, max_length=160, min_length=40)
            except Exception as e:
                logging.error(f"Error getting summary for {json_data['filename']}: {e}")
        
        return json_data
    except Exception as e:
        logging.error(f"Error transcribing {audio_path}: {e}")
        return json_data
def get_seo_info(info=None, video_path=None, audio_path=None, output_dir=None, remove_phrases=None, summarizer=None, get_vid_data=False):
    logger.info(f"Entering get_seo_info: {video_path}")
    get_vid_data = get_vid_data or False
    remove_phrases = remove_phrases or []

    video_path = video_path or info.get('video_path')
    audio_path = audio_path or info.get('audio_path')
    video_directory = info.get('info_directory')
    info_path = os.path.join(video_directory, 'info.json')
    dirname = os.path.dirname(video_path)
    basename = os.path.basename(video_path)
    filename, ext = os.path.splitext(basename)

    # Avoid redundant video_text processing
    if not info.get('video_text'):
        logger.info(f"Processing video_text: {video_path}")
        video_text = analyze_video_text(video_path, output_dir=info['thumbnails_dir'], video_text=info.get('video_text'), remove_phrases=remove_phrases)
        info['video_text'] = video_text
    else:
        video_text = info['video_text']

    info.update({
        'video_text': video_text,
        'video_path': video_path,
        'info_directory': video_directory,
        'info_path': info_path,
        'filename': filename,
        'ext': ext,
        'audio_path': os.path.join(video_directory, 'audio.wav'),
        'video_json': os.path.join(video_directory, 'video_json.json')
    })

    if get_vid_data and not os.path.exists(info['video_json']):
        initiate_process(derive_all_video_meta, video_path, video_directory, info['video_json'], [], '', filename, remove_phrases, summarizer)

    if not os.path.isfile(audio_path):
        extract_audio_from_video(video_path=info['video_path'], audio_path=info['audio_path'])

    info = transcribe_audio_file(info['audio_path'], info, chunk_length_ms=60000)

    # Add SEO metadata
    primary_keyword = info['combined_keywords'][0] if info['combined_keywords'] else filename
    info['seo_title'] = f"{primary_keyword} - {filename}"[:70]
    summary = info.get('summary', 'No summary available.')
    keywords_str = ', '.join(info['combined_keywords'][:3])
    info['seo_description'] = f"{summary[:150]} Explore {keywords_str}. Visit thedailydialectics.com for more!"[:300]
    info['seo_tags'] = [kw for kw in info['combined_keywords'] if kw.lower() not in ['video', 'audio', 'file']]
    best_frame, score, matched_text = pick_optimal_thumbnail(info['video_text'], info['combined_keywords']) or ('default.jpg', 0, '')
    info['thumbnail'] = {
        'file_path': os.path.join(info['thumbnails_dir'], best_frame),
        'alt_text': matched_text[:100]
    }
    audio = AudioSegment.from_wav(info['audio_path'])
    info['duration_seconds'] = len(audio) / 1000
    info['duration_formatted'] = format_timestamp(len(audio))
    export_srt(info['audio_text'], f"{video_directory}/captions.srt")
    info['captions_path'] = f"{video_directory}/captions.srt"
    info['schema_markup'] = {
        "@context": "https://schema.org",
        "@type": "VideoObject",
        "name": info['seo_title'],
        "description": info['seo_description'],
        "thumbnailUrl": info['thumbnail']['file_path'],
        "duration": f"PT{int(info['duration_seconds'] // 60)}M{int(info['duration_seconds'] % 60)}S",
        "uploadDate": datetime.now().isoformat(),
        "contentUrl": info['video_path'],
        "keywords": info['seo_tags']
    }
    info['social_metadata'] = {
        "og:title": info['seo_title'],
        "og:description": info['seo_description'],
        "og:image": info['thumbnail']['file_path'],
        "og:video": info['video_path'],
        "twitter:card": "player",
        "twitter:title": info['seo_title'],
        "twitter:description": info['seo_description'],
        "twitter:image": info['thumbnail']['file_path']
    }
    categories = {'ai': 'Technology', 'cannabis': 'Health', 'elon musk': 'Business'}
    info['category'] = next((v for k, v in categories.items() if k in ' '.join(info['seo_tags']).lower()), 'General')
    info['uploader'] = {"name": "The Daily Dialectics", "url": "https://thedailydialectics.com"}
    info['publication_date'] = datetime.now().isoformat()
    video = VideoFileClip(info['video_path'])
    info['file_metadata'] = {
        'resolution': f"{video.w}x{video.h}",
        'format': 'MP4',
        'file_size_mb': os.path.getsize(info['video_path']) / (1024 * 1024)
    }
    video.close()
    video_id = filename.replace(' ', '-').lower()
    info['canonical_url'] = f"https://thedailydialectics.com/videos/{video_id}"
    update_sitemap(info, f"{video_directory}/../sitemap.xml")

    safe_dump_to_file(data=info, file_path=info['info_path'])
    logger.info(f"Exiting get_seo_info: {video_path}")
    return info

def transcribe_all_video_paths(directory=None, output_dir=None, remove_phrases=None, summarizer=None, get_vid_data=None):
    get_vid_data = get_vid_data or False
    remove_phrases = remove_phrases or []
    directory = directory or os.getcwd()
    output_dir = output_dir if output_dir else make_dirs(directory, 'text_dir')
    video_paths = get_all_file_types(directory=directory, types='video')
    for video_path in video_paths:
        if video_path not in processed_files:
            processed_files.add(video_path)
            lock_file = "/tmp/vid_to_aud.lock"
            with open(lock_file, 'w') as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    transcribe_audio(video_path, output_dir=output_dir, remove_phrases=remove_phrases, summarizer=summarizer, get_vid_data=get_vid_data)
                except IOError:
                    logger.info(f"Another instance is running, skipping: {video_path}")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
def transcribe_vids(video_path=None):
    video_path = video_path or get_video_dir()
    transcribe_all_video_paths(directory=video_path,output_dir=get_text_dir(),remove_phrases=get_remove_phrases(),summarizer=summarizer,get_vid_data=False)
get_all_vids()
