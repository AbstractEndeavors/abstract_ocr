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

logger = get_logFile('vid_to_aud')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vid_to_aud.log'),
        logging.StreamHandler()
    ]
)

# Initialize recognizer
r = sr.Recognizer()




def create_key_value(json_obj,key,value):
    if key not in json_obj:
        json_obj[key]=value
    return json_obj
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





def initiate_process(target,*args):
    p = Process(target=target, args=args)
    p.start()
    logging.info(f"Started process for: {args}")
def transcribe_all_video_paths(directory=None,output_dir=None,remove_phrases=None,summarizer=None,get_vid_data=None):
    get_vid_data = get_vid_data or False
    remove_phrases=remove_phrases or []
    directory = directory or os.getcwd()
    output_dir = output_dir if output_dir else make_dirs(directory,'text_dir')
    paths = glob.glob(path_join(directory, '**', '**'), recursive=True)
    paths = [file_path for file_path in paths if confirm_type(file_path,media_types=get_media_types(['video']))]
    video_paths = get_all_file_types(directory=directory,types='video') or get_all_file_types(directory=abs_dirname,types='videos')
    for video_path in video_paths:
        info = get_info_data(video_path,output_dir=output_dir,remove_phrases=remove_phrases)
        extract_audio_from_video(video_path=info['video_path'],audio_path=info['audio_path'])
        info = transcribe_audio_file(audio_path=info['audio_path'],json_data=info,summarizer=summarizer,get_vid_data=get_vid_data)
        safe_dump_to_file(data=info,file_path=info['info_path']) 
def get_info_data(video_path,output_dir=None,remove_phrases=None):
    
    remove_phrases=remove_phrases or []
    dirname = os.path.dirname(video_path)
    basename = os.path.basename(video_path)
    filename,ext = os.path.splitext(basename)
    video_directory = make_dirs(output_dir,filename)
    info_path = os.path.join(video_directory,'info.json')
    video_text_path = os.path.join(video_directory,'video_text.json')
    audio_path = os.path.join(video_directory,'audio.wav')
    video_json_path = os.path.join(video_directory,'video_json.json')
    info = {}
    if os.path.isfile(info_path):
        info = safe_read_from_file(info_path)
    info['video_path']=video_path
    info['info_directory']=video_directory
    info['info_path']=info_path
    info['filename']=filename
    info['ext']=ext
    info['remove_phrases']=remove_phrases
    info['audio_path']=audio_path
    info['video_json']=video_json_path
    safe_dump_to_file(data=info,file_path=info['info_path'])
    return info
       
    
