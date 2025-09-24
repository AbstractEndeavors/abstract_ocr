from .functions import (logger,
                        create_key_value,
                        timestamp_to_milliseconds,
                        format_timestamp,
                        parse_timestamp,
                        get_time_now_iso,
                        shutil,
                        os,
                        safe_dump_to_file,
                        if_not_dir_return_None,
                        if_none_get_def,
                        getPercentage,
                        getPercent,
                        create_key_value)
from moviepy.editor import *
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
r = sr.Recognizer()
def get_elapsed_time(start_time, end_time):
    elapsed = end_time - start_time
    if end_time < start_time:
        elapsed = 0
    return elapsed

def get_time_diff(start_time, end_time, chunk_length_diff):
    elapsed = get_elapsed_time(start_time, end_time)
    percentage = getPercentage(elapsed, chunk_length_diff)
    new_start = subtract_it(start_time, percentage)
    if new_start <= 0:
        new_start = start_time
    return new_start

def check_if_transcribed_block(audio_text, start_time, end_time):
    for entry in audio_text:
        entry_start = timestamp_to_milliseconds(entry.get("start_time", "0:00"))
        entry_end = timestamp_to_milliseconds(entry.get("end_time", "0:00"))
        if entry_start <= start_time and entry_end >= end_time and entry.get("text", "").strip():
            return True
    uncovered_start, uncovered_end = start_time, end_time
    for entry in audio_text:
        entry_start = timestamp_to_milliseconds(entry.get("start_time", "0:00"))
        entry_end = timestamp_to_milliseconds(entry.get("end_time", "0:00"))
        if not entry.get("text", "").strip():
            continue
        if entry_start <= uncovered_start < entry_end:
            uncovered_start = entry_end
        if entry_start < uncovered_end <= entry_end:
            uncovered_end = entry_start
    return (uncovered_start, uncovered_end)

def get_chunks(audio, chunk_length_ms=None, chunk_length_diff=None):
    chunk_length_ms = if_none_get_def(chunk_length_ms, 50000)
    chunk_length_diff = if_none_get_def(chunk_length_diff, -5)
    chunks = [audio[i:i + (chunk_length_ms - chunk_length_diff)] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def get_text(audio_data):
    try:
        text = r.recognize_google(audio_data)
    except:
        text = ""
    return text

def get_audio_chunk(start_time, end_time, audio, chunk_path=None):
    chunk_path = chunk_path or f"{os.getcwd()}/chunk_partial.wav"
    audio[start_time:end_time].export(chunk_path, format="wav")
    with sr.AudioFile(chunk_path) as source:
        r.adjust_for_ambient_noise(source)
        audio_data = r.record(source)
        return audio_data

def get_chunk_period(i, audio_text, chunk_length_ms, chunk_length_diff, audio_len, renew=False):
    start_time = i * chunk_length_ms
    end_time = min((i + 1) * chunk_length_ms, audio_len)
    if i == 0:
        end_time += chunk_length_diff
    result = (start_time, end_time)
    if not renew:
        result = check_if_transcribed_block(audio_text, start_time, end_time)
    return result

def get_audio_text(start_time, end_time, audio, chunk_path=None):
    audio_data = get_audio_chunk(start_time=start_time,
                                 end_time=end_time,
                                 audio=audio,
                                 chunk_path=chunk_path)
    txt = get_text(audio_data)
    return txt

def get_audio_text_data(i,
                        audio,
                        json_data=None,
                        chunk_length_ms=None,
                        chunk_length_diff=None,
                        file_path=None,
                        renew=False,
                        output_directory=None):
    json_data = json_data or {}
    json_data = create_key_value(json_data, 'audio_text', [])
    output_directory = if_none_get_def(if_not_dir_return_None(output_directory), os.getcwd())
    chunk_path = os.path.join(output_directory, f"chunk_{i}_partial.wav")
    json_data = create_key_value(json_data,
                     'chunk_length_ms',
                     chunk_length_ms or 50000)
    json_data = create_key_value(json_data,
                     'chunk_length_diff',
                     chunk_length_diff or -5)
    json_data = create_key_value(json_data,
                     'renew',
                     renew or False)
    result = get_chunk_period(i,
                              audio_text=json_data['audio_text'],
                              chunk_length_ms=json_data['chunk_length_ms'],
                              chunk_length_diff=json_data['chunk_length_diff'],
                              audio_len=len(audio),
                              renew=json_data['renew'])
    if isinstance(result, tuple):
        logger.info(f"transcribing audio chunk {i}, start_time:{result[0]}, end_time:{result[1]}")
        text = get_audio_text(start_time=result[0],
                              end_time=result[1],
                              audio=audio,
                              chunk_path=chunk_path)
        json_data['audio_text'].append({
            "start_time": format_timestamp(result[0]),
            "end_time": format_timestamp(result[1]),
            "text": text
        })
        if file_path:
            safe_dump_to_file(json_data, file_path)
    else:
        logger.info(f"audio text already transcribed for chunk {i}")
    if os.path.isfile(chunk_path):
        try:
            os.remove(chunk_path)
        except OSError:
            logger.warning(f"Failed to remove chunk file: {chunk_path}")
    return json_data

def get_audio_text_jsons(audio_path,
                         json_data=None,
                         chunk_length_ms=None,
                         chunk_length_diff=None,
                         output_directory=None,
                         file_path=None,
                         renew=False):
    json_data = json_data or {}
    audio = AudioSegment.from_wav(audio_path)
    chunks = get_chunks(audio,
                        chunk_length_ms=chunk_length_ms,
                        chunk_length_diff=chunk_length_diff)
    len_chunks = len(chunks)
    for i in range(len_chunks):
        json_data = get_audio_text_data(i,
                                        audio=audio,
                                        json_data=json_data,
                                        chunk_length_ms=chunk_length_ms,
                                        chunk_length_diff=chunk_length_diff,
                                        file_path=file_path,
                                        renew=renew,
                                        output_directory=output_directory)
        logger.info(f"chunk {i} of {len_chunks} processed")
    return json_data

def transcribe_audio_file(audio_file=None, json_data=None, video_path=None, chunk_length_ms=8000, chunk_length_diff=-5, renew=False):
    logger.info(f"transcribe_audio_file: {audio_path}")
    json_data = json_data or {}
    audio_path = audio_path or json_data.get('audio_path')
    video_path = video_path or json_data.get('video_path')
    json_data = create_key_value(json_data, 'video_path', video_path)
    file_path = json_data.get('info_path') or os.path.join(os.getcwd(), 'info.json')
    json_data = create_key_value(json_data, 'info_path', file_path)
    info_directory = json_data.get('info_directory') or os.path.dirname(file_path)
    json_data = create_key_value(json_data, 'info_directory', info_directory)
    json_data = create_key_value(json_data, 'audio_text', [])
    if not audio_path and not video_path:
        logger.info(f"no audio_path and no video_path: {file_path}")
        return json_data
    if not audio_path and video_path:
        audio_path = os.path.join(os.path.dirname(video_path), 'audio.wav')
        logger.info(f"audio path derived from video_path:{video_path}")
        json_data = create_key_value(json_data, 'audio_path', audio_path)
    if not os.path.isfile(audio_path) and video_path:
        extract_audio_from_video(video_path, audio_path)
        logger.info(f"extracted audio from video:{audio_path}")
    temp_file_path = file_path.replace('.json', '_temp.json')
    json_data = get_audio_text_jsons(audio_path=audio_path,
                                     json_data=json_data,
                                     chunk_length_ms=chunk_length_ms,
                                     chunk_length_diff=chunk_length_diff,
                                     output_directory=info_directory,
                                     file_path=temp_file_path,
                                     renew=renew)
    if os.path.isfile(temp_file_path):
        try:
            shutil.move(temp_file_path, file_path)
            logger.info(f"temp_info_path transfered to info_path:{file_path}")
        except OSError as e:
            logger.error(f"Failed to move temp file to {file_path}: {e}")
    return json_data
def extract_audio_from_video(video_path, audio_path):
    """Extract audio from a video file using moviepy."""
    try:
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} was not created.")
        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {e}")
        return None
    
def get_voice(voice, text=None):
    """Save transcribed text to a file."""
    text = text or ''
    if voice:
        text = text + '\n' + str(voice) if text else str(voice)
    return text
def stitch_text(last_text, text):
    og_text = text
    text = eatInner(text, [' ', '\n', '\t'])
    last_text = eatAll(last_text, [' ', '\n', '\t'])
    last_text_spl = last_text.split(' ')[-1]
    end_match = False
    spl_len = len(last_text_spl)
    text_len = len(text)
    j = 0
    for i in range(spl_len):
        k = i - j
        is_valid = k < text_len
        if end_match == True or not is_valid:
            if last_text_spl[j:i] != text[:k] or spl_len == i + 1 or not is_valid:
                if not is_valid and k > 0:
                    k = k - 1
                og_text = text[k:]
                break
        if end_match == False and last_text_spl[i] == text[0]:
            j = i
            end_match = True
    return og_text
