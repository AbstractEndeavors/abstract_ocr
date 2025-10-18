from abstract_utilities import *
from src.abstract_ocr import *
video_path = '/home/computron/Videos/Screen_Recording_20210403-075423_Gallery_1.mov'
output_dir = '/home/computron/Documents/pythonTools/modules/abstract_ocr/test_dir'
thumbnail_directory = make_dirs(output_dir,'thumbnails')

#info_data = get_all_info_data_call(video_path=video_path,info_dir=output_dir)
metadata = get_video_metadata(video_path)
extract_audio_from_video(video_path,output_dir)
analyze_video_text(video_path,thumbnail_directory)
whisper_result = transcribe_with_whisper_local(output_dir)
export_srt_whisper(whisper_result,output_dir)
full_text = whisper_result.get('text')
summary = get_summary(full_text)
input(summary)
keywords = refine_keywords(full_text)

summary = refine_with_gpt(full_text,task='summary')
input(summary)
seo_title = refine_with_gpt(full_text,task='seo_title')
input(seo_title)
seo_description = refine_with_gpt(full_text,task='seo_description')
input(seo_description)
seo_keywords = refine_with_gpt(full_text,task='seo_keywords')
input(seo_keywords)

