from .imports import *
logger = get_logFile('vid_to_aud')
logger.debug(f"Logger initialized with {len(logger.handlers)} handlers: {[h.__class__.__name__ for h in logger.handlers]}")

logOn=True
DOMAIN='https://typicallyoutliers.com'
UPLOADER='The Daily Dialectics'
MAIN_DIR = "/var/www/typicallyoutliers"
FRONTEND_DIR = f"{MAIN_DIR}/frontend"
CATEGORIES = {}
SRC_DIR = f"{FRONTEND_DIR}/src"
BUILD_DIR = f"{FRONTEND_DIR}/build"
PUBLIC_DIR = f"{FRONTEND_DIR}/public"

STATIC_DIR = f"{BUILD_DIR}/static"

IMGS_URL = f"{DOMAIN}/imgs"
IMGS_DIR = f"{PUBLIC_DIR}/imgs"

REPO_DIR = f"{PUBLIC_DIR}/repository"
VIDEOS_URL = f"{DOMAIN}/videos"
VIDEOS_DIR = f"{REPO_DIR}/videos"
VIDEO_DIR = f"{REPO_DIR}/Video"
TEXT_DIR = f"{REPO_DIR}/text_dir"

VIDEO_OUTPUT_DIR = TEXT_DIR
DIR_LINKS = {TEXT_DIR:'infos',VIDEOS_DIR:'videos',REPO_DIR:'repository',IMGS_DIR:'imgs'}
REMOVE_PHRASES = ['Video Converter', 'eeso', 'Auseesott', 'Aeseesott', 'esoft']
DOMAIN='https://typicallyoutliers.com'
UPLOADER='The Daily Dialectics'
VIDEO_OUTPUT_DIR = TEXT_DIR
DIR_LINKS = {TEXT_DIR:'infos',VIDEOS_DIR:'videos',REPO_DIR:'repository',IMGS_DIR:'imgs'}
REMOVE_PHRASES = ['Video Converter', 'eeso', 'Auseesott', 'Aeseesott', 'esoft']
DOMAIN='https://typicallyoutliers.com'
UPLOADER='The Daily Dialectics'
valid_keys =     ['parent_dir', 'video_path', 'info_dir','info_directory', 'thumbnails_directory', 'info_path',
                  'filename', 'ext', 'remove_phrases', 'audio_path', 'video_json', 'categories', 'uploader',
                  'domain', 'videos_url', 'video_id', 'canonical_url', 'chunk_length_ms', 'chunk_length_diff',
                  'renew', 'whisper_result', 'video_text', 'keywords', 'combined_keywords', 'keyword_density',
                  'summary', 'seo_title', 'seo_description', 'seo_tags', 'thumbnail', 'duration_seconds',
                  'duration_formatted', 'captions_path', 'schema_markup', 'social_metadata', 'category',
                  'publication_date', 'file_metadata']
SAMPLE_RATE = whisper.audio.SAMPLE_RATE  # 16000 Hz

