from typing import Dict, Any, Optional
from pathlib import Path
import os
import shutil
from datetime import datetime
import unicodedata
import re
import hashlib
from abstract_utilities import safe_read_from_json, safe_dump_to_file, get_logFile

# Constants
BASE_URL = "https://typicallyoutliers.com"
TEXT_DIR = '/var/www/typicallyoutliers/frontend/public/repository/text_dir/'
VIDEO_DIR = '/var/www/typicallyoutliers/frontend/public/repository/Video/'
DIR_LINKS = {TEXT_DIR: 'infos', VIDEO_DIR: 'videos'}
logger = get_logFile('video_utils')

class videoPathManager:
    def __init__(self):
        self.video_paths = []
        self.video_ids = os.listdir(TEXT_DIR) if os.path.exists(TEXT_DIR) else []
        self.video_directories = self.get_all_video_paths()

    def get_all_video_paths(self) -> Dict[str, Dict[str, str]]:
        video_directories = {
            video_id: {"directory": os.path.join(TEXT_DIR, video_id)}
            for video_id in self.video_ids
        }
        for video_id in video_directories:
            info_path = os.path.join(video_directories[video_id]["directory"], 'info.json')
            info_data = safe_read_from_json(info_path) or {}
            video_directories[video_id]['video_path'] = info_data.get('video_path', '')
        return video_directories

    def get_video_path(self, video_id: str) -> Optional[str]:
        return self.video_directories.get(video_id, {}).get('video_path')

    def get_video_id(self, video_path: str) -> Optional[str]:
        for video_id, values in self.video_directories.items():
            if video_path == values.get('video_path'):
                return video_id
        video_id = generate_video_id(video_path)
        self.video_ids.append(video_id)
        self.add_path(video_id, video_path)
        return video_id

    def add_path(self, video_id: str, video_path: str) -> Dict[str, str]:
        if video_id not in self.video_directories:
            self.video_directories[video_id] = {}
        self.video_directories[video_id]['video_path'] = video_path
        self.video_directories[video_id]["directory"] = os.path.join(TEXT_DIR, video_id)
        return self.video_directories[video_id]

# Instantiate videoPathManager
video_path_mgr = videoPathManager()

def generate_video_id(path: str, max_length: int = 50) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = unicodedata.normalize('NFKD', base).encode('ascii', 'ignore').decode('ascii')
    base = base.lower()
    base = re.sub(r'[^a-z0-9]+', '-', base).strip('-')
    base = re.sub(r'-{2,}', '-', base)
    if len(base) > max_length:
        h = hashlib.sha1(base.encode()).hexdigest()[:8]
        base = base[:max_length - len(h) - 1].rstrip('-') + '-' + h
    return base

def get_link(path: Optional[str]) -> Optional[str]:
    if path:
        for key, value in DIR_LINKS.items():
            if path.startswith(key):
                rel_path = path[len(key):]
                return f"{BASE_URL}/{value}/{rel_path}"
    return None

def get_complete_video_data(video_path: Optional[str] = None, video_id: Optional[str] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Efficiently retrieve all available video data, minimizing resource usage by reusing existing data.
    Returns a unified dictionary with video metadata, transcription, thumbnails, and SEO data.

    Args:
        video_path (str, optional): Absolute path to the video file.
        video_id (str, optional): Unique identifier for the video (from TEXT_DIR).
        force_refresh (bool): If True, reprocess all data (e.g., retranscribe audio).

    Returns:
        Dict[str, Any]: Comprehensive video data, with defaults for missing fields.
    """
    # Initialize default response
    video_data = {
        "id": None,
        "title": "Untitled Video",
        "embed": "",
        "description": "Check out this video",
        "keywords_str": "",
        "thumbnail_url": "/var/www/typicallyoutliers/frontend/public/imgs/no_image.jpg",
        "contentUrl": "",
        "video_url": "",
        "optimized_video_url": "",
        "ext": "",
        "mime_type": "video/mp4",
        "category": "Education",
        "transcript": "",
        "captions": None,
        "schema_markup": {},
        "social_meta": {},
        "publication_date": datetime.now().strftime("%Y-%m-%d"),
        "file_metadata": {},
        "thumbnail_metadata": {},
        "audio_text": [],
        "error": None
    }

    try:
        # Step 1: Resolve video_id and info_data
        info_data = {}
        if video_id:
            info_path = os.path.join(TEXT_DIR, video_id, 'info.json')
            info_data = safe_read_from_json(info_path) or {}
        elif video_path:
            video_id = video_path_mgr.get_video_id(video_path)
            if not video_id:
                video_id = generate_video_id(video_path)
                video_dir = os.path.join(VIDEO_DIR, video_id)
                new_video_path = os.path.join(video_dir, os.path.basename(video_path))
                if video_path != new_video_path and os.path.isfile(video_path):
                    os.makedirs(video_dir, exist_ok=True)
                    shutil.move(video_path, new_video_path)
                    video_path = new_video_path
                video_path_mgr.add_path(video_id, video_path)
            info_path = os.path.join(TEXT_DIR, video_id, 'info.json')
            info_data = safe_read_from_json(info_path) or {}

        if not video_id:
            video_data["error"] = "Either video_path or video_id must be provided"
            return video_data

        # Initialize info_data if empty
        if not info_data:
            info_data = {
                "video_id": video_id,
                "video_path": video_path or video_path_mgr.get_video_path(video_id),
                "info_path": os.path.join(TEXT_DIR, video_id, "info.json"),
                "info_directory": os.path.join(TEXT_DIR, video_id),
                "publication_date": datetime.now().strftime("%Y-%m-%d")
            }
            os.makedirs(info_data["info_directory"], exist_ok=True)

        # Step 2: Populate video data with existing or default values
        video_data["id"] = video_id
        video_data["video_url"] = info_data.get("video_url", get_link(info_data.get("video_path", "")) or "")
        video_data["contentUrl"] = video_data["video_url"]
        video_data["optimized_video_url"] = video_data["video_url"]
        video_data["publication_date"] = info_data.get("publication_date", datetime.now().strftime("%Y-%m-%d"))
        video_data["audio_text"] = info_data.get("audio_text", [])
        video_data["transcript"] = info_data.get("transcript", "")
        video_data["description"] = info_data.get("seo_description", "Check out this video")
        video_data["title"] = info_data.get("seo_title", "Untitled Video")
        video_data["keywords_str"] = info_data.get("keywords_str", "")
        video_data["schema_markup"] = info_data.get("schema_markup", {})
        video_data["social_meta"] = info_data.get("social_meta", {})
        video_data["file_metadata"] = info_data.get("file_metadata", {})
        video_data["thumbnail_metadata"] = info_data.get("thumbnail_metadata", {})
        video_data["thumbnail_url"] = info_data.get("thumbnail_url", "/var/www/typicallyoutliers/frontend/public/imgs/no_image.jpg")
        video_data["captions"] = info_data.get("captions", None)

        # Set extension and mime type
        if video_data["video_url"]:
            _, ext = os.path.splitext(os.path.basename(video_data["video_url"]))
            video_data["ext"] = ext
            video_data["mime_type"] = "video/mp4" if ext == ".mp4" else "video/webm" if ext == ".webm" else "video/unknown"

        # Step 3: Save updated info_data if changed
        if any(
            info_data.get(key) != video_data.get(key)
            for key in ["video_url", "audio_text", "transcript", "thumbnail_url", "file_metadata", "thumbnail_metadata", "schema_markup", "social_meta"]
        ):
            safe_dump_to_file(info_data, info_data["info_path"])

    except Exception as e:
        logger.error(f"Error processing video {video_id or video_path}: {str(e)}")
        video_data["error"] = str(e)

    return video_data
video_path = '/home/computron/mnt/webserver/typicallyoutliers/backend/typicallyoutliers_flask/functions/videos/Charlotte Gerson Interviewed on Cancer & Medical Industry Fraud 1_5/Charlotte Gerson Interviewed on Cancer & Medical Industry Fraud 1_5.mp4'

data = get_complete_video_data(video_path)
input(data)
