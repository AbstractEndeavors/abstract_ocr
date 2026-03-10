from .imports import (
    get_file_size,
    get_all_file_types,
    Image
    )
def get_frame_number(file_path):
    if isinstance(file_path,dict):
        file_path = file_path.get('frame')
        
    file_path = '.'.join(file_path.split('.')[:-1])
    return int(file_path.split('_')[-1])
def sort_frames(frames=None,directory=None):
    if frames in [None,[]] and directory and os.path.isdir(directory):
        frames = get_all_file_types(types=['image'],directory=directory)
    frames = frames or []
    
    frames = sorted(
        frames,
        key=lambda x: get_frame_number(x) 
    )
    return frames
    
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
