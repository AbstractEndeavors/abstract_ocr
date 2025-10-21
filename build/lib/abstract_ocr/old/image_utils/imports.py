from PIL import Image
from abstract_utilities import get_all_file_types
import os
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
