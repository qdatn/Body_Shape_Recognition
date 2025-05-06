import numpy as np
import requests
from PIL import Image
from io import BytesIO
import uuid

def calc_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

def download_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        # temp_path = f'data/test/temp_{uuid.uuid4().hex}.jpg'
        temp_path = f'/tmp/assets/temp_{uuid.uuid4().hex}.jpg'
        img.save(temp_path)
        return temp_path
    except Exception as e:
        raise ValueError(f"Lỗi khi tải ảnh: {e}")