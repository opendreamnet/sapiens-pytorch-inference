import requests
from io import BytesIO
import cv2
import numpy as np
import logging
from PIL import Image
from fake_useragent import UserAgent

logger = logging.getLogger("scripts.utils.file")

def read_from_url(url: str, gif_frame = 0) -> Image.Image:
    ua = UserAgent()
    header = {'User-Agent': str(ua.chrome)}
    response = requests.get(url, headers=header)

    logger.debug(f"Request to URL: {url} using agent: {header['User-Agent']}")

    image = Image.open(BytesIO(response.content))

    if 'n_frames' in dir(image):
        if 0 <= gif_frame < image.n_frames:
            image.seek(gif_frame)
        elif gif_frame < 0:
            logger.warning('The index when seeking must be a positive integer')
            image.seek(0)
        else:
            logger.warning('An index outside the seek range was specified')
            logger.warning(f'Total frames: {image.n_frames} - Requested: ${gif_frame}')
            image.seek(image.n_frames - 1)

        image = image.convert('RGB')

    return image

def read_from_url_array(url: str, gif_frame = 0):
    image = read_from_url(url, gif_frame)
    return np.array(image)

def read_from_url_cv2(url: str, gif_frame = 0):
    image = read_from_url_array(url, gif_frame)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
