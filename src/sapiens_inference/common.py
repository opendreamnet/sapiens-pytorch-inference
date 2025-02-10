import os
from typing import List, Optional
import requests
import logging
from tqdm import tqdm
from enum import Enum
from huggingface_hub import hf_hub_url
from huggingface_hub.constants import HF_HUB_CACHE
from torchvision import transforms

logger = logging.getLogger("sapiens_inference.common")

class TaskType(Enum):
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"
    POSE = "pose"

def get_path_to_cache(filename: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or HF_HUB_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)

def download(url: str, filename: str):
    temp_filename = f"{filename}.part"
    with open(temp_filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
                'desc': filename,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)
    os.rename(temp_filename, filename)

def download_hf_model(model_name: str, cache_dir: Optional[str] = None) -> str:
    repo_name = model_name.split("/")[0]
    filename = model_name.split("/")[1]

    path = get_path_to_cache(filename, cache_dir)
    if os.path.exists(path):
        return path

    logger.info(f"Model {filename} not found, downloading from HuggingFace Hub...")

    repo_id = f"facebook/{repo_name}"
    url = hf_hub_url(repo_id=repo_id, filename=filename)

    download(url, path)
    logger.info(f"Model downloaded successfully to {path}")

    return path

def create_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               ])

def pose_estimation_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):

    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               ])
