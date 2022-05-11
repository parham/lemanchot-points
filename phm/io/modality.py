
import os
import numpy as np

from PIL import Image
from scipy.io import loadmat
from functools import lru_cache
from typing import List, Tuple, Union

__modality_loaders = {}

def modality_loader(name : Union[str, List[str]]):
    def __embed_func(func):
        global __modality_loaders
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __modality_loaders[n] = func
    return __embed_func

def supported_modality_loaders() -> Tuple:
    return tuple(__modality_loaders.keys())

@lru_cache(maxsize=4)
def load_entity(file_type : str, file : str):
    if not os.path.isfile(file):
        raise ValueError(f'{file} is invalid!')

    if not file_type in supported_modality_loaders():
        raise ValueError(f'{file_type} loader does not exist!')

    return __modality_loaders[file_type](file, file_type)

@modality_loader(['visible', 'thermal', 'depth'])
def image_loader(file : str, file_type : str):
    # Load the visible image
    img = Image.open(file)
    if img is None:
        raise ValueError(f'Loading the {file_type} modality is failed!')
    return np.asarray(img)

@modality_loader('rgbdt')
def rgbdt_loader(file : str, file_type : str):
    rgbdt = loadmat(file)
    if not 'data' in rgbdt:
        raise ValueError(f'{file} is not correctly formated. data field is missing!')
    return rgbdt['data']

@modality_loader('pointcloud')
def pc_loader(file : str, file_type : str):
    raise NotImplementedError()