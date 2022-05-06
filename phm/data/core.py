
from enum import Enum, auto
from functools import lru_cache
import os
from typing import List, Union

import numpy as np

def modal_to_image(img : np.ndarray):
    return ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.0

class MMEntityType(Enum):
    Visible = auto()
    Thermal = auto()
    DepthMap = auto()
    RGBDT = auto()
    PointCloud = auto()

    @staticmethod
    def by_name(name : str):
        return {
            'visible' : MMEntityType.Visible,
            'thermal' : MMEntityType.Thermal,
            'depthmap' : MMEntityType.DepthMap,
            'pointcloud' : MMEntityType.PointCloud,
            'rgbdt' : MMEntityType.RGBDT
        }[name]
    
    def __str__(self) -> str:
        return {
            MMEntityType.Visible : 'visible',
            MMEntityType.Thermal : 'thermal',
            MMEntityType.DepthMap : 'depthmap',
            MMEntityType.PointCloud : 'pointcloud',
            MMEntityType.RGBDT : 'rgbdt'
        }[self]

__mme_loaders = {}

def mme_loader(name : Union[MMEntityType, List[MMEntityType]]):
    def __embed_func(func):
        global __mme_loaders
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __mme_loaders[n] = func
    return __embed_func

def list_mme_loaders() -> List[MMEntityType]:
    global __mme_loaders
    return list(__mme_loaders.keys())

@lru_cache(maxsize=4)
def load_entity(type : MMEntityType, file : str):
    if not os.path.isfile(file):
        raise ValueError(f'{file} is invalid!')

    if not type in __mme_loaders.keys():
        raise ValueError(f'{str(type)} loader does not exist!')

    return __mme_loaders[type](file)
