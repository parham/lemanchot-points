
import os
import numpy as np

from typing import List, Tuple, Union
from scipy.io import savemat
from phm.data import RGBDnT

def save_RGBDnT(file : str, data : RGBDnT):
    if not os.path.isfile(file):
        raise FileNotFoundError(f'{file} not found.')
    if data.rgbdt is None:
        raise ValueError('RGBDT data is missing or corrupted!')
    savemat(file, {'rgbdt' : data}, do_compression=True)

__pcloud_exporters = {}

def point_cloud_exporter(name : Union[str, List[str]]):
    def __embed_func(func):
        global __pcloud_exporters
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __pcloud_exporters[n] = func
    return __embed_func

def supported_mme_exporters() -> Tuple:
    return tuple(__pcloud_exporters.keys())

def save_point_cloud(file : str, data : RGBDnT, file_type : str):
    if not file_type in supported_mme_exporters():
        raise ValueError(f'{file_type} loader does not exist!')
    return __pcloud_exporters[file_type](file, data, file_type)

def rgbdt_to_array3d(data : np.ndarray):
    height, width, channel = data.shape
    data = data.reshape((height * width), channel)
    return data

@point_cloud_exporter('ply_bin')
def write_ply(file : str, data : RGBDnT, file_type : str):
    

@point_cloud_exporter('ply_text')
def write_ply(file : str, data : RGBDnT, file_type : str):
    pass
