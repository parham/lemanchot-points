
import os
import numpy as np

from plyfile import PlyData, PlyElement
from typing import List, Tuple, Union
from scipy.io import savemat, loadmat
from phm.data import RGBDnT
from phm.vtd import rgbdt_to_array3d

__rgbdt__ = 'rgbdt'
__fid__ = 'fid'

def save_RGBDnT(file : str, record : RGBDnT):
    if record.data is None:
        raise ValueError('RGBD&T data is missing or corrupted!')
    
    savemat(file, {
        __rgbdt__ : record.data,
        __fid__ : record.fid
    }, do_compression=True)

def load_RGBDnT(file : str) -> RGBDnT:
    if not os.path.isfile(file):
        raise FileNotFoundError(f'{file} not found.')
    
    obj = loadmat(file)
    if not __rgbdt__ in obj:
        raise ValueError('RGBD&T file is not valid!')

    rgbdt = obj[__rgbdt__]

    fid = obj[__fid__] if __fid__ in obj else 'unknown'

    return RGBDnT(fid=fid, data=rgbdt)

__pcloud_exporters = {}

def point_cloud_exporter(name : Union[str, List[str]]):
    def __embed_func(func):
        global __pcloud_exporters
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __pcloud_exporters[n] = func
    return __embed_func

def supported_point_cloud_exporters() -> Tuple:
    return tuple(__pcloud_exporters.keys())

def save_point_cloud(file : str, data : RGBDnT, file_type : str):
    if not file_type in supported_point_cloud_exporters():
        raise ValueError(f'{file_type} exporter does not exist!')
    return __pcloud_exporters[file_type](file, data, file_type)

@point_cloud_exporter(['ply_txt', 'ply_bin'])
def write_ply(file : str, data : RGBDnT, file_type : str):
    vertex = data.point_cloud
    is_text = True if file_type == 'ply_txt' else False
    # Create Vertex ([x,y,z], r, g, b)
    pdata = PlyData(
        [
            PlyElement.describe(vertex, 'vertex', comments=['points (x,y,z, r,g,b, thermal)'])
        ], text=is_text, byte_order='=', comments=['Multi-modal Point Cloud (position : x y z, color : RGB, Thermal : single value']
    ).write(file)

__pcloud_loaders = {}

def point_cloud_loader(name : Union[str, List[str]]):
    def __embed_func(func):
        global __pcloud_loaders
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __pcloud_loaders[n] = func
    return __embed_func

def supported_point_cloud_loaders() -> Tuple:
    return tuple(__pcloud_loaders.keys())

def load_point_cloud(file : str, file_type : str):
    if not file_type in supported_point_cloud_loaders():
        raise ValueError(f'{file_type} loader does not exist!')
    return __pcloud_loaders[file_type](file, file_type)

@point_cloud_loader(['ply_txt', 'ply_bin'])
def load_ply(file : str, file_type : str):
    pcs = PlyData.read(file)
    points = [np.array(x) for x in pcs['vertex']]
    return points
