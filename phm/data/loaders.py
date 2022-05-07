
import numpy as np

from PIL import Image
from scipy.io import loadmat
from phm.data import mme_loader

@mme_loader(['visible', 'thermal', 'depth'])
def image_loader(file : str, file_type : str):
    # Load the visible image
    img = Image.open(file)
    if img is None:
        raise ValueError(f'Loading the {file_type} modality is failed!')
    return np.asarray(img)

@mme_loader('rgbdt')
def rgbdt_loader(file : str, file_type : str):
    rgbdt = loadmat(file)
    if not 'data' in rgbdt:
        raise ValueError(f'{file} is not correctly formated. data field is missing!')
    return rgbdt['data']

@mme_loader('pointcloud')
def pc_loader(file : str, file_type : str):
    raise NotImplementedError()