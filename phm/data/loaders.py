
import numpy as np
from PIL import Image
from scipy.io import loadmat

from phm.data.core import MMEntityType, mme_loader

@mme_loader([MMEntityType.Visible, MMEntityType.Thermal, MMEntityType.DepthMap])
def visible_loader(file : str):
    # Load the visible image
    img = Image.open(file)
    if img is None:
        raise ValueError('Loading the modality is failed!')
    return np.asarray(img)

@mme_loader(MMEntityType.RGBDT)
def rgbdt_loader(file : str):
    # Load the RGBDT data
    rgbdt = loadmat(file)
    if not 'data' in rgbdt:
        raise ValueError(f'{file} is not correctly formated. data field is missing!')
    return rgbdt['data']

@mme_loader(MMEntityType.PointCloud)
def pc_loader(file : str):
    raise NotImplementedError()