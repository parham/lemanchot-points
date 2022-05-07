
from dataclasses import dataclass
import os
from typing import Dict, List
import cv2
import numpy as np

from phm.control_point import cpselect
from PIL import Image
from scipy.io import savemat, loadmat

from phm.data import MMEContainer, gray_to_rgb
from phm.data.data import modal_to_image

@dataclass
class RGBDnT:
    visible : np.ndarray
    thermal : np.ndarray
    depth : np.ndarray
    homography : np.ndarray

    def list_modalities(self):
        return (self.visible, self.thermal, self.depth)

def show_rgbdt(data : RGBDnT):
    import numpy as np
    import matplotlib.pyplot as plt

    w = 10
    h = 10
    fig = plt.figure(figsize=(2, 2))
    
    fig.add_subplot(2, 2, 1)
    plt.imshow(data.visible)
    fig.add_subplot(2, 2, 2)
    plt.imshow(data.thermal)
    fig.add_subplot(2, 2, 3)
    plt.imshow(data.depth)
    
    plt.show()

def blend_vt(data : RGBDnT):
    thermal_rgb = gray_to_rgb(data.thermal)
    fused = data.visible.copy()
    fused[thermal_rgb > 0] = thermal_rgb[thermal_rgb > 0]
    return fused

class VTD_Alignment:
    __thermal__ = 'thermal'
    __visible__ = 'visible'
    __depth__ = 'depth'
    __homography__ = 'homography'

    def __init__(self, target_dir : str = None) -> None:
        self._homography = None
        self.target_dir = target_dir if target_dir is not None else os.getcwd()
        self.homography_file = os.path.join(self.target_dir, 'homography.mat')

    @property
    def homography(self):
        return self._homography

    def cp_to_opencv(self, cps : List):
        source = np.zeros((len(cps), 2))
        dest = np.zeros((len(cps), 2))
        for index in range(len(cps)):
            p = cps[index]
            source[index, 0] = p['img1_x']
            source[index, 1] = p['img1_y']
            dest[index, 0] = p['img2_x']
            dest[index, 1] = p['img2_y']
        return source, dest

    def reset(self):
        self._homography = None

    def save(self):
        mat = {
            self.__homography__ : self.homography
        }
        savemat(self.homography_file, mat, do_compression=True)
    
    def load(self) -> bool:
        if not os.path.isfile(self.homography_file):
            return False
        d = loadmat(self.homography_file)
        if not self.__homography__ in d:
            raise ValueError('Homography file is not valid!')
        self._homography = d[self.__homography__]
        return True

    def _init(self, data : MMEContainer):
        if self._homography is not None:
            return
        # STEP 01 : Run control point selector toolbox
        cps = cpselect(data['thermal'].data, data['visible'].data)
        source, dest = self.cp_to_opencv(cps)
        # STEP 02 : estimate homography transformation
        h, _ = cv2.findHomography(source, dest)
        self._homography = h
        self.save()

    def compute_rgbdt(self, data : MMEContainer):
        if not self.__thermal__ in data.modality_names or \
           not self.__visible__ in data.modality_names or \
           not self.__depth__ in data.modality_names:
            raise ValueError(f'Data container does not have {self._src_type} or {self._dsc_type}')
        # Check homography availability
        self._init(data)
        # Corrent the thermal image
        thermal = modal_to_image(data[self.__thermal__].data)
        visible = data[self.__visible__].data
        depth = data[self.__depth__].data
        corrected_thermal = cv2.warpPerspective(thermal, self.homography, 
            (visible.shape[1], visible.shape[0]))
        
        return RGBDnT(visible, corrected_thermal, depth, self._homography)
