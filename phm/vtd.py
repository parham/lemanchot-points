
import os
import cv2
import json
import numpy as np
import open3d as o3d

from typing import List
from scipy.io import savemat, loadmat

from phm.utils import gray_to_rgb, modal_to_image
from phm.control_point import cpselect
from phm.data import MMEContainer, RGBDnT

__homography__ = 'homography'

def rgbdt_to_array3d(data : np.ndarray):
    height, width, channel = data.shape
    data = data.reshape((height * width), channel)
    vertex_list = [tuple(x.tolist()) for x in data]
    return np.array(vertex_list, 
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), # position
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), # color
            ('thermal', 'u1') # thermal
        ])

def load_depth_camera_params(file : str):
    if file is None or not os.path.isfile(file):
        raise FileNotFoundError(f'{file} does not exist!')
    dconfig = None
    with open(file) as fdc:
        dconfig = json.load(fdc)
    return dconfig

def load_pinhole(file : str):
    dparam = load_depth_camera_params(file)
    f_x = dparam['K'][0]
    p_x = dparam['K'][2]
    f_y = dparam['K'][4]
    p_y = dparam['K'][5]
    return o3d.camera.PinholeCameraIntrinsic(
        width=dparam['width'], 
        height=dparam['height'],
        fx = f_x, fy = f_y,
        cx = p_x, cy = p_y
    )

def load_homography(file : str, silent : bool = False):
    if file is None or not os.path.isfile(file):
        if not silent:
            raise FileNotFoundError(f'{file} does not exist!')
        else:
            return
    # Load Homography
    d = loadmat(file)
    if not __homography__ in d and d[__homography__] is not None:
        raise ValueError('Homography file is not valid!')
    return d[__homography__]

def save_homography(file : str, homography : np.ndarray):
    mat = {
        __homography__ : homography
    }
    savemat(file, mat, do_compression=True)

class VTD_Alignment:
    __thermal__ = 'thermal'
    __visible__ = 'visible'
    __depth__ = 'depth'

    def __init__(self, 
        target_dir : str = None,
        depth_param_file : str = None
    ) -> None:
        self._homography = None
        self._depth_params = None
        self.target_dir = target_dir if target_dir is not None else os.getcwd()
        self.homography_file = os.path.join(self.target_dir, 'homography.mat')
        self.depth_param_file = depth_param_file 

    @property
    def homography(self):
        return self._homography

    @property
    def depth_camera_params(self):
        return self._depth_params

    @property
    def pinhole_camera(self) -> o3d.camera.PinholeCameraIntrinsic:
        dparam = self.depth_camera_params
        f_x = dparam['K'][0]
        p_x = dparam['K'][2]
        f_y = dparam['K'][4]
        p_y = dparam['K'][5]
        return o3d.camera.PinholeCameraIntrinsic(
            width=dparam['width'], 
            height=dparam['height'],
            fx = f_x, fy = f_y,
            cx = p_x, cy = p_y
        )

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
        # Save homography
        save_homography(self.homography_file, self.homography)

    def load(self) -> bool:
        # Load Homography
        self._homography = load_homography(self.homography_file, silent=True)
        # Load Depth Camera Parameters
        self._depth_params = load_depth_camera_params(self.depth_param_file)

    def __init(self, data : MMEContainer):
        if self._homography is not None:
            return
        # STEP 01 : Run control point selector toolbox
        cps = cpselect(data['thermal'].data, data['visible'].data)
        source, dest = self.cp_to_opencv(cps)
        # STEP 02 : estimate homography transformation
        h, _ = cv2.findHomography(source, dest)
        self._homography = h
        self.save()

    def __align(self, data : MMEContainer) -> RGBDnT:
        if not self.__thermal__ in data.modality_names or \
           not self.__visible__ in data.modality_names or \
           not self.__depth__ in data.modality_names:
            raise ValueError(f'Data container does not have {self._src_type} or {self._dsc_type}')
        # Check homography availability
        self.__init(data)
        # Corrent the thermal image
        thermal = modal_to_image(data[self.__thermal__].data)
        visible = data[self.__visible__].data
        depth = data[self.__depth__].data
        corrected_thermal = cv2.warpPerspective(thermal, self.homography, 
            (visible.shape[1], visible.shape[0]))
        
        return visible, corrected_thermal, depth

    def compute(self, data : MMEContainer) -> RGBDnT:
        visible, thermal, depth = self.__align(data)
        return self._fuse(visible, thermal, depth)

    def _fuse(self, visible, thermal, depth):
        # Form the point cloud
        # P = d * [(x - p_x) / f_x , (y - p_y) / f_y, 1]^-1
        height, width = depth.shape

        X = np.linspace(0, width-1, width)
        Y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(X, Y)
        # Form the Pinhole Camera parameters
        f_x = self._depth_params['K'][0]
        p_x = self._depth_params['K'][2]
        f_y = self._depth_params['K'][4]
        p_y = self._depth_params['K'][5]
        # Correct the coordinates
        Z = depth / 1000
        X = np.multiply(Z, (X - p_x) / f_x)
        Y = np.multiply(Z, (Y - p_y) / f_y)
        
        data =  np.dstack((X,Y,Z, gray_to_rgb(visible), thermal))
        return RGBDnT(data)
