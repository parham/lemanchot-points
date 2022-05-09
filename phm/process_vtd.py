
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import open3d as o3d

from PIL import Image
from scipy.io import savemat, loadmat

from phm.control_point import cpselect
from phm.data import MMEContainer, gray_to_rgb
from phm.data.data import modal_to_image
from phm.data.modality_io import rgbdt_loader

@dataclass
class RGBDnT:
    visible : np.ndarray
    thermal : np.ndarray
    depth : np.ndarray
    homography : np.ndarray
    # O3D entities
    o3d_visible : o3d.geometry.Image = None
    o3d_thermal : o3d.geometry.Image = None
    o3d_depth : o3d.geometry.Image = None
    rgbd_visible : o3d.geometry.RGBDImage = None
    rgbd_thermal : o3d.geometry.RGBDImage = None
    pinhole_depth_params : o3d.camera.PinholeCameraIntrinsic = None
    pcs_visible : o3d.geometry.PointCloud = None
    pcs_thermal : o3d.geometry.PointCloud = None

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

def blend_vt(data : RGBDnT, alpha : float = 0.6):
    thermal_rgb = gray_to_rgb(data.thermal)
    fused = (data.visible.copy()).astype(np.float)
    ttemp = thermal_rgb * alpha
    vtemp = data.visible * (1 - alpha)
    fused[thermal_rgb > 0] = vtemp[thermal_rgb > 0] + ttemp[thermal_rgb > 0]
    return modal_to_image(fused)

class VTD_Alignment:
    __thermal__ = 'thermal'
    __visible__ = 'visible'
    __depth__ = 'depth'
    __homography__ = 'homography'

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
        if self.homography_file is None or not os.path.isfile(self.homography_file):
            return False
        if self.depth_param_file is None or not os.path.isfile(self.depth_param_file):
            return False
        # Load Homography
        d = loadmat(self.homography_file)
        if not self.__homography__ in d:
            raise ValueError('Homography file is not valid!')
        self._homography = d[self.__homography__]
        # Load Depth Camera Parameters
        dconfig = None
        with open(self.depth_param_file) as fdc:
            dconfig = json.load(fdc)
        self._depth_params = dconfig
        return True

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

    def packing_rgbdt(self, data : MMEContainer):
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
        
        return RGBDnT(visible, corrected_thermal, depth, self._homography)

    def __generate_o3d(self, data : RGBDnT):
        visible = data.visible
        thermal = data.thermal
        depth = data.depth
        ######## Save image temporarily
        temp_dir = os.path.join(os.getcwd(),'tmp')
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        # Save the modalities
        vis_file = os.path.join(temp_dir, '.visible.png')
        th_file = os.path.join(temp_dir, '.thermal.png')
        dp_file = os.path.join(temp_dir, '.depth.png')
        Image.fromarray(visible).save(vis_file)
        Image.fromarray(gray_to_rgb(thermal.astype(np.uint8))).save(th_file)
        Image.fromarray(depth).save(dp_file)
        
        data.o3d_visible = o3d.io.read_image(vis_file)
        data.o3d_thermal = o3d.io.read_image(th_file)
        data.o3d_depth = o3d.io.read_image(dp_file)
 
        data.rgbd_visible = o3d.geometry.RGBDImage.create_from_color_and_depth(
            data.o3d_visible, data.o3d_depth,
            depth_scale=1000.0, depth_trunc=5.0)
        data.rgbd_thermal = o3d.geometry.RGBDImage.create_from_color_and_depth(
            data.o3d_thermal, data.o3d_depth,
            depth_scale=1000.0, depth_trunc=5.0)
        
        # Form the Pinhole Camera parameters
        f_x = self._depth_params['K'][0]
        p_x = self._depth_params['K'][2]
        f_y = self._depth_params['K'][4]
        p_y = self._depth_params['K'][5]
        data.pinhole_depth_params = o3d.camera.PinholeCameraIntrinsic(
            width=visible.shape[1], 
            height=visible.shape[0],
            fx = f_x, fy = f_y,
            cx = p_x, cy = p_y
        )

        #     [ fx   0   cx ] 
        # K = [ 0    fy  cy ] 
        #     [ 0    0   1  ] 
        # 
        # P = d * [(x - p_x) / f_x , (y - p_y) / f_y, 1]^-1
        data.pcs_visible = o3d.geometry.PointCloud.create_from_rgbd_image(
            data.rgbd_visible, 
            intrinsic=data.pinhole_depth_params
        )
        data.pcs_thermal = o3d.geometry.PointCloud.create_from_rgbd_image(
            data.rgbd_thermal, 
            intrinsic=data.pinhole_depth_params
        )

        return data

    def compute(self, data : MMEContainer):
        # Create the RGBD&T data
        rgbdt = self.packing_rgbdt  (data)
        # Create the o3d objects
        rgbdt = self.__generate_o3d(rgbdt)
        return rgbdt
