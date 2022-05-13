
import os
import numpy as np
import open3d as o3d
import imageio

from PIL import Image
from pathlib import Path
from dataclasses import dataclass

import scipy

@dataclass
class RGBDnT:
    data : np.ndarray # channels : X Y Z R G B & T
    fid : str = ''

    @property
    def depth_image(self):
        return self.data[:,:,2]
    
    @property
    def visible_image(self):
        return self.data[:,:,3:-1]
    
    @property
    def thermal_image(self):
        return self.data[:,:,-1]
    
    @property
    def point_cloud(self):
        height, width, channel = self.data.shape
        tmp = self.data.reshape((height * width), channel)
        vertex_list = [tuple(x.tolist()) for x in tmp]
        return np.array(vertex_list, 
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), # position
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), # color
                ('thermal', 'u1') # thermal
            ])
    
    def visible_point_cloud(self):
        height, width, channel = self.data.shape
        tmp = self.data.reshape((height * width), channel)
        vertex_list = [tuple(x[:-1].tolist()) for x in tmp]
        return np.array(vertex_list, 
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), # position
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
    
    def thermal_point_cloud(self, remove_nulls : bool = False):
        height, width, channel = self.data.shape
        tmp = self.data.reshape((height * width), channel)
        vertex_list = [tuple(x[(0,1,2,-1)].tolist()) for x in tmp if x[-1] <= 0]
        return np.array(vertex_list, 
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), # position
                ('thermal', 'u1') # thermal
            ])

    def _convert_img_o3d(self, img : np.ndarray, fname : str):
        temp_dir = os.path.join(os.getcwd(),'tmp')
        if not os.path.isdir(temp_dir):
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
        temp_file = os.path.join(temp_dir, fname)
        imageio.imwrite(temp_file, img)
        o3d_img = o3d.io.read_image(temp_file)
        os.unlink(temp_file)
        return o3d_img

    def to_visible_image_o3d(self):
        return self._convert_img_o3d(self.visible_image, '.visible.png')
    
    def to_thermal_image_o3d(self):
        tmp = np.stack((self.thermal_image, self.thermal_image, self.thermal_image), axis=2)
        return self._convert_img_o3d(tmp, '.thermal.png')
    
    def to_depth_image_o3d(self):
        return self._convert_img_o3d(self.depth_image, '.depth.png')
    
    def _convert_RGBD_o3d(self, 
        color : o3d.geometry.Image,
        depth : o3d.geometry.Image,
        depth_scale : float, 
        depth_trunc : float
    ):
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=depth_scale, depth_trunc=depth_trunc)

    def to_RGBD_visible_o3d(self, 
        depth_scale : float = 1000.0, 
        depth_trunc : float = 5.0):
        return self._convert_RGBD_o3d(
            self.to_visible_image_o3d(),
            self.to_depth_image_o3d(),
            depth_scale=depth_scale, 
            depth_trunc=depth_trunc)
    
    def to_RGBD_thermal_o3d(self, 
        depth_scale : float = 1000.0, 
        depth_trunc : float = 5.0):
        return self._convert_RGBD_o3d(
            self.to_thermal_image_o3d(),
            self.to_depth_image_o3d(),
            depth_scale=depth_scale, 
            depth_trunc=depth_trunc)
    
    def _convert_point_cloud_o3d(self,
        rgbd : o3d.geometry.RGBDImage,
        intrinsic : o3d.camera.PinholeCameraIntrinsic):
        return o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic=intrinsic)

    def to_point_cloud_visible_o3d(self,
        intrinsic : o3d.camera.PinholeCameraIntrinsic):
        return self._convert_point_cloud_o3d(self.to_RGBD_visible_o3d(), intrinsic)
    
    def to_point_cloud_thermal_o3d(self,
        intrinsic : o3d.camera.PinholeCameraIntrinsic):
        return self._convert_point_cloud_o3d(self.to_RGBD_thermal_o3d(), intrinsic)
