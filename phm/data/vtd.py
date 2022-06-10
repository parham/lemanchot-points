
import copy
from multiprocessing.sharedctypes import Value
import os
import numpy as np
import open3d as o3d
import imageio

from pathlib import Path
from dataclasses import dataclass

__depth_scale__ = 1000

def filter_out_zero_thermal(pct):
    points = np.asarray(pct.points) 
    temps = np.asarray(pct.colors)
    temps_v = np.mean(temps, axis=1)

    pct.points = o3d.utility.Vector3dVector(points[temps_v > 0, :])
    pct.colors = o3d.utility.Vector3dVector(temps[temps_v > 0, :])
    return pct

class O3DPointCloudWrapper:
    def get_thermal_point_cloud(self, **kwargs):
        raise NotImplementedError('get_thermal_point_cloud is not implemented!')
    
    def get_visible_point_cloud(self, **kwargs):
        raise NotImplementedError('get_visible_point_cloud is not implemented')
    
    def get_fused_point_cloud(self, **kwargs):
        pcv = self.get_visible_point_cloud(**kwargs)
        pct = self.get_thermal_point_cloud(**kwargs)

        return self._fuse_point_cloud(pcv, pct)

    def _fuse_point_cloud(self, pcv, pct):
        pcv_copy = copy.deepcopy(pcv)
        temps = np.asarray(pct.colors)
        colors = np.asarray(pcv_copy.colors)
        temps_v = np.mean(temps, axis=1)

        colors[temps_v > 0, :] = temps[temps_v > 0, :]
        pcv_copy.colors = o3d.utility.Vector3dVector(colors)
        return pcv_copy

    def __getitem__(self, index):
        pc = None
        if index == 0:
            pc = self.get_visible_point_cloud()
        elif index == 1:
            pc = self.get_thermal_point_cloud(remove_invalids=True)
        else:
            raise IndexError()
        return pc
    
    def __len__(self):
        return 2

@dataclass
class RGBDnT(O3DPointCloudWrapper):

    data : np.ndarray # channels : X Y Z R G B & T
    fid : str = ''

    @property
    def depth_image(self):
        return np.asarray(self.data[:,:,2] * __depth_scale__, np.uint16)

    @depth_image.setter
    def depth_image(self, depth):
        if not isinstance(depth, np.ndarray):
            raise ValueError('The depth image should be numpy matrix!')
        dsize = depth.shape
        tsize = self.data.shape
        # Check image dimensions
        if dsize[0] != tsize[0] or dsize[1] != tsize[1]:
            raise ValueError('The input depth image does not follow the RGBD&T dimension!')
        # Check image channels
        if len(dsize) > 2 and dsize[2] > 1:
            raise ValueError('Depth Image cannot have multiple channels!')
        # Convert the image to proper format
        tmp = (depth / __depth_scale__).astype(self.data.dtype)
        self.data[:,:,2] = tmp

    @property
    def visible_image(self):
        return np.asarray(self.data[:,:,3:-1], np.uint8)
    
    @property
    def thermal_image(self):
        return np.asarray(self.data[:,:,-1], np.uint8)
    
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
    
    def thermal_point_cloud(self):
        height, width, channel = self.data.shape
        tmp = self.data.reshape((height * width), channel)
        vertex_list = [tuple(map(x.__getitem__, (0,1,2,-1))) for x in tmp if x[-1] <= 0]
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
        depth_scale : float = __depth_scale__, 
        depth_trunc : float = 5.0):
        return self._convert_RGBD_o3d(
            self.to_visible_image_o3d(),
            self.to_depth_image_o3d(),
            depth_scale=depth_scale, 
            depth_trunc=depth_trunc)
    
    def to_RGBD_thermal_o3d(self, 
        depth_scale : float = __depth_scale__, 
        depth_trunc : float = 5.0):
        return self._convert_RGBD_o3d(
            self.to_thermal_image_o3d(),
            self.to_depth_image_o3d(),
            depth_scale=depth_scale, 
            depth_trunc=depth_trunc)
    
    def _convert_point_cloud_o3d(self,
        rgbd : o3d.geometry.RGBDImage,
        intrinsic : o3d.camera.PinholeCameraIntrinsic,
        calc_normals : bool = True):
        ps = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic=intrinsic)
        if calc_normals :
            if not ps.has_normals():
                ps.estimate_normals()
            ps.normalize_normals()
        return ps

    def to_point_cloud_visible_o3d(self,
        intrinsic : o3d.camera.PinholeCameraIntrinsic,
        calc_normals : bool = False):
        return self._convert_point_cloud_o3d(self.to_RGBD_visible_o3d(), intrinsic, calc_normals)
    
    def to_point_cloud_thermal_o3d(self,
        intrinsic : o3d.camera.PinholeCameraIntrinsic,
        calc_normals : bool = False,
        remove_invalids : bool = False):

        pct = self._convert_point_cloud_o3d(self.to_RGBD_thermal_o3d(), intrinsic, calc_normals)
        if remove_invalids:
            pct = filter_out_zero_thermal(pct)
        return pct

    def to_point_cloud_fusion_o3d(self,
        intrinsic : o3d.camera.PinholeCameraIntrinsic,
        calc_normals : bool = False):

        pcv = self.__data.to_point_cloud_visible_o3d(intrinsic, calc_normals=calc_normals)
        pct = self.__data.to_point_cloud_thermal_o3d(intrinsic, calc_normals=calc_normals)

        return self._fuse_point_cloud(pcv, pct)

    def get_visible_point_cloud(self, **kwargs):
        return self.to_point_cloud_visible_o3d(
            intrinsic = kwargs['intrinsic'], 
            calc_normals = kwargs['calc_normals'] if 'calc_normals' in kwargs else None
        )

    def get_thermal_point_cloud(self, **kwargs):
        return self.to_point_cloud_thermal_o3d(
            intrinsic = kwargs['intrinsic'], 
            calc_normals = kwargs['calc_normals'] if 'calc_normals' in kwargs else None,
            remove_invalids = kwargs['remove_invalids'] if 'remove_invalids' in kwargs else None
        )

@dataclass
class DualPointCloudPack(O3DPointCloudWrapper):

    visible_pointcloud : o3d.geometry.PointCloud
    thermal_pointcloud : o3d.geometry.PointCloud

    def get_thermal_point_cloud(self, **kwargs):
        pct = self.thermal_pointcloud 
        if 'remove_invalids' in kwargs and kwargs['remove_invalids']:
            pct = filter_out_zero_thermal(copy.deepcopy(pct))

        return pct
    
    def get_visible_point_cloud(self, **kwargs):
        return self.visible_pointcloud
