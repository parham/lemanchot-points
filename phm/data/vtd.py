
import numpy as np
import open3d as o3d

from dataclasses import dataclass

@dataclass
class RGBDnT:
    visible : np.ndarray
    thermal : np.ndarray
    depth : np.ndarray
    homography : np.ndarray
    rgbdt : np.ndarray = None # channels : X Y Z R G B & T

    def list_modalities(self):
        return (self.visible, self.thermal, self.depth)

@dataclass
class RGBDnT_O3D(RGBDnT):
    # O3D entities
    o3d_visible : o3d.geometry.Image = None
    o3d_thermal : o3d.geometry.Image = None
    o3d_depth : o3d.geometry.Image = None
    rgbd_visible : o3d.geometry.RGBDImage = None
    rgbd_thermal : o3d.geometry.RGBDImage = None
    pinhole_depth_params : o3d.camera.PinholeCameraIntrinsic = None
    pcs_visible : o3d.geometry.PointCloud = None
    pcs_thermal : o3d.geometry.PointCloud = None

    def list_o3d_modalities(self):
        return (self.o3d_visible, self.o3d_thermal, self.o3d_depth)

    @staticmethod
    def pack(data : RGBDnT):
        return RGBDnT_O3D(
            data.visible, 
            data.thermal, 
            data.depth, 
            homography=data.homography
        )
