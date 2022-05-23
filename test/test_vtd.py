
import unittest
import sys,os
import open3d as o3d

from PIL import Image

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phm.utils import blend_vt, show_modalities_grid
from phm.io import load_mme, load_point_cloud, save_point_cloud
from phm.vtd import VTD_Alignment

class Test_VTD(unittest.TestCase):

    def test_compute_o3d(self):
        data = load_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/mat/mme_1625604430816.mat', 'mat')
        vtd = VTD_Alignment(
            target_dir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/vtd',
            depth_param_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/camera_info.json'
        )
        vtd.load()
        rgbdt = vtd.compute(data)
        pt_rgb = rgbdt.to_point_cloud_visible_o3d(vtd.pinhole_camera)
        pt_thermal = rgbdt.to_point_cloud_thermal_o3d(vtd.pinhole_camera)
        
        viz_obj = o3d.visualization.VisualizerWithVertexSelection()
        viz_obj.create_window('Test')
        viz_obj.add_geometry(pt_rgb)
        viz_obj.run()
        viz_obj.destroy_window()
        # o3d.visualization.draw_geometries([pt_rgb, pt_thermal])
        # o3d.visualization.draw_geometries([rgbdt.to_point_cloud_visible_o3d(vtd.pinhole_camera)])

    # def test_compute_np(self):
    #     data = load_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/mat/mme_1625604430816.mat', 'mat')
    #     vtd = VTD_Alignment(
    #         target_dir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/vtd',
    #         depth_param_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/camera_info.json'
    #     )
    #     vtd.load()
    #     rgbdt = vtd.compute(data)
    #     save_point_cloud('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/test.ply', rgbdt, 'ply_txt')
    #     load_point_cloud('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/test.ply', file_type='ply_txt')

if __name__ == '__main__':
    unittest.main()