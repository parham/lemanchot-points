
import unittest
import sys,os
import open3d as o3d

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from PIL import Image

from phm.data import load_mme
from phm.process_vtd import VTD_Alignment, VTD_Alignment_O3D, blend_vt, show_modalities_grid

class Test_VTD(unittest.TestCase):
    def test_compute_rgbdt(self):
        data = load_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/mat/mme_1625604430816.mat', 'mat')
        vtd = VTD_Alignment(
            target_dir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/',
            depth_param_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/camera_info.json'
        )
        vtd.load()

        rgbdt = vtd.pack(data)
        show_modalities_grid(rgbdt)

        # Blend
        fused = blend_vt(rgbdt, 0.85)
        fused_img = Image.fromarray(fused)
        fused_img.show()
        fused_img.save('fused.png')

    def test_compute_o3d(self):
        data = load_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/mat/mme_1625604430816.mat', 'mat')
        vtd = VTD_Alignment_O3D(
            target_dir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/',
            depth_param_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/camera_info.json'
        )
        vtd.load()
        rgbdt = vtd.compute(data)
        o3d.visualization.draw_geometries([rgbdt.pcs_visible])

    def test_compute_np(self):
        data = load_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/mat/mme_1625604430816.mat', 'mat')
        vtd = VTD_Alignment(
            target_dir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/',
            depth_param_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/camera_info.json'
        )
        vtd.load()
        rgbdt = vtd.compute(data)
        

if __name__ == '__main__':
    unittest.main()