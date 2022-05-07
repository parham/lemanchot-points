
import unittest
import sys,os

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from PIL import Image

from phm.data import load_mme
from phm.process_vtd import VTD_Alignment, blend_vt, show_rgbdt

class Test_VTD(unittest.TestCase):
    def test_compute_rgbdt(self):
        data = load_mme('/home/phm/Datasets/multi-modal/20210706_multi_modal/mat/mme_1625604430816.mat', 'mat')
        vtd = VTD_Alignment(target_dir = '/home/phm/Datasets/multi-modal/20210706_multi_modal')
        vtd.load()

        rgbdt = vtd.compute_rgbdt(data)
        show_rgbdt(rgbdt)

        # Blend
        fused = blend_vt(rgbdt)
        fused_img = Image.fromarray(fused)
        fused_img.show()
        fused_img.save('/home/phm/Datasets/multi-modal/20210706_multi_modal/fused.png')

if __name__ == '__main__':
    unittest.main()