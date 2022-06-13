
import unittest
import os
import sys
import open3d.visualization.gui as gui

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phm.io import load_RGBDnT
from phm.visualization import VTD_Visualization
from phm.vtd import load_pinhole
from phm.io.vtd import load_dual_point_cloud

class Test_Visualization(unittest.TestCase):
    # def test_win_app_vtd(self):
    #     data = load_RGBDnT('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20220525_concrete_vertical/vtd/vtd_1653490108856.mat')
        
    #     gui.Application.instance.initialize()
    #     w = VTD_Visualization(data, 
    #         load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20220525_concrete_vertical/depth/camera_info.json'), 
    #         'PHM Visualization Test', 1024, 768)

    #     # Run the event loop. This will not return until the last window is closed.
    #     gui.Application.instance.run()

    def test_win_app_pc(self):
        
        data = load_dual_point_cloud(
            visible_pc_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/preprocessing/pc_visible_1.ply',
            thermal_pc_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/preprocessing/pc_thermal_1.ply'
        )

        gui.Application.instance.initialize()
        w = VTD_Visualization(data, 
            load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20220525_concrete_vertical/depth/camera_info.json'), 
            'PHM Visualization Test', 1024, 768)

        # Run the event loop. This will not return until the last window is closed.
        gui.Application.instance.run()

if __name__ == '__main__':
    unittest.main()