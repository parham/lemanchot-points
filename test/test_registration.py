
import os
import sys
import unittest

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui


sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phm.data.vtd import DualPointCloudPack, O3DPointCloudWrapper
from phm.pipeline.core import ConvertToPC_Step, FilterDepthRange_Step, Pipeline, PipelineStep, RGBDnTBatch
from phm.pipeline.icp_pipeline import ColoredICPRegistar_Step
from phm.pipeline.manual_pipeline import ManualRegistration_Step
from phm.visualization import visualize_vtd
from phm.vtd import load_pinhole

class Test_Colored_ICP(unittest.TestCase):
    def test_colored_icp(self):
        # colored pointcloud registration
        # This is implementation of following paper
        # J. Park, Q.-Y. Zhou, V. Koltun,
        # Colored Point Cloud Registration Revisited, ICCV 2017

        batch = RGBDnTBatch(
            root_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd',
            filenames= [
                'vtd_1626967963384.mat', 'vtd_1626967965865.mat', 'vtd_1626967973439.mat' # 
            ]
        )

        pipobj = Pipeline([
            FilterDepthRange_Step(),
            ConvertToPC_Step(
                depth_params_file = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json',
                data_batch_key = 'prp_frames'),
            ManualRegistration_Step(
                depth_params=load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'),
                data_pcs_key='pcs'),
        ])

        res = pipobj(batch)
        res_pc = res['fused_pc']

        visualize_vtd(
            res_pc,
            load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'), 
            'PHM Visualization', 1024, 768
        )

    # def test_colored_icp(self):
    #     # colored pointcloud registration
    #     # This is implementation of following paper
    #     # J. Park, Q.-Y. Zhou, V. Koltun,
    #     # Colored Point Cloud Registration Revisited, ICCV 2017

    #     batch = RGBDnTBatch(
    #         root_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd',
    #         filenames= [
    #             'vtd_1626967963384.mat', 'vtd_1626967965865.mat', 'vtd_1626967973439.mat'
    #         ]
    #     )

    #     pipobj = Pipeline([
    #         FilterDepthRange_Step(),
    #         ConvertToPC_Step(
    #             data_batch_key = 'prp_frames',
    #             depth_params_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'),
    #         ColoredICPRegistar_Step(data_pcs_key='pcs'),
    #     ])

    #     res = pipobj(batch)
    #     res_pc = res['fused_pc']

    #     visualize_vtd(
    #         res_pc,
    #         load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'), 
    #         'PHM Visualization', 1024, 768
    #     )

if __name__ == '__main__':
    unittest.main()