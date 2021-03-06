
import os
import sys
import unittest

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui


sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phm.pipeline.probreg_pipeline import CPDRegistration_Step, FilterregRegistration_Step, GMMTreeRegistration_Step, SVRRegistration_Step
from phm.data.vtd import DualPointCloudPack, O3DPointCloudWrapper
from phm.pipeline.core import ConvertToPC_Step, FilterDepthRange_Step, Pipeline, PipelineStep, PointCloudSaver_Step, RGBDnTBatch
from phm.pipeline.o3d_pipeline import ColoredICPRegistar_Step, O3DRegistrationMetrics_Step
from phm.pipeline.manual_pipeline import ManualRegistration_Step
from phm.visualization import visualize_vtd
from phm.vtd import load_pinhole

class Test_Registration(unittest.TestCase):

    def test_filterreg_registration(self):
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
            FilterregRegistration_Step(
                voxel_size = 0.05,
                data_pcs_key='pcs', maxiter=40),
            O3DRegistrationMetrics_Step(data_pcs_key='pcs')
        ])

        res = pipobj(batch)
        res_pc = res['fused_pc']
        print(res['metrics'])

        visualize_vtd(
            res_pc,
            load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'), 
            'PHM Visualization', 1024, 768
        )

    def test_gmmtree_registration(self):
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
            GMMTreeRegistration_Step(
                voxel_size = 0.05,
                data_pcs_key='pcs', maxiter=40),
            O3DRegistrationMetrics_Step(data_pcs_key='pcs')
        ])

        res = pipobj(batch)
        res_pc = res['fused_pc']

        visualize_vtd(
            res_pc,
            load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'), 
            'PHM Visualization', 1024, 768
        )

    def test_svr_registration(self):
        root_dir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/'
        batch = RGBDnTBatch(
            root_dir = os.path.join(root_dir, 'vtd'),
            filenames = [
                'vtd_1626967963384.mat', 'vtd_1626967965865.mat', 'vtd_1626967973439.mat' # 
            ]
        )

        depth_param_file = os.path.join(root_dir, 'depth/camera_info.json')
        depth_param = load_pinhole(depth_param_file)

        pipobj = Pipeline([
            FilterDepthRange_Step(),
            ConvertToPC_Step(
                depth_params_file = depth_param_file,
                data_batch_key = 'prp_frames'),
            SVRRegistration_Step(
                voxel_size = 0.05,
                data_pcs_key='pcs', maxiter=40),
            ColoredICPRegistar_Step(data_pcs_key='pcs', max_iter=[1, 1, 1]),
            PointCloudSaver_Step(
                data_pcs_key='aligned_pcs',
                depth_param=depth_param,
                result_dir=os.path.join(root_dir, 'results/aligned_pcs'),
                method_name='svr'
            ),
            PointCloudSaver_Step(
                data_pcs_key='fused_pc',
                depth_param=depth_param,
                result_dir=os.path.join(root_dir, 'results/fused_pcs'),
                method_name='svr'
            )
        ])

        res = pipobj(batch)
        res_pc = res['fused_pc']

        visualize_vtd(
            res_pc,
            load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'), 
            'PHM Visualization', 1024, 768
        )

    def test_cpd_registration(self):
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
            CPDRegistration_Step(
                voxel_size = 0.05,
                data_pcs_key='pcs'),
        ])

        res = pipobj(batch)
        res_pc = res['fused_pc']

        visualize_vtd(
            res_pc,
            load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'), 
            'PHM Visualization', 1024, 768
        )

    def test_manual_registration(self):
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

    def test_colored_icp_registration(self):
        # colored pointcloud registration
        # This is implementation of following paper
        # J. Park, Q.-Y. Zhou, V. Koltun,
        # Colored Point Cloud Registration Revisited, ICCV 2017

        batch = RGBDnTBatch(
            root_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd',
            filenames= [
                'vtd_1626967963384.mat', 'vtd_1626967965865.mat', 'vtd_1626967973439.mat'
            ]
        )

        pipobj = Pipeline([
            FilterDepthRange_Step(),
            ConvertToPC_Step(
                data_batch_key = 'prp_frames',
                depth_params_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'),
            ColoredICPRegistar_Step(data_pcs_key='pcs'),
            O3DRegistrationMetrics_Step(data_pcs_key='pcs')
        ])

        res = pipobj(batch)
        res_pc = res['fused_pc']
        print(res['metrics'])

        visualize_vtd(
            res_pc,
            load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'), 
            'PHM Visualization', 1024, 768
        )

if __name__ == '__main__':
    unittest.main()

