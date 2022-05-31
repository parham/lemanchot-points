
import copy
import logging
import os
import sys
import numpy as np
import open3d as o3d

from typing import List

from send2trash import TrashPermissionError

from phm.data import RGBDnT
from phm.data.vtd import __depth_scale__
from phm.pipeline import Pipeline
from phm.pipeline.core import PipelineStep

from phm.visualization import VTD_Visualization
from phm.vtd import load_pinhole
import open3d.visualization.gui as gui

class ColoredICPRegistar_Step(PipelineStep):
    def __init__(self,
        depth_params_file : str,
        voxel_radius = [0.04, 0.02, 0.01],
        max_iter = [50, 30, 16],
        data_batch_key : str = 'batch'
    ):
        super().__init__({
            'batch' : data_batch_key
        })
        self.depth_params = load_pinhole(depth_params_file)
        self.voxel_radius = voxel_radius
        self.max_iter = max_iter
    
    def _calc_transform(self, batch):
        for index in range(len(batch)-1):
            fixed = batch[index]
            moving = batch[index+1]

            fixed_viz = fixed[0]
            moving_viz = moving[0]
            
            current_transformation = np.identity(4)
            # Transformations
            transformations = []
            transformations.append(np.identity(4))
            for scale in range(3):
                try:
                    iter = self.max_iter[scale]
                    radius = self.voxel_radius[scale]
                    print([iter, radius, scale])

                    print("3-1. Downsample with a voxel size %.2f" % radius)
                    fixed_down = fixed_viz.voxel_down_sample(radius)
                    moving_down = moving_viz.voxel_down_sample(radius)

                    print("3-2. Estimate normal.")
                    fixed_down.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                    moving_down.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

                    print("3-3. Applying colored point cloud registration")
                    result_icp = o3d.pipelines.registration.registration_colored_icp(
                        fixed_down, moving_down, radius, current_transformation,
                        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                            relative_rmse=1e-6,
                                                                            max_iteration=iter))
                    current_transformation = result_icp.transformation
                except Exception as ex:
                    logging.warning(ex)

            last_transformation = transformations[-1]
            updated_transformation = current_transformation * last_transformation
            transformations.append(updated_transformation)
            return transformations

    def _impl_func(self, **kwargs):
        batch = kwargs['batch']
        # Colored ICP registration
        # This is implementation of following paper
        # J. Park, Q.-Y. Zhou, V. Koltun,
        # Colored Point Cloud Registration Revisited, ICCV 2017
        print('Registering using Colored ICP method ...')
        transformations = self._calc_transform(batch)
        for index in range(transformations):
            trans = transformations[index]
            pcviz, pcth = batch[index]
            pcviz.transform(trans)
            pcth.transform(trans)


class PHM_ICP_Pipeline (Pipeline):

    def __init__(self, 
        root_dir: str, 
        filenames: List[str],
        depth_range = [1, 2.5]
    ):
        super().__init__(root_dir, filenames)
        self.depth_range = depth_range
    
    def execute(self):
        # 1. Loading Data Frames
        print('Loading Data Frames ...')
        frames = self.load_data_frames()
        # 2. Preprocess the RGBD&T frames
        print('Preprocess the RGBD&T frames ...')
        frames = list(map(lambda x : self.__apply_depth_range(x, self.depth_range), frames))
        # 3. Apply Colored ICP Registration
        print('Registering using Colored ICP method ...')

        # Colored ICP registration
        # This is implementation of following paper
        # J. Park, Q.-Y. Zhou, V. Koltun,
        # Colored Point Cloud Registration Revisited, ICCV 2017
        voxel_radius = [0.04, 0.02, 0.01]
        max_iter = [50, 30, 16]
        depth_params = load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json')
        
        current_transformation = np.identity(4) 
        
        source = frames[0].to_point_cloud_visible_o3d(depth_params, False)
        target = frames[2].to_point_cloud_visible_o3d(depth_params, False)

        for scale in range(3):
            try:
                iter = max_iter[scale]
                radius = voxel_radius[scale]
                print([iter, radius, scale])

                print("3-1. Downsample with a voxel size %.2f" % radius)
                source_down = source.voxel_down_sample(radius)
                target_down = target.voxel_down_sample(radius)

                print("3-2. Estimate normal.")
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

                print("3-3. Applying colored point cloud registration")
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=iter))
                current_transformation = result_icp.transformation
            except Exception as ex:
                logging.warning(ex)

        def draw_registration_result_original_color(source, target, transformation):
            source_temp = copy.deepcopy(source)
            source_temp.transform(transformation)
            o3d.visualization.draw_geometries([source_temp, target])

        print("-- Visualize registered point cloud")
        draw_registration_result_original_color(source, target, result_icp.transformation)

    def __apply_depth_range(self, frame : RGBDnT, drange):
        depth = frame.depth_image
        # Filter out depth values
        depth[depth < (drange[0] * __depth_scale__)] = 0
        depth[depth > (drange[1] * __depth_scale__)] = 0
        frame.depth_image = depth

        return frame
