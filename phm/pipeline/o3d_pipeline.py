
import numpy as np
import open3d as o3d

from phm.data.vtd import DualPointCloudPack, __depth_scale__
from phm.pipeline.core import AbstractRegistration_Step, PipelineStep

class O3DRegistrationMetrics_Step(PipelineStep):
    def __init__(self, 
        data_pcs_key : str,
        initial_transformation = None
    ):
        super().__init__({'pcs' : data_pcs_key})
        self.initial_transformation = initial_transformation
        self.__reset_metrics()

    def __reset_metrics(self):
        self.metrics = {
            'inlier_rmse' : [], # RMSE of all inlier correspondences. Lower is better.
            'fitness' : [] # The overlapping area (# of inlier correspondences / # of points in source). Higher is better.
        }

    def _transform_point_cloud(self, data, transformation):
        # Transform Visible Pointcloud
        data[0].transform(transformation)
        # Transform Thermal Pointcloud
        data[1].transform(transformation)
        return data

    def _register(self, src, tgt):
        source = src
        target = tgt

        radius = 0.06
        iter = 1
        current_transformation = self.initial_transformation if self.initial_transformation is not None else np.identity(4)

        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        
        return result_icp

    def _impl_func(self, **kwargs):
        self.__reset_metrics()
        batch = kwargs['pcs']

        pcs = list()
        res_pc = list(batch[0])
        for index in range(1,len(batch)):
            source = batch[index][0]
            target = res_pc[0]
            res_transformation = self._register(source, target)
            batch[index] = self._transform_point_cloud(batch[index], res_transformation.transformation)

            res_pc[0] += batch[index][0]
            res_pc[1] += batch[index][1]

            self.metrics['inlier_rmse'].append(res_transformation.inlier_rmse)
            self.metrics['fitness'].append(res_transformation.fitness)

        return {
            'metrics' : self.metrics
        }

class ColoredICPRegistar_Step(AbstractRegistration_Step):
    def __init__(self,
        voxel_radius = [0.04, 0.02, 0.01],
        max_iter = [50, 30, 16],
        voxel_size = 0.05,
        data_pcs_key : str = 'pcs',
        initial_transformation = None
    ):
        super().__init__(data_pcs_key = data_pcs_key)
        self.voxel_radius = voxel_radius
        self.max_iter = max_iter
        self.voxel_size = voxel_size
        self.initial_transformation = initial_transformation

    def _register(self, src, tgt):
        source = src
        target = tgt

        voxel_radius = [0.06, 0.04, 0.04]
        max_iter = [50, 30, 14]

        current_transformation = self.initial_transformation if self.initial_transformation is not None else np.identity(4)
        print("Colored point cloud registration")
        for scale in range(3):
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
    
        return current_transformation

class Easy_MultiModalPCFusion_Step(PipelineStep):
    def __init__(self, data_pcs_key : str = 'pcs'):
        super().__init__({
            'pcs' : data_pcs_key
        })
    
    def _impl_func(self, **kwargs):
        batch = kwargs['pcs']
        vfused = o3d.geometry.PointCloud()
        tfused = o3d.geometry.PointCloud()
        for vpc, thpc in batch:
            vfused += vpc
            tfused += thpc
        
        fusedpc = DualPointCloudPack(
            vfused.voxel_down_sample(voxel_size=0.005), 
            tfused.voxel_down_sample(voxel_size=0.005)
        )

        return {
            'fused_pc' : fusedpc
        }