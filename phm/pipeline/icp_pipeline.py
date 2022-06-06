
import numpy as np
import open3d as o3d

from phm.data.vtd import DualPointCloudPack, __depth_scale__
from phm.pipeline.core import AbstractRegistration_Step, PipelineStep

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