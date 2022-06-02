
import logging
import numpy as np
import open3d as o3d

from phm.data.vtd import DualPointCloudPack, __depth_scale__
from phm.pipeline.core import PipelineStep

class ColoredICPRegistar_Step(PipelineStep):
    def __init__(self,
        voxel_radius = [0.04, 0.02, 0.01],
        max_iter = [50, 30, 16],
        voxel_size = 0.05,
        data_pcs_key : str = 'pcs'
    ):
        super().__init__({
            'pcs' : data_pcs_key
        })
        self.voxel_radius = voxel_radius
        self.max_iter = max_iter
        self.voxel_size = voxel_size
    
    def _calc_transform(self, batch):
        transformations = []
        # Add Identity matrix
        transformations.append(np.identity(4))
        for index in range(1, len(batch)):
            fixed = batch[index-1]
            moving = batch[index]

            fixed_viz = fixed[0]
            moving_viz = moving[0]

            fixed_down, fixed_fpfh = self._preprocess_point_cloud(fixed_viz, self.voxel_size)
            moving_down, moving_fpfh = self._preprocess_point_cloud(moving_viz, self.voxel_size)

            initial_reg = self._execute_global_registration(fixed_down, moving_down, fixed_fpfh, moving_fpfh, self.voxel_size)

            last_transformation = transformations[-1]
            current_transformation = last_transformation * initial_reg.transformation
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
            
            updated_transformation = last_transformation * current_transformation
            transformations.append(updated_transformation)

        return transformations

    def _execute_global_registration(self,
        source_down, target_down, 
        source_fpfh, target_fpfh, voxel_size
    ):
        distance_threshold = voxel_size * 1.5
        print("RANSAC registration on downsampled point clouds.")
        print("Since the downsampling voxel size is %.3f," % voxel_size)
        print("We use a liberal distance threshold %.3f." % distance_threshold)
        # result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        #     source_down, target_down, source_fpfh, target_fpfh,
        #     o3d.pipelines.registration.FastGlobalRegistrationOption(
        #         maximum_correspondence_distance=distance_threshold))
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, 
            source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(), 3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.3),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    def _preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def _impl_func(self, **kwargs):
        batch = kwargs['pcs']
        # Colored ICP registration
        # This is implementation of following paper
        # J. Park, Q.-Y. Zhou, V. Koltun,
        # Colored Point Cloud Registration Revisited, ICCV 2017
        res = []
        print('Registering using Colored ICP method ...')
        transformations = self._calc_transform(batch)
        for index in range(len(transformations)):
            trans = transformations[index]
            pcviz, pcth = batch[index]
            pcviz.transform(trans)
            pcth.transform(trans)
            res.append((pcviz, pcth))

        return {
            'pcs' : res 
        }
        
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