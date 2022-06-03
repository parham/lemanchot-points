

import numpy as np
import open3d as o3d
from phm.data.vtd import DualPointCloudPack

from phm.pipeline.core import PipelineStep
from phm.visualization import pick_points_point_cloud


class ManualRegistration_Step(PipelineStep):
    def __init__(self, 
        depth_params,
        data_pcs_key : str = 'pcs'):
        super().__init__({
            'pcs' : data_pcs_key
        })
        self.depth_params = depth_params
    
    def _impl_func(self, **kwargs):
        batch = kwargs['pcs']
        res_pc = list(batch[0])
        for index in range(1,len(batch)):
            source = batch[index][0]
            target = res_pc[0]
            current_transformation = self._register(source, target)
            # Transform Visible Pointcloud
            batch[index][0].transform(current_transformation)
            # Transform Thermal Pointcloud
            batch[index][1].transform(current_transformation)
            res_pc[0] += batch[index][0]
            res_pc[1] += batch[index][1]
        
        return {'fused_pc' : DualPointCloudPack(res_pc[0], res_pc[1])}
    
    def _register(self, src, tgt):
        source = src
        target = tgt

        print('Selecting control points for source point cloud!')
        picked_id_source = pick_points_point_cloud(source)
        print('Selecting control points for target point cloud!')
        picked_id_target = pick_points_point_cloud(target)

        if len(picked_id_source) <= 3 or len(picked_id_target) <= 3:
            raise ValueError('Number of control points are not sufficient!')
        if len(picked_id_source) != len(picked_id_target):
            raise ValueError('Number of selected control points in both point clouds are not the same!')

        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target
        
        # Compute a rough transform using the correspondences given by user
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))
        # Point-To-Point ICP for refinement
        threshold = 0.05  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        transformation = reg_p2p.transformation
        return transformation
        

