

import copy
import numpy as np

from probreg import cpd
from probreg import l2dist_regs
from probreg import gmmtree
from probreg import filterreg

from phm.data.vtd import DualPointCloudPack
from phm.pipeline.core import AbstractRegistration_Step, PipelineStep

class AbstractProbregRegistration_Step(AbstractRegistration_Step):
    def __init__(self, data_pcs_key: str):
        super().__init__(data_pcs_key)
    
    def _impl_func(self, **kwargs):
        batch = kwargs['pcs']
        res_pc = list(batch[0])
        for index in range(1,len(batch)):
            source = batch[index][0]
            target = res_pc[0]
            res = self._register(source, target)
            current_transformation = res.transformation
            # Transform Visible Pointcloud
            batch[index][0].points = current_transformation.transform(batch[index][0].points)
            # Transform Thermal Pointcloud
            batch[index][1].points = current_transformation.transform(batch[index][1].points)

            res_pc[0] += batch[index][0]
            res_pc[1] += batch[index][1]
        
        return {'fused_pc' : DualPointCloudPack(res_pc[0], res_pc[1])}

class FilterregRegistration_Step(AbstractProbregRegistration_Step):
    def __init__(self, 
        data_pcs_key : str,
        voxel_size = 0.05,
        sigma2 = None,
        maxiter: int = 1,
        tol: float = 1.0e-3
    ):
        super().__init__(data_pcs_key = data_pcs_key)
        self.voxel_size = voxel_size
        self.maxiter = maxiter
        self.threshold = tol
        self.sigma2 = sigma2

    def _register(self, src, tgt):
        source = src
        target = tgt
        
        source.remove_non_finite_points()
        target.remove_non_finite_points()

        # transform target point cloud
        source_d = source.voxel_down_sample(voxel_size=self.voxel_size)
        target_d = target.voxel_down_sample(voxel_size=self.voxel_size)

        # compute cpd registration
        current_transformation = filterreg.registration_filterreg(source_d, target_d,
            maxiter = self.maxiter,
            tol = self.threshold,
            sigma2 = self.sigma2
        )

        return current_transformation

class GMMTreeRegistration_Step(AbstractProbregRegistration_Step):
    def __init__(self, 
        data_pcs_key : str,
        voxel_size = 0.05,
        maxiter: int = 1,
        tol: float = 1.0e-3
    ):
        super().__init__(data_pcs_key = data_pcs_key)
        self.voxel_size = voxel_size
        self.maxiter = maxiter
        self.threshold = tol

    def _register(self, src, tgt):
        source = src
        target = tgt
        
        source.remove_non_finite_points()
        target.remove_non_finite_points()

        # transform target point cloud
        source_d = source.voxel_down_sample(voxel_size=self.voxel_size)
        target_d = target.voxel_down_sample(voxel_size=self.voxel_size)

        # compute cpd registration
        current_transformation = gmmtree.registration_gmmtree(source_d, target_d,
            maxiter = self.maxiter,
            tol = self.threshold
        )

        return current_transformation

class SVRRegistration_Step(AbstractProbregRegistration_Step):
    def __init__(self, 
        data_pcs_key : str,
        voxel_size = 0.05,
        tf_type_name: str = "rigid",
        maxiter: int = 1,
        tol: float = 1.0e-3
    ):
        super().__init__(data_pcs_key = data_pcs_key)
        self.voxel_size = voxel_size
        self.tf_type_name = tf_type_name
        self.maxiter = maxiter
        self.threshold = tol

    def _register(self, src, tgt):
        source = src
        target = tgt
        
        source.remove_non_finite_points()
        target.remove_non_finite_points()

        # transform target point cloud
        source_d = source.voxel_down_sample(voxel_size=self.voxel_size)
        target_d = target.voxel_down_sample(voxel_size=self.voxel_size)
        # compute cpd registration
        current_transformation = l2dist_regs.registration_svr(source_d, target_d,
            tf_type_name = self.tf_type_name,
            opt_maxiter = self.maxiter,
            opt_tol = self.threshold
        )

        return current_transformation

    def _impl_func(self, **kwargs):
        batch = kwargs['pcs']
        res_pc = list(batch[0])
        for index in range(1,len(batch)):
            source = batch[index][0]
            target = res_pc[0]
            current_transformation = self._register(source, target)
            # Transform Visible Pointcloud
            batch[index][0].points = current_transformation.transform(batch[index][0].points)
            # Transform Thermal Pointcloud
            batch[index][1].points = current_transformation.transform(batch[index][1].points)

            res_pc[0] += batch[index][0]
            res_pc[1] += batch[index][1]
        
        return {'fused_pc' : DualPointCloudPack(res_pc[0], res_pc[1])}


class CPDRegistration_Step(AbstractProbregRegistration_Step):
    def __init__(self, 
        data_pcs_key : str,
        tf_type_name : str = 'rigid', # type('rigid', 'affine', 'nonrigid', 'nonrigid_constrained')
        voxel_size = 0.05,
        maxiter = 50, 
        tol = 0.001,

    ):
        super().__init__(data_pcs_key = data_pcs_key)
        self.voxel_size = voxel_size
        self.tf_type_name = tf_type_name
        self.maxiter = maxiter
        self.threshold = tol

    def _register(self, src, tgt):
        source = src
        target = tgt
        
        source.remove_non_finite_points()
        target.remove_non_finite_points()

        # transform target point cloud
        th = np.deg2rad(30.0)
        source_d = source.voxel_down_sample(voxel_size=self.voxel_size)
        target_d = target.voxel_down_sample(voxel_size=self.voxel_size)
        # compute cpd registration
        current_transformation  = cpd.registration_cpd(source_d, target_d,
            tf_type_name = self.tf_type_name,
            maxiter = self.maxiter,
            tol = self.threshold
        )

        return current_transformation


