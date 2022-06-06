
import os
from sys import prefix

from typing import Dict, List, Tuple
from progress.bar import Bar
from pathlib import Path

import open3d as o3d

from phm.data import RGBDnT
from phm.io import load_RGBDnT
from phm.data.vtd import DualPointCloudPack, __depth_scale__
from phm.vtd import load_pinhole

class RGBDnTBatch:
    def __init__(self, root_dir : str, filenames : List[str]):
        super().__init__()
        # Check file availability
        if len(filenames) == 0:
            raise ValueError('No RGBD&T file exist!')
        # Check Root Directory Availability
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f'{root_dir} does not exist!')
        # Initialize the file list
        files = tuple(map(lambda x : os.path.join(root_dir, x), filenames))
        files = tuple(filter(os.path.isfile, files))
        self.files = files
    
    @property
    def count(self):
        return len(self.files)

    def __call__(self):
        return (load_RGBDnT(fx) for fx in self.files)

class PipelineStep:
    def __init__(self, key_arg_map : Dict[str,str]):
        self.key_map = key_arg_map

    def _generate_args(self, **kwargs):
        if not all(x in kwargs.keys() for x in self.key_map.keys()):
            raise ValueError('Required data is not provided!')
        res = {}
        for arg, key in self.key_map.items():
            res[arg] = kwargs[key]
        return res

    def _impl_func(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return self._impl_func(**self._generate_args(**kwargs))

class PointCloudSaver_Step(PipelineStep):
    def __init__(self, 
        data_pcs_key : str,
        result_dir : str,
        depth_param,
        method_name : str = 'pc'):
        super().__init__({'pcs' : data_pcs_key})
        self.result_dir = result_dir
        self.method_name = method_name
        self.depth_param = depth_param
        # Make the directory if not exist!
        Path(result_dir).mkdir(parents=True, exist_ok=True)

    def _impl_func(self, **kwargs):
        batch = kwargs['pcs']
        pcs = batch if isinstance(batch, list) else [batch]
        index = 1
        print(f'\nTotal Number of Point Clouds : {len(pcs)}')
        for pc in pcs:
            fname_viz = f'{self.method_name}_visible_{index}.ply'
            file_viz = os.path.join(self.result_dir, fname_viz)
            fname_th = f'{self.method_name}_thermal_{index}.ply'
            file_th = os.path.join(self.result_dir, fname_th)
            print(f'Saving {fname_viz} (Visible) ...')
            o3d.io.write_point_cloud(file_viz, 
                pc.get_visible_point_cloud(intrinsic=self.depth_param), 
                write_ascii = True, print_progress = True)
            print(f'Saving {fname_th} (Thermal) ...')
            o3d.io.write_point_cloud(file_th, 
                pc.get_thermal_point_cloud(intrinsic=self.depth_param), 
                write_ascii = True, print_progress = True)
            index += 1

class AbstractRegistration_Step(PipelineStep):
    def __init__(self, data_pcs_key : str):
        super().__init__({'pcs' : data_pcs_key})
        self.pcs_key = data_pcs_key
    
    def _impl_func(self, **kwargs):
        batch = kwargs['pcs']

        pcs = list([DualPointCloudPack(batch[0][0], batch[0][1])])
        res_pc = list(batch[0])
        for index in range(1,len(batch)):
            source = batch[index][0]
            target = res_pc[0]
            current_transformation = self._register(source, target)
            batch[index] = self._transform_point_cloud(batch[index], current_transformation)

            res_pc[0] += batch[index][0]
            res_pc[1] += batch[index][1]
            pcs.append(DualPointCloudPack(batch[index][0], batch[index][1]))

        return {
            'aligned_pcs' : pcs,
            f'{self.pcs_key}' : batch,
            'fused_pc' : DualPointCloudPack(res_pc[0], res_pc[1])
        }
    
    def _transform_point_cloud(self, data : Tuple, transformation):
        # Transform Visible Pointcloud
        data[0].transform(transformation)
        # Transform Thermal Pointcloud
        data[1].transform(transformation)
        return data

    def _register(self, src, tgt):
        pass

class FilterDepthRange_Step(PipelineStep):
    def __init__(self, depth_range = [1, 2.5], data_batch_key : str = 'batch'):
        super().__init__({
            'batch' : data_batch_key
        })
        self.depth_range = depth_range
    
    def __apply_depth_range(self, frame : RGBDnT):
        depth = frame.depth_image
        drange = self.depth_range
        # Filter out depth values
        depth[depth < (drange[0] * __depth_scale__)] = 0
        depth[depth > (drange[1] * __depth_scale__)] = 0
        frame.depth_image = depth

        return frame

    def _impl_func(self, **kwargs):
        batch = kwargs['batch']
        return {
            'prp_frames' : list(map(lambda x : self.__apply_depth_range(x), batch()))
        }

class ConvertToPC_Step(PipelineStep):
    def __init__(self,
        depth_params_file : str,
        data_batch_key : str = 'batch'):
        super().__init__({
            'batch' : data_batch_key
        })
        self.depth_params = load_pinhole(depth_params_file)
    
    def _impl_func(self, **kwargs):
        data = kwargs['batch']
        return {
            'pcs': list(map(self.__convert_pc, data))
        }
    
    def __convert_pc(self, data : RGBDnT):
        return (
            data.to_point_cloud_visible_o3d(self.depth_params, False),
            data.to_point_cloud_thermal_o3d(self.depth_params, False)
        )

class Pipeline(object):

    def __init__(self, steps : List[PipelineStep] = None):
        self.steps = steps if steps is not None else []
    
    @property
    def steps_count(self):
        return len(self.steps)

    def __call__(self, batch : RGBDnTBatch):
        # Check data availability
        if batch.count == 0:
            return None
        
        res = {'batch' : batch}
        with Bar('Processing using pipeline steps', max=self.steps_count) as bar:
            for step in self.steps:
                d_res = step(**res)
                if d_res is not None:
                    res = {**res, **d_res}
                bar.next()
        
        return res