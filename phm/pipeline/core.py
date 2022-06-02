
import os

from typing import Any, Callable, Dict, List
from progress.bar import Bar

from phm.data import RGBDnT
from phm.io import load_RGBDnT
from phm.data.vtd import __depth_scale__
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
                res = {**res, **step(**res)}
                bar.next()
        
        return res