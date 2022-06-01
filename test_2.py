
import open3d as o3d

import phm.pipeline as pip
from phm.pipeline.core import ConvertToPC_Step, FilterDepthRange_Step, Pipeline, RGBDnTBatch
from phm.pipeline.icp_pipeline import ColoredICPRegistar_Step

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
    ColoredICPRegistar_Step(data_batch_key='pcs')
])

res = pipobj(batch)

tmp = [th for (viz, th) in res['pcs']]
o3d.visualization.draw_geometries([tmp[0]], mesh_show_wireframe=True)
# print(res)

# obj = pip.PHM_ICP_Pipeline(
#     root_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd',
#     filenames= [
#         'vtd_1626967963384.mat', 'vtd_1626967965865.mat', 'vtd_1626967973439.mat'
#     ]
# )

# obj.execute()