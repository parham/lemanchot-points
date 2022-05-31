
import phm.pipeline as pip
from phm.pipeline.core import ConvertToPC_Step, FilterDepthRange_Step, Pipeline, RGBDnTBatch

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
        depth_params_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json')
])

res = pipobj(batch)
# print(res)

# obj = pip.PHM_ICP_Pipeline(
#     root_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd',
#     filenames= [
#         'vtd_1626967963384.mat', 'vtd_1626967965865.mat', 'vtd_1626967973439.mat'
#     ]
# )

# obj.execute()