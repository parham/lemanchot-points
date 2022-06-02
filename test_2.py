
import open3d as o3d
import open3d.visualization.gui as gui

from phm.pipeline.core import ConvertToPC_Step, FilterDepthRange_Step, Pipeline, RGBDnTBatch
from phm.pipeline.icp_pipeline import ColoredICPRegistar_Step, Easy_MultiModalPCFusion_Step
from phm.visualization import VTD_Visualization
from phm.vtd import load_pinhole

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
    ColoredICPRegistar_Step(data_pcs_key='pcs'),
    Easy_MultiModalPCFusion_Step(data_pcs_key='pcs')
])

res = pipobj(batch)

# tmp = [th for (viz, th) in res['pcs']]
pc = res['fused_pc']

gui.Application.instance.initialize()
w = VTD_Visualization(pc, 
    load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json'), 
    'PHM Visualization', 1024, 768)

# Run the event loop. This will not return until the last window is closed.
gui.Application.instance.run()

# o3d.visualization.draw_geometries([pc], mesh_show_wireframe=True)

