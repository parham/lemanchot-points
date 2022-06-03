
import copy
import numpy as np
import open3d as o3

from probreg import cpd

# load source and target point cloud
source = o3.io.read_point_cloud('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/pc/pcs_1626967965865.ply')
source.remove_non_finite_points()
target = o3.io.read_point_cloud('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/pc/pcs_1626967976820.ply')
target.remove_non_finite_points()
# transform target point cloud
th = np.deg2rad(30.0)
source_d = source.voxel_down_sample(voxel_size=0.05)
target_d = target.voxel_down_sample(voxel_size=0.05)

# compute cpd registration
tf_param, _, _ = cpd.registration_cpd(source_d, target_d)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

result += target
o3.visualization.draw_geometries([result])