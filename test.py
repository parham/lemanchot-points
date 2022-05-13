
import open3d as o3d

from phm.io.vtd import load_point_cloud

file = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/pc/pcs_1625604432756.ply'
# pcd = o3d.io.read_point_cloud(file)

pcd = load_point_cloud(file, 'ply_txt')
print(pcd)