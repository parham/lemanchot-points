
import os
import sys
import unittest


sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phm.io import load_mme, load_RGBDnT
from phm.dataset import Dataset_LoadableFunc, VTD_Dataset, create_point_cloud_dataset, create_vtd_dataset

class Test_Dataset(unittest.TestCase):
    def test_mat_dataset(self):
        dataset = Dataset_LoadableFunc(
            '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/mat',
            'mat',
            load_mme
        )
        print(f'Number of Samples : {len(dataset)}')
        counter = 0
        for x in dataset:
            print(x)
            counter += 1
    
    def test_load_vtd_dataset(self):
        dataset = VTD_Dataset(
            '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/vtd'
        )
        print(f'Number of Samples : {len(dataset)}')
        counter = 0
        for x in dataset:
            print(x)
            counter += 1

    def test_create_vtd_dataset(self):
        create_vtd_dataset(
            in_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/mat',
            target_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/vtd',
            depth_param_file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/camera_info.json',
            in_type='mat'
        )

    def test_create_point_cloud_dataset(self):
        create_point_cloud_dataset(
            in_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/vtd',
            target_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/pc',
            file_type='ply_txt'
        )

if __name__ == '__main__':
    unittest.main()