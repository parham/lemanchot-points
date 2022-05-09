
import logging
import unittest
import sys,os

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phm.data import save_mme, MMEContainer, create_mme_dataset

class Test_MME_Record(unittest.TestCase):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
    )

    def test_save_mme(self):
        container = MMEContainer()
        container.set_metadatas({
            'name' : 'Parham',
            'family' : 'Nooralishahi',
            'age' : 34,
            'gpa' : 4.22
        })
        container.add_entity(
            type='visible',
            file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/visible/visible_1625604430816.png'
        )
        container.add_entity(
            type='thermal',
            file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/thermal/thermal_1625604430816.png'
        )
        container.add_entity(
            type='depth',
            file='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/depth_1625604430816.png'
        )

        save_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/test.mme', container, file_type='mme')
        save_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/test.mat', container, file_type='mat')
    
    def test_create_dataset(self):
        dir = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal'
        create_mme_dataset(root_dir=dir, file_type='mme')
        create_mme_dataset(root_dir=dir, file_type='mat')

if __name__ == '__main__':
    unittest.main()