
import os
import sys
import logging
import unittest


sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from phm.data import MMEContainer, MMERecord
from phm.io import load_entity, save_mme


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

        vis_file = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/visible/visible_1625604430816.png'
        thermal_file = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/thermal/thermal_1625604430816.png'
        depth_file = '/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/depth_1625604430816.png'
        
        container.add_entity(MMERecord(
            type='visible',
            file=vis_file,
            data=load_entity('visible', vis_file)
        ))
        container.add_entity(MMERecord(
            type='thermal',
            file=thermal_file,
            data=load_entity('thermal', thermal_file)
        ))
        container.add_entity(MMERecord(
            type='depth',
            file=depth_file,
            data=load_entity('depth', depth_file)
        ))

        save_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/test.mme', container, file_type='mme')
        save_mme('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/test.mat', container, file_type='mat')

if __name__ == '__main__':
    unittest.main()