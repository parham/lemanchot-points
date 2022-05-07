
import logging
import unittest
import sys,os

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image

from phm.control_point import cpselect
from phm.process_vtd import data_packing

class Test_CPS(unittest.TestCase):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
    )

    def test_cpselect(self):
        img1 = Image.open("/home/phm/Datasets/multi-modal/20210706_multi_modal/visible/visible_1625604430816.png")
        img2 = Image.open("/home/phm/Datasets/multi-modal/20210706_multi_modal/thermal/thermal_1625604430816.png")

        img1 = np.asarray(img1)
        img2 = np.asarray(img2)

        data_packing(img1, img2)

if __name__ == '__main__':
    unittest.main()