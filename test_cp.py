
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np

from phm.control_point import cpselect
from PIL import Image

from phm.process import data_fusion_cps

img1 = Image.open("/home/phm/Datasets/multi-modal/20210706_multi_modal/visible/visible_1625604430816.png")
img2 = Image.open("/home/phm/Datasets/multi-modal/20210706_multi_modal/thermal/thermal_1625604430816.png")

img1 = np.asarray(img1)
img2 = np.asarray(img2)

data_fusion_cps(img1, img2)

