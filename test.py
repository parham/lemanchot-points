
from os import wait
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image

from phm.data import load_mme, gray_to_rgb
from phm.process_vtd import VTD_Alignment, blend_vt

print('test')