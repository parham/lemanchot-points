
import cv2
import numpy as np

from phm.control_point import cp_to_opencv, cpselect


def data_packing(img_1 : np.ndarray, img_2 : np.ndarray):
    # STEP 01 : Run control point selector toolbox
    cps = cpselect(img_1, img_2)
    source, dest = cp_to_opencv(cps)
    # STEP 02 : estimate homography transformation
    h, status = cv2.findHomography(source, dest)
    status.ravel()
    print(status)