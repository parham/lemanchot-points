
from os import wait
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image

import open3d as o3d

from phm.data import load_mme, gray_to_rgb
from phm.process_vtd import VTD_Alignment, blend_vt

# img = cv.imread('/home/phm/Datasets/multi-modal/20210706_multi_modal/visible/visible_1625604430816.png', 0)
# # Initiate ORB detector
# orb = cv.ORB_create()
# # find the keypoints with ORB
# kp = orb.detect(img,None)
# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp)
# # draw only keypoints location,not size and orientation
# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()

data = load_mme('/home/phm/Datasets/multi-modal/20210706_multi_modal/mat/mme_1625604430816.mat', 'mat')
vtd = VTD_Alignment(
    target_dir = '/home/phm/Datasets/multi-modal/20210706_multi_modal',
    depth_param_file = '/home/phm/Datasets/multi-modal/20210706_multi_modal/depth/camera_info.json'
)
vtd.load()

rgbdt = vtd.compute(data)

o3d.visualization.draw_geometries([rgbdt.pcs_thermal])

# timg = Image.fromarray(rgbdt.thermal)
# timg.show()
# thermal_gray = timg.convert('L')
# thermal_gray.save('/home/phm/Datasets/multi-modal/20210706_multi_modal/a.png')
# # Blend
# fused = blend_vt(rgbdt) 
# fused_img = Image.fromarray(fused)
# fused_img.show()
# fused_img.save('/home/phm/Datasets/multi-modal/20210706_multi_modal/fused.png')

# vtd.compute_point_cloud(rgbdt)

#     [fx  0 cx] [  383.53265380859375   0                    318.4489440917969   ]
# K = [ 0 fy cy] [  0                    383.53265380859375   236.36776733398438  ]
#     [ 0  0  1] [  0                    0                    1                   ]

# vizimg = Image.blend(visible, thermal_gray, alpha=0.4)
# vizimg.show()