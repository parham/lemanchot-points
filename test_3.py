
import copy
import os
import sys

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from phm.io.vtd import load_RGBDnT

from phm.visualization import pick_points
from phm.vtd import load_pinhole

data = load_RGBDnT('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd/vtd_1626967976820.mat')
pinhole = load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/depth/camera_info.json')

ps = pick_points(data, pinhole)