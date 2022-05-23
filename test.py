
import os
import sys
import open3d.visualization.gui as gui

from phm.io import load_RGBDnT
from phm.visualization import VTD_Visualization
from phm.vtd import load_pinhole

data = load_RGBDnT('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/vtd/vtd_1625604434719.mat')

gui.Application.instance.initialize()

w = VTD_Visualization(data, 
    load_pinhole('/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210706_multi_modal/depth/camera_info.json'), 
    1024, 768)
w.load_visible()

if len(sys.argv) > 1:
    path = sys.argv[1]
    if os.path.exists(path):
        w.load(path)
    else:
        w.window.show_message_box("Error", "Could not open file '" + path + "'")

# Run the event loop. This will not return until the last window is closed.
gui.Application.instance.run()