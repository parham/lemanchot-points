
import numpy as np

from phm.data.vtd import RGBDnT

modal_to_image = lambda img : (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.0).astype(np.uint8)

gray_to_rgb = lambda img : img if len(img.shape) > 2 and img.shape[2] == 3 else np.stack((img, img, img), axis=2)

ftype_to_filext = lambda x : x.split('_')[0]

def show_modalities_grid(data : RGBDnT):
    import numpy as np
    import matplotlib.pyplot as plt

    w = 10
    h = 10
    fig = plt.figure(figsize=(2, 2))
    
    fig.add_subplot(2, 2, 1)
    plt.imshow(data.visible)
    fig.add_subplot(2, 2, 2)
    plt.imshow(data.thermal)
    fig.add_subplot(2, 2, 3)
    plt.imshow(data.depth)
    
    plt.show()

def blend_vt(data : RGBDnT, alpha : float = 0.6):
    thermal_rgb = gray_to_rgb(data.thermal)
    fused = (data.visible.copy()).astype(np.float64)
    ttemp = thermal_rgb * alpha
    vtemp = data.visible * (1 - alpha)
    fused[thermal_rgb > 0] = vtemp[thermal_rgb > 0] + ttemp[thermal_rgb > 0]

    return modal_to_image(fused)