
import numpy as np

modal_to_image = lambda img : (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.0).astype(np.uint8)
gray_to_rgb = lambda img : img if len(img.shape) > 2 and img.shape[2] == 3 else np.stack((img, img, img), axis=2)
