import numpy as np
import pandas as pd
from skimage.io import imshow, imsave
from matplotlib import pyplot as plt
from skimage.filters import gaussian

def show_pic(pic, idx=[0, 1, 2], sigma=10, figsize=(30,30), dpi=80, plugin='matplotlib'):
    g = pic[:,:,idx].copy()
    g -= g.min()
    g /= g.max()
    g *= 255
    g = gaussian(g, sigma=sigma, multichannel=True)
    #plt.figure(figsize=figsize, dpi=dpi)
    #imshow(g.astype(np.int32), plugin=plugin)
    return g