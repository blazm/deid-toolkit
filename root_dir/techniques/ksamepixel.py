
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter

def k_same_pixel(clusters, clustered_images, k=2):
    
    deidentified = {}

    for k in clusters.keys():
        imgs = np.array(clustered_images[k])
        di = np.mean(imgs, axis=0) # check if axis ok 
        deidentified[k] = di
    
    return deidentified