import numpy as np
from shared import normalize
from scipy.misc import imsave, imread
from skimage.transform import resize as imresize
from os import path
import random
from math import sqrt
from time import time
from skimage.exposure import equalize_hist, adjust_gamma

import pickle
from maps import Config, DiskMap, DigitalElevationModel

with open(path.join(Config.BASE_DIR, 'dem.pkl'), 'rb') as fp:
    dem = pickle.load(fp)

print(dem, dem.shape, dem.preview_roi)

# Load vertical and horizontal gradient maps
sobelv = DiskMap(path.join(Config.WORKING_DIR, 'sobelv8.npy'), dtype=np.int8, mode='r', shape=dem.shape)
sobelh = DiskMap(path.join(Config.WORKING_DIR, 'sobelh8.npy'), dtype=np.int8, mode='r', shape=dem.shape)
elevation = DiskMap(path.join(Config.WORKING_DIR, 'dem8.npy'), dtype=np.uint8, mode='r', shape=dem.shape)
background = sobelh.slice_normalize(dem.preview_roi)

#hist, bin_edges = np.histogram(elevation, 64)
#print('Elevation histogram', (hist.astype(dtype=np.float))/(np.prod(elevation.shape)), bin_edges)

h_ = dem.preview_roi[3] - dem.preview_roi[1]
w_ = dem.preview_roi[2] - dem.preview_roi[0]
roi_shape = (h_, w_, 3)

def index_to_yx(index, rows, cols):
    y = index // cols
    x = index % cols
    return (y, x)

def format_time(t):
    t = int(t*1000)
    return '{}:{}:{}.{}'.format(str(t//3600000).zfill(2), str(t%3600000//60000).zfill(2), str(t%60000//1000).zfill(2), str(t%1000).zfill(3))

trace_map = DiskMap(path.join(Config.WORKING_DIR, 'trace.npy'), dtype=Config.TYPE_FLOAT, mode='w+', shape=dem.shape)
height, width, *_ = dem.shape
length = height * width

seed = random.randrange(length)
multiplier = 6364136223846793005
increment = 1442695040888963407
modulus = 2 ** 64

t_init = time()
t = 0
t_save = 0
iterations = length // 4
for i in range(iterations):

    # Debug information
    if time() - t > 2:
        elapsed = time() - t_init
        ete = 'N/A'
        if i > 0: ete = format_time(elapsed*iterations/i - elapsed)
        percent = i*10000//iterations/100
        print('Pixels {} of {}, {}%, elapsed: {}, ete: {}'.format(i, iterations, percent, format_time(elapsed), ete))
        t = time()
    if time() - t_save > 600:
        rgb_map = np.ndarray(roi_shape, dtype=np.uint8)
        trace = trace_map.slice_normalize(dem.preview_roi)
        rgb_map[:, :, 0] = trace
        rgb_map[:, :, 1] = background
        print('Snapshot for preview', rgb_map.shape)
        imsave(path.join(Config.PREVIEW_DIR, 'trace.png'), rgb_map)    
        del trace
        t_save = time()

    # Initial point of a climber
    py, px = index_to_yx(seed, height, width)
    
    if elevation[py, px] < 8:
        # Skip initial points on sea or plain
        pass

    else:
        # Momentum of climber
        my = 0
        mx = 0
        # Last position of climber
        y_ = -1
        x_ = -1
        for s in range(128):
            # Move climber
            my += sobelv[py, px]
            mx += sobelh[py, px]
            if my > 127:
                py += 1
                my = my % 127
            elif my < -127:
                py -= 1
                my = my % -127
            if mx > 127:
                px += 1
                mx = mx % 127
            elif mx < -127:
                px -= 1
                mx = mx % -127

            # Trace climber
            if py < 0 or py >= height or px < 0 or px >= width:
                break
            if trace_map[py, px] >= 64:
                pass
            elif py != y_ or px != x_:
                # Don't plot same pixel twice
                trace_map[py, px] += 1.
                #trace_map[py, px] = 1.

            y_ = py
            x_ = px
            #print('move', v, v[0]*v[0]+v[1]*v[1], p)

    seed = (multiplier*seed + increment) % (length + 1)

del background, trace_map, sobelv, sobelh, elevation