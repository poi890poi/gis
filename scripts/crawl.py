import numpy as np
from shared import normalize
from scipy.misc import imsave, imread
from skimage.transform import resize as imresize
import os.path
import random
from math import sqrt
from time import time
from skimage.exposure import equalize_hist, adjust_gamma

working_dir = '../../data'

# Load vertical and horizontal gradient maps
#sobelv = np.memmap(os.path.join(working_dir, 'sobelv.npy'), dtype=np.float, mode='r')
#sobelh = np.memmap(os.path.join(working_dir, 'sobelh.npy'), dtype=np.float, mode='r')
sobelv = np.load(os.path.join(working_dir, 'sobelv.npy'), mmap_mode='r')
sobelh = np.load(os.path.join(working_dir, 'sobelh.npy'), mmap_mode='r')

d_preview = imread(os.path.join(working_dir, 'sobelv.png'))
print('d_preview', d_preview.shape, d_preview.dtype)

def index_to_yx(index, rows, cols):
    y = index // cols
    x = index % cols
    return np.array([y, x], dtype=np.float)

def round(f):
    return int(f + 0.5)

def format_time(t):
    t = int(t*1000)
    return '{}:{}:{}.{}'.format(str(t//3600000).zfill(2), str(t%3600000//60000).zfill(2), str(t%60000//1000).zfill(2), str(t%1000).zfill(3))

def snapshot_preview(background, trace_map, preview_shape, working_dir):
    # Save for preview
    trace_ = normalize(imresize(trace_map, preview_shape))
    rgb_map = np.ndarray(trace_.shape + (3,), dtype=np.uint8)
    print('Snapshot for preview', rgb_map.shape)
    rgb_map[:, :, 0] = (trace_*255).astype(dtype=np.uint8)
    rgb_map[:, :, 1] = background
    fn_preview = os.path.join(working_dir, 'trace.png')
    imsave(fn_preview, rgb_map)    

fn_trace = os.path.join(working_dir, 'trace.dat')
trace_map = np.memmap(fn_trace, dtype=np.float, mode='w+', shape=sobelv.shape)
height, width, *_ = sobelv.shape
length = height * width
print('dimension', height, width, length)

seed = random.randrange(length)
multiplier = 6364136223846793005
increment = 1442695040888963407
modulus = 2 ** 64

t_init = time()
t = 0
t_save = 0
iterations = length // 4
for i in range(iterations):

    if time() - t > 2:
        elapsed = time() - t_init
        ete = 'N/A'
        if i > 0: ete = format_time(elapsed * iterations / i)
        percent = i*10000//iterations/100
        print('Pixels {} of {}, {}%, elapsed: {}, ete: {}'.format(i, iterations, percent, format_time(elapsed), ete))
        t = time()

    if time() - t_save > 60:
        snapshot_preview(d_preview, trace_map, trace_map.shape, working_dir)
        t_save = time()

    p = index_to_yx(seed, height, width)
    y_ = -1
    x_ = -1
    #print('seed', seed, p)
    for s in range(128):
        y = round(p[0])
        x = round(p[1])
        if y < 0 or y >= height or x < 0 or x >= width:
            continue
        if trace_map[y, x] >= 32:
            pass
        elif y != y_ or x != x_:
            # Don't plot a pixel twice
            trace_map[y, x] += 1.
        v = np.array([sobelv[y, x], sobelh[y, x]], dtype=np.float)
        p += v
        y_ = y
        x_ = x
        #print('move', v, v[0]*v[0]+v[1]*v[1], p)
    seed = (multiplier*seed + increment) % (length + 1)