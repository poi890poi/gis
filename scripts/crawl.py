import numpy as np
from shared import normalize
from scipy.misc import imsave
import os.path
import random
from math import sqrt
from time import time
from skimage.exposure import equalize_hist, adjust_gamma

working_dir = '../../data'

# Load vertical and horizontal gradient maps
sobelv = np.load(os.path.join(working_dir, 'sobelv.npy'))
sobelh = np.load(os.path.join(working_dir, 'sobelh.npy'))

# Create slope map for visualization
d_map = np.sqrt(sobelv*sobelv + sobelh*sobelh)
d_preview = np.copy(d_map)
d_preview[:] = (normalize(d_preview))
fn_preview = os.path.join(working_dir, 'slope.png')
imsave(fn_preview, (d_preview*255).astype(np.uint8))

# Normalize vectors (keep direction and discard length)
d_map[d_map==0] = 1.
sobelv /= d_map
sobelh /= d_map

def index_to_yx(index, rows, cols):
    y = index // cols
    x = index % cols
    return np.array([y, x], dtype=np.float)

def round(f):
    return int(f + 0.5)

def format_time(t):
    t = int(t*1000)
    return '{}:{}:{}.{}'.format(str(t//3600000).zfill(2), str(t%3600000//60000).zfill(2), str(t%60000//1000).zfill(2), str(t%1000).zfill(3))

fn_trace = os.path.join(working_dir, 'trace.dat')
trace_map = np.memmap(fn_trace, dtype=np.float, mode='w+', shape=d_map.shape)
height, width, *_ = d_map.shape
length = height * width
print('dimension', height, width, length)

seed = random.randrange(length)
multiplier = 6364136223846793005
increment = 1442695040888963407
modulus = 2 ** 64

t_init = time()
t = 0
iterations = length // 4
for i in range(iterations):
    if time() - t > 2:
        elapsed = time() - t_init
        ete = 'N/A'
        if i > 0: ete = format_time(elapsed * iterations / i)
        print('Pixels {} of {}, {}%, elapsed: {}, ete: {}'.format(i, iterations, i*10000//iterations/100, format_time(elapsed), ete))
        t = time()
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

# Save for preview
trace_map[:] = normalize(trace_map)
rgb_map = np.ndarray(trace_map.shape + (3,), dtype=np.uint8)
print(rgb_map.shape)
rgb_map[:, :, 0] = (trace_map*255).astype(dtype=np.uint8)
rgb_map[:, :, 1] = (d_preview*255).astype(dtype=np.uint8)
fn_preview = os.path.join(working_dir, 'trace.png')
imsave(fn_preview, rgb_map)