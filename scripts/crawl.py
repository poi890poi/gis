import numpy as np
from shared import normalize
from scipy.misc import imsave
import os.path
import random
from math import sqrt
from time import time

working_dir = '../../data'

# Load vertical and horizontal gradient maps
sobelv = np.load(os.path.join(working_dir, 'sobelv.npy'))
sobelh = np.load(os.path.join(working_dir, 'sobelh.npy'))

# Create slope map for visualization
d_map = np.sqrt(sobelv*sobelv + sobelh*sobelh)
d_preview = np.copy(d_map)
d_preview[:] = normalize(d_preview)
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

fn_trace = os.path.join(working_dir, 'trace.dat')
trace_map = np.memmap(fn_trace, dtype=np.float, mode='w+', shape=d_map.shape)
height, width, *_ = d_map.shape
length = height * width
print('dimension', height, width, length)

seed = random.randrange(length)
multiplier = 6364136223846793005
increment = 1442695040888963407
modulus = 2**64

t = 0
iterations = length//8
for i in range(iterations):
    if time() - t > 2:
        print('Pixels {} of {}, {}%'.format(i, iterations, i*100//iterations))
        t = time()
    p = index_to_yx(seed, height, width)
    #print('seed', seed, p)
    for s in range(64):
        y = round(p[0])
        x = round(p[1])
        if y < 0 or y >= height or x < 0 or x >= width:
            continue
        v = np.array([sobelv[y, x], sobelh[y, x]], dtype=np.float)
        trace_map[y, x] += 1.
        p += v
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