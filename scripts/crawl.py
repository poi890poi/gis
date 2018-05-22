import numpy as np
from shared import normalize
from scipy.misc import imsave
import os.path

working_dir = '../../data'

sobelv = np.load(os.path.join(working_dir, 'sobelv.npy'))
sobelh = np.load(os.path.join(working_dir, 'sobelh.npy'))

print('norm', sobelv.shape, np.amin(sobelv), np.amax(sobelv), np.amin(sobelh), np.amax(sobelh))

sobelv = (sobelv-0.5)*2
sobelh = (sobelh-0.5)*2

print('[-1, 1]', sobelv.shape, np.amin(sobelv), np.amax(sobelv), np.amin(sobelh), np.amax(sobelh))

slope_map = sobelv * sobelv + sobelh * sobelh
print('sqr dist', np.amin(slope_map), np.amax(slope_map))
normalize(slope_map)
print('norm', np.amin(slope_map), np.amax(slope_map))

fn_preview = os.path.join(working_dir, 'slope.png')
imsave(fn_preview, (slope_map*255).astype(np.uint8))

print('preview saved', slope_map.shape, fn_preview)
