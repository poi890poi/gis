import tensorflow as tf
import numpy as np
import os.path as path
import sys
from time import time

ua = np.memmap('./uint.npy', dtype=np.uint16, mode='w+', shape=(1, 4000, 4000, 1))
fa = np.memmap('./float.npy', dtype=np.float16, mode='w+', shape=(1, 4000, 4000, 1))

for atype in ['uint16', 'flaot16']:
    if atype=='uint16':
        atarget = ua
    else:
        atarget = fa

    t = time()
    atarget[:] = 3
    print('Assigning {} array: {}'.format(atype, time()-t))

    t = time()
    atarget += 5
    print('Adding {} array: {}'.format(atype, time()-t))

    t = time()
    atarget *= 5
    print('Multiplying {} array: {}'.format(atype, time()-t))

    t = time()
    resized = tf.image.resize_images(atarget, (400, 400))
    print('Resizing {} array: {}'.format(atype, time()-t))

if not path.isfile('./large.npy'):
    mm = np.memmap('./large.npy', dtype=np.float, mode='w+', shape=(1, 4000, 4000, 1))
    mm[:] = 3.
    mm.flush()
    sys.exit()
    
mm = np.memmap('./large.npy', dtype=np.float, mode='r', shape=(1, 4000, 4000, 1))
print('memmap', mm.shape, mm.dtype)

resized = tf.image.resize_images(
    mm,
    (400, 400)
)
print('memmap', mm.shape, mm.dtype)
print('resized', resized.shape, mm.dtype)


