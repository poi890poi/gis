import gdal
import numpy as np
from scipy import signal
from skimage.transform import resize as imresize
from scipy.misc import imsave
from shared import normalize

import sys
import os.path
from shutil import copyfile
from tempfile import mkdtemp

class ConvKernels:
    identity = np.array([[ 0.0, 0.0, 0.0],
                         [ 0.0, 1.0, 0.0],
                         [ 0.0, 0.0, 0.0]])
    gaussian = np.array([[ 1.0, 2.0, 1.0],
                         [ 2.0, 4.0, 2.0],
                         [ 1.0, 2.0, 1.0]])
    prom = np.array([[ -1.0, -2.0, -1.0],
                     [ -2.0, 12.0, -2.0],
                     [ -1.0, -2.0, -1.0]])
    prom5 = np.array([[ -1.0, -4.0, -6.0, -4.0, -1.0],
                      [ -4.0,-16.0,-24.0,-16.0, -4.0],
                      [ -6.0,-24.0,220.0,-24.0, -6.0],
                      [ -4.0,-16.0,-24.0,-16.0, -4.0],
                      [ -1.0, -4.0, -6.0, -4.0, -1.0]])
    sobelh = np.array([[  1.0,  0.0, -1.0],
                       [  2.0,  0.0, -2.0],
                       [  1.0,  0.0, -1.0]])
    sobelv = np.array([[  1.0,  2.0,  1.0],
                       [  0.0,  0.0,  0.0],
                       [ -1.0, -2.0, -1.0]])  
    sobelh5 = np.array([[  1.0,  2.0,  0.0, -2.0, -1.0],
                        [  4.0,  8.0,  0.0, -8.0, -4.0],
                        [  6.0, 12.0,  0.0,-12.0, -6.0],
                        [  4.0,  8.0,  0.0, -8.0, -4.0],
                        [  1.0,  2.0,  0.0, -2.0, -1.0]])
    sobelv5 = np.array([[  1.0,  4.0,  6.0,  4.0,  1.0],
                        [  2.0,  8.0, 12.0,  8.0,  2.0],
                        [  0.0,  0.0,  0.0,  0.0,  0.0],
                        [ -2.0, -8.0,-12.0, -8.0, -2.0],
                        [ -1.0, -4.0, -6.0, -4.0, -1.0]])

def save_preview(path, source_array, target_shape=[4000, 4000]):
    height, width, *_ = source_array.shape

    resize_rate = min(target_shape[0]/height, target_shape[1]/width)
    target_shape = (np.array([height, width], dtype=np.float)*resize_rate).astype(dtype=np.int)

    resized = imresize(source_array, (target_shape[0], target_shape[1]))
    np.save('', resized)

    # Save preview PNG
    preview = (normalize(resized)*255.).astype(np.uint8)
    imsave(path, preview)
    print('Preview saved', path, target_shape)

def process_save(source_array, kernel, source, prefix, band):
    try:
        fn_full = prefix + '.dat'
        fn_small = prefix + ''
        fn_preview = prefix + '.png'

        # Calculate fixed-aspect resizing for preview
        height, width, *_ = source_array.shape
        resize_target = [4000, 4000]
        resize_rate = min(resize_target[0]/height, resize_target[1]/width)
        resize_target = (np.array([height, width], dtype=np.float)*resize_rate).astype(dtype=np.int)

        print('Convolving...', height, width)

        # Convolve
        convolved = np.memmap(fn_full, dtype=np.float, mode='w+', shape=source_array.shape)
        kernel_combined = signal.convolve2d(kernel, ConvKernels.gaussian, mode='full')
        convolved[:] = signal.convolve2d(source_array, kernel_combined, mode='same')
        convolved.flush()
        print('Convolved data saved', fn_full, sys.getsizeof(convolved))
        
        # Save resized array for proof of concept
        # Do not normalize here; vector length normalization should be applied instead
        resized = imresize(convolved, (resize_target[0], resize_target[1]))
        np.save(fn_small, resized)

        # Save preview PNG
        preview = (normalize(resized)*255.).astype(np.uint8)
        print(np.amin(resized), np.amax(resized), np.amin(preview), np.amax(preview))
        imsave(fn_preview, preview)
        print('Preview saved', fn_preview, resize_target)

    except:
        raise

working_dir = '../../data'
source = os.path.join(working_dir, './tif file/twdtm_asterV2_30m.tif')
print(source)
padding = 5

dem = gdal.Open(source)
xsize = dem.RasterXSize
ysize = dem.RasterYSize
bands = dem.RasterCount

for i in range(1, bands + 1):
    print()
    print('Processing band {} of {}'.format(i, bands))
    band_i = dem.GetRasterBand(i)

    raster = np.memmap(os.path.join(mkdtemp(), 'working_dem.dat'), dtype=np.float, mode='w+', shape=(ysize, xsize))
    raster[:] = band_i.ReadAsArray()

    print('Raster loaded', ysize, xsize, sys.getsizeof(raster))

    # Kernel size of 3x3 is choosen because in 30m DEM it spans 90m and is already too rough for hiking
    process_save(raster, ConvKernels.prom, source, os.path.join(working_dir, 'prom'), i)
    #process_save(raster, ConvKernels.prom5, source, os.path.join(working_dir, 'prom5'), i)
    process_save(raster, ConvKernels.sobelv, source, os.path.join(working_dir, 'sobelv'), i)
    process_save(raster, ConvKernels.sobelh, source, os.path.join(working_dir, 'sobelh'), i)
    #process_save(raster, ConvKernels.sobelv5, source, os.path.join(working_dir, 'sobelv5'), i)
    #process_save(raster, ConvKernels.sobelh5, source, os.path.join(working_dir, 'sobelh5'), i)

dem = None

# Prepare vector length map and normalize sobel
sobelv = np.load(os.path.join(working_dir, 'sobelv.npy'), mmap_mode='r+')
sobelh = np.load(os.path.join(working_dir, 'sobelh.npy'), mmap_mode='r+')

d_map = np.memmap(os.path.join(working_dir, 'vlen.npy'), dtype=np.float, mode='w+', shape=sobelv.shape)
d_map[:] = np.sqrt(sobelv*sobelv + sobelh*sobelh)

# Save vector length as preview PNG
d_preview = imresize(d_map, d_map.shape)
d_preview[:] = normalize(d_preview)
fn_preview = os.path.join(working_dir, 'vlen.png')
imsave(fn_preview, (d_preview*255).astype(np.uint8))

# Normalize vectors (keep direction and discard length)
d_map[d_map==0] = 1.
sobelv /= d_map
sobelh /= d_map
sobelv.flush()
sobelh.flush()

print()