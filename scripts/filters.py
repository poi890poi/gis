import gdal, osr
import numpy as np
import tensorflow as tf
from scipy import signal
from skimage.transform import resize as imresize
from scipy.misc import imsave
from shared import normalize, normalize_symm
from skimage import exposure

import sys
import os.path
from shutil import copyfile
from tempfile import mkdtemp, gettempdir

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

def save_preview(path, source_array, transform=None, target_shape=[4000, 4000]):
    if transform is not None:
        lt = latlon_to_pixel(transform, (121.46883, 24.88643))
        rb = latlon_to_pixel(transform, (121.68856, 24.68991))
        print('pixels', transform, lt, rb)
        roi = [min(lt[0], rb[0]), min(lt[1], rb[1]), max(lt[0], rb[0]), max(lt[1], rb[1])]
        print(roi)
        width = roi[2] - roi[0]        
        height = roi[3] - roi[1]        

        resize = np.array(source_array[roi[1]:roi[1]+height, roi[0]:roi[0]+width])

        # Save preview PNG
        preview = (normalize(resize)*255.).astype(np.uint8)
        imsave(path, exposure.rescale_intensity(preview))
        print('Preview saved', path, target_shape)

    else:
        # Calculate fixed-aspect resizing for preview
        height, width, *_ = source_array.shape

        resize_rate = min(target_shape[0]/height, target_shape[1]/width)
        target_shape = (np.array([height, width], dtype=np.float)*resize_rate).astype(dtype=np.int)

        resize = np.memmap(os.path.join(gettempdir(), 'resizing.dat'), dtype=source_array.dtype, mode='w+', shape=source_array.shape)
        resize[:] = source_array
        resize = imresize(resize, target_shape)

        # Save preview PNG
        preview = (normalize(resize)*255.).astype(np.uint8)
        imsave(path, exposure.rescale_intensity(preview))
        print('Preview saved', path, target_shape)

def process_save(source_array, kernel, source, prefix, band, transform):
    try:
        fn_full = prefix + '.dat'
        fn_small = prefix + ''

        # Convolve
        tf_shape = (1,) + source_array.shape + (1,)
        print('Convolving...', source_array.shape, tf_shape)
        convolved = np.memmap(fn_full, dtype=np.float, mode='w+', shape=source_array.shape)
        kernel_combined = signal.convolve2d(kernel, ConvKernels.gaussian, mode='full')
        convolved[:] = signal.convolve2d(source_array, kernel_combined, mode='same')
        convolved.flush()
        print('Convolved data saved', fn_full, sys.getsizeof(convolved))

        fn_preview = prefix + '.png'
        save_preview(fn_preview, convolved, transform)

    except:
        raise

print(gettempdir())

working_dir = '../../data'
source = os.path.join(working_dir, './tif file/twdtm_asterV2_30m.tif')
print(source)
padding = 5

dem = gdal.Open(source)
xsize = dem.RasterXSize
ysize = dem.RasterYSize
bands = dem.RasterCount
print('DEM opened', xsize, ysize, bands)

def pixel_to_latlon(transform, pixel):
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    return (x_origin + pixel[0]*pixel_width, y_origin + pixel[1]*pixel_height)

def latlon_to_pixel(transform, latlon):
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]
    return (round((latlon[0]-x_origin)/pixel_width), round((latlon[1]-y_origin)/pixel_height))

projection = dem.GetProjection()
print('projection', projection)
transform = dem.GetGeoTransform()
lt = pixel_to_latlon(transform, (0, 0))
rb = pixel_to_latlon(transform, (xsize, ysize))
print('latlon', transform, lt, rb)
lt = latlon_to_pixel(transform, lt)
rb = latlon_to_pixel(transform, rb)
print('pixels', transform, lt, rb)

for i in range(1, bands + 1):
    print()
    print('Processing band {} of {}'.format(i, bands))
    band_i = dem.GetRasterBand(i)

    raster = np.memmap(os.path.join(working_dir, 'dem.dat'), dtype=np.float, mode='w+', shape=(ysize, xsize))
    raster[:] = band_i.ReadAsArray()

    fn_preview = os.path.join(working_dir, 'dem.png')
    save_preview(fn_preview, raster, dem.GetGeoTransform())

    print('Raster loaded', ysize, xsize, sys.getsizeof(raster))

    # Kernel size of 3x3 is choosen because in 30m DEM it spans 90m and is already too rough for hiking
    process_save(raster, ConvKernels.prom, source, os.path.join(working_dir, 'prom'), i, dem.GetGeoTransform())
    #process_save(raster, ConvKernels.prom5, source, os.path.join(working_dir, 'prom5'), i)
    process_save(raster, ConvKernels.sobelv, source, os.path.join(working_dir, 'sobelv'), i, dem.GetGeoTransform())
    process_save(raster, ConvKernels.sobelh, source, os.path.join(working_dir, 'sobelh'), i, dem.GetGeoTransform())
    #process_save(raster, ConvKernels.sobelv5, source, os.path.join(working_dir, 'sobelv5'), i)
    #process_save(raster, ConvKernels.sobelh5, source, os.path.join(working_dir, 'sobelh5'), i)

# Prepare vector length map and normalize sobel
#sobelv = np.memmap(fn_full, dtype=np.float, mode='w+', shape=(1,)+source_array.shape+(1,))
sobelv = np.memmap(os.path.join(working_dir, 'sobelv.dat'), dtype=np.float, mode='r+', shape=(ysize, xsize))
sobelh = np.memmap(os.path.join(working_dir, 'sobelh.dat'), dtype=np.float, mode='r+', shape=(ysize, xsize))

d_map = np.memmap(os.path.join(working_dir, 'vlen.dat'), dtype=np.float, mode='w+', shape=sobelv.shape)
d_map[:] = np.sqrt(sobelv*sobelv + sobelh*sobelh)

# Save vector length as preview PNG
fn_preview = os.path.join(working_dir, 'vlen.png')
save_preview(fn_preview, d_map, dem.GetGeoTransform())

# Normalize vectors (keep direction and discard length)
d_map[d_map==0] = 1.
sobelv /= d_map
sobelh /= d_map
sobelv.flush()
sobelh.flush()

fn_preview = os.path.join(working_dir, 'sobelv-n.png')
save_preview(fn_preview, sobelv, dem.GetGeoTransform())
fn_preview = os.path.join(working_dir, 'sobelh-n.png')
save_preview(fn_preview, sobelh, dem.GetGeoTransform())

dem = None

print()