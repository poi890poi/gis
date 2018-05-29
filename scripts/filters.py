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

def convert_dtype(source_array, outpath, dtype):
    cast_ = np.memmap(outpath, dtype=dtype, mode='w+', shape=source_array.shape)
    cast_[:] = source_array.astype(dtype=dtype)
    cast_.flush()
    return cast_

def save_preview(path, source_array, transform=None, target_shape=[4000, 4000]):
    if transform is not None:
        lt = latlon_to_pixel(transform, (121., 25.))
        rb = latlon_to_pixel(transform, (122., 24.))
        print('pixels', transform, lt, rb)
        roi = [min(lt[0], rb[0]), min(lt[1], rb[1]), max(lt[0], rb[0]), max(lt[1], rb[1])]
        print(roi)
        width = roi[2] - roi[0]        
        height = roi[3] - roi[1]        

        resize = np.array(source_array[roi[1]:roi[1]+height, roi[0]:roi[0]+width]).astype(dtype=np.float)

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

        del resize

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

        del convolved

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

print()
# Process only first band
i = 1
print('Processing band {} of {}'.format(i, bands))
band_i = dem.GetRasterBand(i)

raster = np.memmap(os.path.join(working_dir, 'dem-f64.dat'), dtype=np.float, mode='w+', shape=(ysize, xsize))
raster[:] = band_i.ReadAsArray()
print('Raster loaded', ysize, xsize, sys.getsizeof(raster))

process_save(raster, ConvKernels.prom5, source, os.path.join(working_dir, 'prom-f64'), i, dem.GetGeoTransform())
process_save(raster, ConvKernels.sobelv5, source, os.path.join(working_dir, 'sobelv-f64'), i, dem.GetGeoTransform())
process_save(raster, ConvKernels.sobelh5, source, os.path.join(working_dir, 'sobelh-f64'), i, dem.GetGeoTransform())

# Normalize elevation, convert to int8, save preview
raster[:] = normalize(raster)
raster *= 255
dem_s = convert_dtype(raster, os.path.join(working_dir, 'dem.dat'), np.uint8)
fn_preview = os.path.join(working_dir, 'dem.png')
save_preview(fn_preview, dem_s, dem.GetGeoTransform())
del dem_s
del raster

# Prepare vector length map and normalize sobel
sobelv = np.memmap(os.path.join(working_dir, 'sobelv-f64.dat'), dtype=np.float, mode='r+', shape=(ysize, xsize))
sobelh = np.memmap(os.path.join(working_dir, 'sobelh-f64.dat'), dtype=np.float, mode='r+', shape=(ysize, xsize))

d_map = np.memmap(os.path.join(working_dir, 'vlen-f64.dat'), dtype=np.float, mode='w+', shape=sobelv.shape)
d_map[:] = np.sqrt(sobelv*sobelv + sobelh*sobelh)

# Normalize vectors (keep direction and discard length)
d_map[d_map==0] = 1.
sobelv /= d_map
sobelh /= d_map

# Normalize slope, convert to int8, save preview
d_map[:] = normalize(d_map)
d_map *= 255
d_map_s = convert_dtype(d_map, os.path.join(working_dir, 'vlen.dat'), np.uint8)
fn_preview = os.path.join(working_dir, 'vlen.png')
save_preview(fn_preview, d_map_s, dem.GetGeoTransform())
del d_map
del d_map_s

# Convert to int8
sobelv *= 127
sobelv_s = convert_dtype(sobelv, os.path.join(working_dir, 'sobelv.dat'), np.int8)
sobelh *= 127
sobelh_s = convert_dtype(sobelh, os.path.join(working_dir, 'sobelh.dat'), np.int8)

# Save for preview
fn_preview = os.path.join(working_dir, 'sobelv.png')
save_preview(fn_preview, sobelv_s, dem.GetGeoTransform())
fn_preview = os.path.join(working_dir, 'sobelh.png')
save_preview(fn_preview, sobelh_s, dem.GetGeoTransform())
del sobelv
del sobelh
del sobelv_s
del sobelh_s

# Normalize prominence, convert to int8, save preview
prom = np.memmap(os.path.join(working_dir, 'prom-f64.dat'), dtype=np.float, mode='r+', shape=(ysize, xsize))
prom[:] = normalize_symm(prom)
prom *= 127
prom_s = convert_dtype(prom, os.path.join(working_dir, 'prom.dat'), np.int8)
fn_preview = os.path.join(working_dir, 'prom.png')
save_preview(fn_preview, prom_s, dem.GetGeoTransform())
del prom
del prom_s

dem = None

print()