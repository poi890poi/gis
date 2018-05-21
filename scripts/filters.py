import gdal
import numpy as np
from scipy import signal
from scipy.misc import imresize, imsave

import sys
import os.path
from shutil import copyfile

class ConvKernels:
    identity = np.array([[ 0.0, 0.0, 0.0],
                         [ 0.0, 1.0, 0.0],
                         [ 0.0, 0.0, 0.0]])
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

def process_save(source_array, kernel, source, destination, band):
    try:
        preview = destination + '.png'
        destination = destination + '.tif'

        height, width, *_ = source_array.shape
        resize_target = [1920, 1920]
        resize_rate = min(resize_target[0]/height, resize_target[1]/width)
        resize_target = (np.array([height, width], dtype=np.float)*resize_rate).astype(dtype=np.int)
        print(height, width, resize_target)

        # convolve
        convolved = signal.convolve2d(source_array, kernel, mode='same').astype(np.int16)
        
        imsave(preview, imresize(convolved, (resize_target[0], resize_target[1])))
        print('Convolved DEM saved for preview', preview)

        # save convolved raster
        if not os.path.isfile(destination):
            copyfile(source, destination)            

        out_raster = gdal.Open(destination, gdal.GA_Update)
        out_band = out_raster.GetRasterBand(band)
        out_band.WriteArray(convolved, 0, 0)
        out_band.FlushCache()
        out_raster = None
        print('Convolved DEM saved', destination)

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

print(xsize, ysize, bands)

for i in range(1, bands + 1):
    print("processing band " + str(i) + " of " + str(bands))
    band_i = dem.GetRasterBand(i)

    raster = band_i.ReadAsArray().astype(np.float)
    print("convolving band " + str(i) + " of " + str(bands))

    process_save(raster, ConvKernels.prom, source, os.path.join(working_dir, 'prom'), i)
    process_save(raster, ConvKernels.prom5, source, os.path.join(working_dir, 'prom5'), i)
    process_save(raster, ConvKernels.sobelh, source, os.path.join(working_dir, 'sobelh'), i)
    process_save(raster, ConvKernels.sobelv, source, os.path.join(working_dir, 'sobelv'), i)
    process_save(raster, ConvKernels.sobelh5, source, os.path.join(working_dir, 'sobelh5'), i)
    process_save(raster, ConvKernels.sobelv5, source, os.path.join(working_dir, 'sobelv5'), i)

dem = None