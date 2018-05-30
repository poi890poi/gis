import gdal, osr
import numpy as np
from scipy import signal
from scipy.misc import imsave
from skimage.transform import resize as imresize
from skimage import exposure

from shared import normalize

import sys
import os.path as path
import os
from os import makedirs
import uuid
from shutil import copyfile
import urllib.request
import wget, patoolib
import pickle

class Config:
    TYPE_FLOAT = np.float32
    BASE_DIR = '../../data'
    WORKING_DIR = BASE_DIR + '/dat'
    TEMP_DIR = BASE_DIR + '/tmp'
    PREVIEW_DIR = BASE_DIR + '/preview'

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

class DiskMap(np.memmap):
    def __new__(cls, filename, dtype, mode, shape):
        return np.memmap.__new__(cls, filename=filename, dtype=dtype, mode=mode, shape=shape)

    def __init__(self, filename, dtype, mode, shape):
        self.path = filename
        self.dir, self.filename = path.split(filename)
        print('numpy.memmap opened at {}'.format(filename))

    def unmap(self):
        self._mmap.close()
        return self.path

    @staticmethod
    def remove_file(dmap_obj):
        dpath = dmap_obj.unmap()
        del dmap_obj
        print('removing', dpath)
        os.remove(dpath)
        print('numpy.memmap {} removed from disk'.format(dpath))

    def get_writable_temp(self):
        tmppath = path.join(Config.TEMP_DIR, str(uuid.uuid4()))
        copy = DiskMap(tmppath, dtype=self.dtype, mode='w+', shape=self.shape)
        copy[:] = self[:]
        return copy

    def copy_normalize(self, signed=False):
        tmppath = path.join(Config.TEMP_DIR, str(uuid.uuid4()))
        copy = DiskMap(tmppath, dtype=Config.TYPE_FLOAT, mode='w+', shape=self.shape)
        copy[:] = self[:]
        normalize(copy, signed)
        return copy

    def copy_astype(self, dtype, outpath=''):
        if outpath=='':
            outpath = path.join(Config.TEMP_DIR, str(uuid.uuid4()))
        copy = DiskMap(outpath, dtype=dtype, mode='w+', shape=self.shape)
        copy[:] = self.astype(dtype=dtype)[:]
        return copy

    def slice_normalize(self, roi):
        roi = [min(roi[0], roi[2]), min(roi[1], roi[3]), max(roi[0], roi[2]), max(roi[1], roi[3])]
        w_ = roi[2] - roi[0]
        h_ = roi[3] - roi[1]
        sliced = np.array(self[roi[1]:roi[1]+h_, roi[0]:roi[0]+w_]).astype(dtype=Config.TYPE_FLOAT)
        return (normalize(sliced)*255.).astype(np.uint8)

    def slice_save_preview(self, outpath, roi):
        normalized = self.slice_normalize(roi)
        imsave(outpath, exposure.rescale_intensity(normalized))
        del normalized
        print('Sliced preview saved', outpath, roi)

    def resize_save_preview(self, outpath, target_shape=[4000, 4000]):
        h_, w_, *_ = self.shape
        rrate_ = min(target_shape[0]/h_, target_shape[1]/w_)
        target_shape = (np.array([h_, w_], dtype=Config.TYPE_FLOAT)*rrate_).astype(dtype=np.int)

        tmppath = path.join(Config.TEMP_DIR, str(uuid.uuid4()))
        resized = DiskMap(tmppath, dtype=self.dtype, mode='w+', shape=self.shape)
        resized[:] = self[:]
        resized = imresize(resized, target_shape)
        normalized = (normalize(resized)*255.).astype(np.uint8)
        imsave(outpath, exposure.rescale_intensity(normalized))
        del normalized
        print('Resized preview saved', outpath, target_shape)

        DiskMap.remove_file(resized)

    def save_8bit(self, outpath, signed=False, pre_normalized=False):
        if pre_normalized:
            if signed:
                cast = self.copy_astype(np.int8, outpath)
            else:
                cast = self.copy_astype(np.uint8, outpath)
        else:
            normalized_ = self.copy_normalize(signed)
            if signed:
                normalized_ *= 127
                cast = normalized_.copy_astype(np.int8, outpath)
            else:
                normalized_ *= 255
                cast = normalized_.copy_astype(np.uint8, outpath)
            DiskMap.remove_file(normalized_)
        return cast

class DigitalElevationModel:
    def __init__(self, inpath):
        print('Opening DEM object...', inpath)
        self.dir, self.filename = path.split(inpath)
        root, ext = path.splitext(self.filename)
        self.raster_path = path.join(self.dir, root + '.npy')
        self.gdal_obj = gdal.Open(inpath)
        self.xsize = self.gdal_obj.RasterXSize
        self.ysize = self.gdal_obj.RasterYSize
        self.shape = (self.ysize, self.xsize)
        self.bands = self.gdal_obj.RasterCount
        self.projection = self.gdal_obj.GetProjection()
        self.transform = self.gdal_obj.GetGeoTransform()
        self.raster = None
        print('DEM object opened, ({}, {}, {}), transformation: {}'.format(self.xsize, self.ysize, self.bands, self.transform))
        if self.bands > 1: print('The DEM has more than 1 band and this script will process only the first one')

    def pixel_to_latlon(self, pixel):
        x_ = self.transform[0]
        y_ = self.transform[3]
        pixel_width = self.transform[1]
        pixel_height = self.transform[5]
        return (x_ + pixel[0]*pixel_width, y_ + pixel[1]*pixel_height)

    def latlon_to_pixel(self, latlon):
        x_ = self.transform[0]
        y_ = self.transform[3]
        pixel_width = self.transform[1]
        pixel_height = self.transform[5]
        return (round((latlon[0]-x_)/pixel_width), round((latlon[1]-y_)/pixel_height))

    def get_raster_create(self):
        if self.raster is None:
            if path.isfile(self.raster_path):
                self.raster = DiskMap(self.raster_path, dtype=Config.TYPE_FLOAT, mode='r', shape=self.shape)
            else:
                self.raster = DiskMap(self.raster_path, dtype=Config.TYPE_FLOAT, mode='w+', shape=self.shape)
                self.raster[:] = self.gdal_obj.GetRasterBand(1).ReadAsArray()
        print('Raster loaded', np.amin(self.raster), np.amax(self.raster))
        return self.raster

    def set_preview_roi(self, lt, rb):
        lt = self.latlon_to_pixel((121., 25.))
        rb = self.latlon_to_pixel((122., 24.))
        self.preview_roi = [lt[0], lt[1], rb[0], rb[1]]

    @staticmethod
    def dump_meta(dem):
        # Dump as pickle to for future use. Call this only when raster is no longer used.
        del dem.gdal_obj
        del dem.raster
        with open(path.join(Config.BASE_DIR, 'dem.pkl'), 'wb') as fp:
            pickle.dump(dem, fp)

class MapsCreator:
    def __init__(self, source_url):
        if not path.isdir(Config.BASE_DIR): makedirs(Config.BASE_DIR)
        if not path.isdir(Config.WORKING_DIR): makedirs(Config.WORKING_DIR)
        if not path.isdir(Config.TEMP_DIR): makedirs(Config.TEMP_DIR)
        if not path.isdir(Config.PREVIEW_DIR): makedirs(Config.PREVIEW_DIR)

        self.source_dem = path.join(Config.BASE_DIR, './tif file/twdtm_asterV2_30m.tif')
        # Donwload from source URL if DEM is not found
        if not path.isfile(self.source_dem):
            print('Download and extract DEM...', source_url)
            zpath = wget.download(source_url, out=Config.BASE_DIR)
            patoolib.extract_archive(zpath, outdir=Config.BASE_DIR)

        self.dem = DigitalElevationModel(self.source_dem)
        self.raster = self.dem.get_raster_create()

    def copy_convolve(self, kernel, prefix, gaussian=True):
        # Convolve
        outpath = path.join(Config.WORKING_DIR, prefix+'.npy')
        print('Convolving...', self.raster.shape)
        convolved = DiskMap(outpath, dtype=Config.TYPE_FLOAT, mode='w+', shape=self.raster.shape)

        kernel_combined = kernel
        if gaussian:
            kernel_combined = signal.convolve2d(kernel, ConvKernels.gaussian, mode='full')

        convolved[:] = signal.convolve2d(self.raster, kernel_combined, mode='same')
        print('Convolution done')

        return convolved

    def save_8bit_n_preview(self, source, prefix, signed=False, pre_normalized=False):
        npypath = path.join(Config.WORKING_DIR, prefix + '.npy')
        cast_ = source.save_8bit(npypath, signed, pre_normalized)
        cast_.slice_save_preview(path.join(Config.PREVIEW_DIR, prefix + '.png'), self.dem.preview_roi)
        del cast_

    def prepare_maps(self):
        # Normalize elevation, convert to uint8, save preview
        self.save_8bit_n_preview(self.raster, 'dem8')

        # Prepare prominence maps
        prominence = self.copy_convolve(ConvKernels.prom5, 'prominence')
        self.save_8bit_n_preview(prominence, 'prominence8', signed=True)
        del prominence

        # Prepare sobel gradient maps
        sobelv = self.copy_convolve(ConvKernels.sobelv5, 'sobelv')
        sobelh = self.copy_convolve(ConvKernels.sobelh5, 'sobelh')

        # Use vertical and horizontal gradient maps to create slope map (vector length of sobel gradients)
        sobelv_ = sobelv.get_writable_temp()
        sobelh_ = sobelh.get_writable_temp()
        print('sobelv loaded', np.amin(sobelv_), np.amax(sobelh_))
        print('sobelh loaded', np.amin(sobelh_), np.amax(sobelh_))
        sobelv_ *= sobelv_
        sobelh_ *= sobelh_
        print('sobelv squared', np.amin(sobelv_), np.amax(sobelh_))
        print('sobelh squared', np.amin(sobelh_), np.amax(sobelh_))
        vlen = DiskMap(path.join(Config.WORKING_DIR, 'vlen.npy'), dtype=Config.TYPE_FLOAT, mode='w+', shape=sobelv.shape)
        vlen[:] = sobelv_[:]
        vlen += sobelh_
        print('sobel added', np.amin(vlen), np.amax(vlen))
        np.sqrt(vlen, vlen)
        DiskMap.remove_file(sobelv_)
        DiskMap.remove_file(sobelh_)

        # Normalize vectors (keep only gradient direction and discard length information)
        vlen[vlen==0] = 1.
        sobelv /= vlen
        sobelh /= vlen
        sobelv *= 127
        sobelh *= 127

        self.save_8bit_n_preview(vlen, 'vlen8')
        self.save_8bit_n_preview(sobelv, 'sobelv8', signed=True, pre_normalized=True)
        self.save_8bit_n_preview(sobelh, 'sobelh8', signed=True, pre_normalized=True)

        del vlen
        del sobelv
        del sobelh

if __name__ == "__main__":
    print()
    print('Load DEM and preparing maps...')

    source = 'http://gis.rchss.sinica.edu.tw/gps/wp-content/uploads/2012/02/20120201_twdtm_asterV2_30m.rar'
    mc = MapsCreator(source)
    mc.dem.set_preview_roi((121., 25.), (122., 24.))
    mc.prepare_maps()
    DigitalElevationModel.dump_meta(mc.dem)
    del mc

    print('All tasks done')
    print()