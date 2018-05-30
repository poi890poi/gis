import numpy as np

def normalize(arr, signed=False):
    #hist, bin_edges = np.histogram(arr, 16)
    #print('Normalization...', signed, arr.dtype, arr.shape, hist, bin_edges)
    min_ = np.amin(arr)
    max_ = np.amax(arr)

    if signed:
        max_ = max(-min_, max_)
        min_ = -max_

    print('Normalization... signed: {}, dtype: {}, shape: {}, min/max: ({}, {})'.format(signed, arr.dtype, arr.shape, min_, max_))
        
    arr -= min_
    arr /= (max_ - min_)
    return arr