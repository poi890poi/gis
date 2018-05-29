import numpy as np

def normalize(arr):
    print('normalize', arr.dtype)
    min_ = np.amin(arr)
    max_ = np.amax(arr)

    #hist, bin_edges = np.histogram(arr, 64)
    #print('hist', hist, bin_edges)

    arr -= min_
    arr /= (max_ - min_)
    print(min_, max_, np.amin(arr), np.amax(arr))
    return arr

def normalize_symm(arr):
    print('symmetrical normalize', arr.dtype)
    min_ = np.amin(arr)
    max_ = np.amax(arr)
    max_ = max(-min_, max_)
    min_ = -max_
    arr -= min_
    arr /= (max_ - min_)
    print(min_, max_, np.amin(arr), np.amax(arr))
    return arr