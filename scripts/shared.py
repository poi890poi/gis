import numpy as np

def normalize(arr):
    print('normalize', arr.dtype)
    min = np.amin(arr)
    max = np.amax(arr)

    #hist, bin_edges = np.histogram(arr, 64)
    #print('hist', hist, bin_edges)

    arr -= min
    arr /= (max - min)
    print(min, max, np.amin(arr), np.amax(arr))
    return arr

def normalize_symm(arr):
    print('symmetrical normalize', arr.dtype)
    min = np.amin(arr)
    max = np.amax(arr)
    max = max(-min, max)
    min = -max
    arr -= min
    arr /= (max - min)
    print(min, max, np.amin(arr), np.amax(arr))
    return arr