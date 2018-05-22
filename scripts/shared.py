import numpy as np

def normalize(arr):
    min = np.amin(arr)
    max = np.amax(arr)
    arr -= min
    arr /= (max-min)
    print(min, max, np.amin(arr), np.amax(arr))
    return arr