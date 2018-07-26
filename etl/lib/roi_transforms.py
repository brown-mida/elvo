import numpy as np
from scipy.ndimage.interpolation import zoom


def round_base32(x, base=32):
    return int(base * round(float(x)/base))


def convert_multiple_32(arr):
    dims = np.shape(arr)
    newx = round_base32(dims[0])
    newy = round_base32(dims[1])
    newz = round_base32(dims[2])
    return zoom(arr, (newx / dims[0],
                      newy / dims[1],
                      newz / dims[2])
                )
