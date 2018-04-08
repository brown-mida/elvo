import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter

a = np.random.rand(200, 200, 24)
a = np.expand_dims(a, -1)
print(np.shape(a))
