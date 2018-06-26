import glob

import numpy as np
import pymesh

modelnet_10_dir = '/home/rladbsgh/Downloads/ModelNet10/**/*.off'
modelnet_40_dir = '/home/rladbsgh/Downloads/ModelNet40/**/*.off'

files = []
files.extend(glob.glob(modelnet_10_dir, recursive=True))
# files.extend(glob.glob(modelnet_40_dir, recursive=True))
print(len(files))
print(files[0])

mesh = pymesh.load_mesh(files[0])
np.load(files[0])
