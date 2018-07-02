import numpy as np
import os
from scipy.ndimage.interpolation import zoom

BLACKLIST = ['LAUIHISOEZIM5ILF',
             '2018050121043822', 
             '2018050120260258']  


def get_transform(data_loc, dest_loc):
    filelist = sorted([f for f in os.listdir(data_loc)])
    for i, file in enumerate(filelist):
        print('{} / {}'.format(i, len(filelist)))
        blacklisted = False
        for each in BLACKLIST:
            if each in file:
                blacklisted = True

        if not blacklisted:
            img = np.load('{}/{}'.format(data_loc, file))
            img = transform_image(img)
            file_id = file.split('/')[-1]
            np.save('{}/{}'.format(dest_loc, file_id), img)


def transform_image(image):
    image = np.moveaxis(image, 0, -1)

    # Set bounds
    image[image < -40] = -40
    image[image > 400] = 400

    # Interpolate axis
    dims = np.shape(image)
    image = zoom(image, (220 / dims[0],
                         220 / dims[1],
                         1))
    return image

get_transform('data/mip_axial', 'data/mip_transform')
