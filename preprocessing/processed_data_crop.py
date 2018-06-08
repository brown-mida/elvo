# import inline
# import matplotlib
#
# # % matplotlib inline
#
# import numpy as np  # linear algebra
# import os
# import matplotlib.pyplot as plt
#
# import measure
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#
# INPUT_FOLDER = '/Users/Sumera/Desktop/data-1521428185'
#
#
# def plot_3d(image, threshold=-300):
#     # Position the scan upright,
#     # so the head of the patient would be at the top facing the camera
#     p = image.transpose(2, 1, 0)
#
#     verts, faces = measure.marching_cubes_classic(p, threshold)
#
#     # Fancy indexing: `verts[faces]` to generate a collection of triangles
#     mesh = Poly3DCollection(verts[faces])
#     face_color = [0.45, 0.45, 0.75]
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)
#
#     ax.set_xlim(0, p.shape[0])
#     ax.set_ylim(0, p.shape[1])
#     ax.set_zlim(0, p.shape[2])
#
#     plt.show()
#
#
# patients = os.listdir(INPUT_FOLDER)
# patient1 = np.load(INPUT_FOLDER + patients[2])
#
#
# def crop_center(img, cropx, cropy, cropz):  # cropping center of image!
#     x, y, z = img.shape
#     startx = x // 2 - (cropx // 2)
#     starty = y // 2 - (cropy // 2)
#     startz = z // 2 - (cropz // 2)
#     return img[startx:startx + cropx, starty:starty + cropy,
#            startz:startz + cropz]
#
#
# image_cropped = crop_center(patient1, 150, 150, 64)
# print(image_cropped)
#
#
# def plot_3d(image, ax):
#     # Position the scan upright,
#     # so the head of the patient would be at the top facing the camera
#     p = image.transpose(2, 1, 0)
#
#     # p = image
#
#     # used marching_cubes_classic bc lewiner didn't work for some reason :/
#     verts, faces = measure.marching_cubes_classic(p,
#                                                   40)
#
#     # Fancy indexing: `verts[faces]` to generate a collection of triangles
#     mesh = Poly3DCollection(verts[faces], alpha=0.70)
#     face_color = [0.45, 0.45, 0.75]
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)
#
#     ax.set_xlim(0, p.shape[0])
#     ax.set_ylim(0, p.shape[1])
#     ax.set_zlim(0, p.shape[2])
#
#     plt.show()
#
#
# fig = plt.figure(figsize=(10, 10))
# axis = fig.add_subplot(111, projection='3d')
# plot_3d(patient1, axis)
# plot_3d(image_cropped, axis)
