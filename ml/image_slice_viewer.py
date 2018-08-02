"""
===================
Image Slices Viewer
===================

Scroll through 2D image slices of a 3D array.
"""
from __future__ import print_function


class IndexTracker(object):
    def __init__(self, ax1, ax2, X, Y):
        self.ax1 = ax1
        self.ax2 = ax2
        ax1.set_title('use scroll wheel to navigate images')

        self.X = X
        self.Y = Y
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im_x = ax1.imshow(self.X[:, :, self.ind])
        self.im_y = ax2.imshow(self.Y[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im_x.set_data(self.X[:, :, self.ind])
        self.im_y.set_data(self.Y[:, :, self.ind])
        self.ax1.set_ylabel('mip %s' % self.ind)
        self.ax2.set_ylabel('pred %s' % self.ind)
        self.im_x.axes.figure.canvas.draw()
        self.im_y.axes.figure.canvas.draw()
