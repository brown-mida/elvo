"""
===================
Image Slices Viewer
===================

Allows us to through 2D image slices of a 3D array. Specifically used in
3d_viz_second_stage.py to give us a MIP/heat-map that we can look at side-by-
side. Could be useful for demos/in shipping something to doctors.
"""
from __future__ import print_function


class IndexTracker(object):
    def __init__(self, ax1, ax2, X, Y):
        """

        :param ax1: axis of the MIP
        :param ax2: axis of the heat-map
        :param X: 3D array of MIP slices
        :param Y: 3D array of heat-map slices
        """
        self.ax1 = ax1
        self.ax2 = ax2
        ax1.set_title('use scroll wheel to navigate images')
        self.X = X
        self.Y = Y
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        # Set initial image that's seen
        self.im_x = ax1.imshow(self.X[:, :, self.ind])
        self.im_y = ax2.imshow(self.Y[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        """
        Handles scrolling up and down and updating the image shown on canvas
        :param event: a scroll event
        :return:
        """
        print("%s %s" % (event.button, event.step))

        # Case scroll up
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices

        # Case scroll down
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """
        Updates the images actually shown on the pyplot canvas
        :return:
        """
        # Set image according to whatever index should be there
        self.im_x.set_data(self.X[:, :, self.ind])
        self.im_y.set_data(self.Y[:, :, self.ind])
        # Set axis labels
        self.ax1.set_ylabel('mip %s' % self.ind)
        self.ax2.set_ylabel('pred %s' % self.ind)
        # Render the image
        self.im_x.axes.figure.canvas.draw()
        self.im_y.axes.figure.canvas.draw()
