import os

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Conv3d, Linear, Module


# class ConvBlock(Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = BatchNorm3d(out_channels)
#         self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = BatchNorm3d(out_channels)
#
#     def forward(self, input):
#         residual = input
#         input = self.conv1(input)
#         input = self.bn1(input)
#         input = F.relu(input)
#         input = self.conv2(input)
#         input = self.bn2(input)
#         input += residual
#         return F.relu(input)
#
#
# class ResNet3d(Module):
#     def __init__(self):
#         super().__init__()
#         self.block1 = ConvBlock(200, 32)
#         self.block2 = ConvBlock(32, 64)
#         self.pool = F.avg_pool3d(2)
#
#     def forward(self, x):
#         pass


class SimpleCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv3d(1, 10, kernel_size=3)
        self.conv2 = Conv3d(10, 20, kernel_size=3)
        self.fc1 = Linear(64, 1)  # TODO: Figure out the input dim

    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv1(x), 2))
        x = F.relu(F.max_pool3d(self.conv2(x), 2))
        x = x.view(320, -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


if __name__ == '__main__':
    images = []
    for filename in os.listdir('data-1517851899'):
        images.append(np.load('data-1517851899' + '/' + filename))
    print('Data Loaded')

    images = np.stack(scipy.ndimage.interpolation.zoom(arr,
                                                       32 / 200,
                                                       mode='nearest')
                      for arr in images)
    print(images.shape)

    model = SimpleCNN()
    for data in images:
        data = Variable(torch.from_numpy(data)).unsqueeze(0)
        # target = Variable(target)
        output = model(data)
