import os

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Conv3d, Linear, Module, BatchNorm3d, Sequential, MaxPool3d


class ResNetBlock(Module):
    """A block represented by one residual connection.
    """

    def __init__(self, in_channels, out_channels):
        # TODO: Include downsampling, different stride lengths
        """
        :param n_channels: (int) Number of channels in the input image
        :param out_channels: (int) Number of channels produced by the convolution
        :param downsample: A function for downsampl
        """
        super().__init__()
        self.conv1 = Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn1 = BatchNorm3d(out_channels)
        self.conv2 = Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = BatchNorm3d(out_channels)

    def forward(self, input):
        residual = input
        input = self.conv1(input)
        input = self.bn1(input)
        input = F.relu(input)
        input = self.conv2(input)
        input = self.bn2(input)
        residual = self.downsample(input)  # Makes sure that the dimensions match
        input += residual
        return F.relu(input)


class ResNet3d(Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm3d(32)
        self.maxpool = MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 32
        # 3 blocks w/ 64 output channels
        self.group1 = self._block_group(ResNetBlock, 32, 3)
        # 3 blocks w/ 128 output channels
        self.group2 = self._block_group(ResNetBlock, 64, 3)
        self.pool = F.avg_pool3d(7)
        self.fc = Linear(512, 2)

    def _block_group(self, block, out_channels, num_blocks):
        """A sequence of blocks with the same number of out_channels
        :param block: The class of the block which compose the layer
        :param out_channels: The number of output channels for each block
        :param num_blocks: The number of blocks in the layer
        :return:
        """
        # Update the in_channels parameter for this sequence
        self.in_channels = out_channels * block.expansion
        # Create our group of num_blocks layers
        layers = [block(self.in_channels, out_channels)]
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        # Convert from 3D image w/ channels into a prediction
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


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

    images = np.stack(
        scipy.ndimage.interpolation.zoom(arr, 32 / 200, mode='nearest')
        for arr in images
    )
    print(images.shape)

    model = SimpleCNN()
    for data in images:
        data = Variable(torch.from_numpy(data)).unsqueeze(0)
        # target = Variable(target)
        output = model(data)
