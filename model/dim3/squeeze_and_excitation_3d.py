"""
3D Squeeze and Excitation Modules
*****************************
3D Extensions of the following 2D squeeze and excitation blocks:
    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
New Project & Excite block, designed specifically for 3D inputs
    'quote'
    Coded by -- Anne-Marie Rickmann (https://github.com/arickm)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
import torch
from torch import nn as nn
from torch.nn import functional as F


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor

class PELayer(nn.Module):
    """
        Redesign the spatial information integration method for
        feature recalibration, based on the original
        Project & Excite Module
    """

    def __init__(self, num_channels, D, H, W, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param D, H, W: Spatial dimension of the input feature cube
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(PELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.convModule = nn.Sequential(
            nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1), \
            nn.ReLU(inplace=True), \
            nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1), \
            nn.Sigmoid())
        self.spatialdim = [D, H, W]

        self.D_squeeze = nn.Conv3d(in_channels=D, out_channels=1, kernel_size=1, stride=1)
        self.H_squeeze = nn.Conv3d(in_channels=H, out_channels=1, kernel_size=1, stride=1)
        self.W_squeeze = nn.Conv3d(in_channels=W, out_channels=1, kernel_size=1, stride=1)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor, mapping
        """
        squared_tensor = torch.pow(input_tensor, exponent=2)

        # Project:
        # Average along channels and different axes
        D, H, W = self.spatialdim[0], self.spatialdim[1], self.spatialdim[2]
        D_channel = input_tensor.permute(0, 2, 1, 3, 4)  # B, D, C, H, W
        H_channel = input_tensor.permute(0, 3, 2, 1, 4)  # B, H, D, C, W

        squeeze_tensor_1D = self.D_squeeze(D_channel)  # B, 1, C, H, W

        squeeze_tensor_W = squeeze_tensor_1D.permute(0, 3, 1, 2, 4)  # B, H, 1, C, W
        squeeze_tensor_W = self.H_squeeze(squeeze_tensor_W).permute(0, 3, 2, 1, 4)  # B, C, 1, 1, W

        squeeze_tensor_H = squeeze_tensor_1D.permute(0, 4, 1, 3, 2)  # B, W, 1, H, C
        squeeze_tensor_H = self.W_squeeze(squeeze_tensor_H).permute(0, 4, 2, 3, 1)  # B, C, 1, H, 1

        squeeze_tensor_D = self.H_squeeze(H_channel).permute(0, 4, 2, 1, 3)  # B, W, D, 1, C
        squeeze_tensor_D = self.W_squeeze(squeeze_tensor_D).permute(0, 4, 2, 3, 1)  # B, C, D, 1, 1

        final_squeeze_tensor = squeeze_tensor_W + squeeze_tensor_H + squeeze_tensor_D
        # Excitation:
        final_squeeze_tensor = self.convModule(final_squeeze_tensor)

        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        feature_mapping = torch.sum(squared_tensor, dim=1, keepdim=True)

        return output_tensor, feature_mapping

class SELayer3D(Enum):
    """
    Enum restricting the type of SE Blockes available. So that type checking can be adding when adding these blocks to
    a neural network::
        if self.se_block_type == se.SELayer3D.CSE3D.value:
            self.SELayer = se.ChannelSELayer3D(params['num_filters'])
        elif self.se_block_type == se.SELayer3D.SSE3D.value:
            self.SELayer = se.SpatialSELayer3D(params['num_filters'])
        elif self.se_block_type == se.SELayer3D.CSSE3D.value:
            self.SELayer = se.ChannelSpatialSELayer3D(params['num_filters'])
        elif self.se_block_type == se.SELayer3D.PE.value:
            self.SELayer = se.ProjectExcite(params['num_filters')
    """
    NONE = 'NONE'
    CSE3D = 'CSE3D'
    SSE3D = 'SSE3D'
    CSSE3D = 'CSSE3D'
    PE = 'PE'

if __name__ == '__main__':
    input = torch.randn(2, 16, 8, 8, 8)  # 2 batch size    16输入通道   8,8,8为一个通道的样本
    _, channel, _, _, _ = input.shape
    model = ChannelSpatialSELayer3D(channel)
    output = model(input)
    print(output.size())