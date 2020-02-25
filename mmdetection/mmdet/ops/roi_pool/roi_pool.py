import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from torchvision.ops import roi_pool


class RoIPool(nn.Module):

    def __init__(self, out_size, spatial_scale, use_torchvision=True):
        super(RoIPool, self).__init__()

        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.use_torchvision = True

    def forward(self, features, rois):
            return roi_pool(features, rois, self.out_size,
                            self.spatial_scale)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}'.format(
            self.out_size, self.spatial_scale)
        format_str += ', use_torchvision={})'.format(self.use_torchvision)
        return format_str