import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair


class DeformRoIPoolingFunction(Function):

    pass


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    pass

class DeformRoIPoolingPack(DeformRoIPooling):

    pass


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    pass