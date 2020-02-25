import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair


class DeformConvFunction(Function):

    pass

class ModulatedDeformConvFunction(Function):

    pass

deform_conv = DeformConvFunction.apply
modulated_deform_conv = ModulatedDeformConvFunction.apply


class DeformConv(nn.Module):

    pass

class ModulatedDeformConv(nn.Module):

    pass

class DeformConvPack(DeformConv):

    pass

class ModulatedDeformConvPack(ModulatedDeformConv):

    pass