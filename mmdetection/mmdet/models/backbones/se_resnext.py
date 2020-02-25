import torch.nn as nn
import pretrainedmodels 

from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES

@BACKBONES.register_module
class SEResNeXt(nn.Module):
    """SE-ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    def __init__(self,
                 depth,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 norm_eval=True):
        super().__init__()

        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval


        model = getattr(pretrainedmodels, 'se_resnext{}_32x4d'.format(depth))(pretrained=None)
        self.layer0 = model.layer0
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.res_layers = ['layer{}'.format(i) for i in range(1, 5)]

        self._freeze_stages()


    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained):
        if pretrained is None:
            return
        print ('Loading <{}> pretrained weights ...'.format(pretrained))
        model = getattr(pretrainedmodels, 'se_resnext{}_32x4d'.format(self.depth))(pretrained=pretrained)
        for layer_name in ['layer0'] + self.res_layers:
            layer_dict = getattr(model, layer_name).state_dict()
            getattr(self, layer_name).load_state_dict(layer_dict) 

    def forward(self, x):
        x = self.layer0(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
