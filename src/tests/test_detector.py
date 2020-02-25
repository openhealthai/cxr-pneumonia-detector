import numpy as np
import torch

from mmcv.utils.config import Config, ConfigDict
from mmcv.parallel import MMDataParallel
from mmdet.models.builder import build_detector

from factory import builder

mmdet_cfg = Config.fromfile('factory/models/SERetinaNeXt50.py')

model = build_detector(mmdet_cfg.model,
    mmdet_cfg.train_cfg,
    mmdet_cfg.test_cfg)

model = MMDataParallel(model, device_ids=[0])


X = np.ones(([2, 3, 512, 512]))
X = torch.from_numpy(X).float()

img_meta = {0 : {
    'img_shape': (3, 512, 512),
    'scale_factor': 1.,
    'flip': False
} }

y_pred = []
for x in X:
    y_pred.append(model([x.unsqueeze(0)], img_meta=[img_meta], return_loss=False))