import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(pred,
                       target,
                       gamma=2.0,
                       alpha=0.25):
    pred_sigmoid = pred.sigmoid()
    target = F.one_hot(target.long(), pred.shape[-1]+1)[:,1:].type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    return loss

# TODO: remove this module
class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        assert logits.is_cuda
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '(gamma={}, alpha={})'.format(
            self.gamma, self.alpha)
        return tmpstr
