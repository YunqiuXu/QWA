# Modified from https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot_embedding(labels, num_classes=2):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, use_cuda=True, use_alter=False):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        self.use_alter = use_alter

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2
        t = one_hot_embedding(y.data.cpu(), self.num_classes)
        if self.use_cuda:
            t = Variable(t).cuda()  # [N,20]
        else:
            t = Variable(t)
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w.detach(), reduction='sum')

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        t = one_hot_embedding(y.data.cpu(), self.num_classes)
        if self.use_cuda:
            t = Variable(t).cuda()
        else:
            t = Variable(t)
        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()
        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, preds, labels):
        """
        preds: [batch, 2] tensor
        labels: [batch, ] tensor
        """
        if not self.use_alter:
            return self.focal_loss(preds, labels)
        else:
            return self.focal_loss_alt(preds, labels)
