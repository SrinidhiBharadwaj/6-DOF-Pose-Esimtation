import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss
