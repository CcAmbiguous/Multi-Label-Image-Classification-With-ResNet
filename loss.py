import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiLabelBCELoss(nn.Module):
    def __init__(self):
        super(MultiLabelBCELoss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

