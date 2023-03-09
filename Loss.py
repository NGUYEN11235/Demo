import torch.nn as nn
import torch
import torch.nn.functional as F

class TMSE(nn.Module):
    """
    Temporal MSE Loss Function
    Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
    arXiv: https://arxiv.org/pdf/1903.01945.pdf
    """

    def __init__(self, threshold=4):
        super().__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")
        self.ignore_index = 255

    def forward(self, preds, gts):
        total_loss = 0.0
        batch_size = preds.shape[0]
        for pred, gt in zip(preds, gts):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]
            loss = self.mse(F.log_softmax(pred[:, 1:], dim=1), F.log_softmax(pred[:, :-1], dim=1))
            loss = torch.clamp(loss, min=0, max=self.threshold ** 2)
            total_loss += torch.mean(loss)
        return total_loss / batch_size


class CE(nn.Module):
    def __init__(self, weight):
        super().__init__()
        weight = weight.to('cuda')
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, gt):
        loss = self.ce(pred, gt)
        return loss


class Loss_fcn(nn.Module):
    def __init__(self, weight_class=torch.tensor([1.,1.,1.,1.]).to('cuda'), threshold=4, weight_tmse=0.15):
        super().__init__()
        self.ce = CE(weight_class)
        self.tmse = TMSE(threshold)
        self.weight = weight_tmse

    def forward(self, pred, gt):
        loss = self.ce(pred, gt) + self.weight * self.tmse(pred, gt)
        return loss
