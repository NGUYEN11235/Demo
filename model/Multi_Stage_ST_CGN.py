from Multi_Stage_TCN import SingleStageTCN
from ST_GCN import ST_GCN
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import torch

class MS_ST_GCN(nn.Module):
    def __init__(self, in_channels, n_features, n_classes, n_layers, n_stages):
      super(MS_ST_GCN, self).__init__()
      self.stage1 = ST_GCN(in_channels, 64, n_classes, n_layers)
      stages = [
          SingleStageTCN(n_classes, n_features, n_classes, n_layers)
          for _ in range(n_stages - 1)
      ]
      self.stages = nn.ModuleList(stages)

      if n_classes == 1:
          self.activation = nn.Sigmoid()
      else:
          self.activation = nn.Softmax(dim=1)

    def forward(self, x):
      if self.training:
            # for training
            outputs = []
            out = self.stage1(x)
            outputs.append(out)
            for stage in self.stages:
                out = stage(self.activation(out))
                outputs.append(out)
            return outputs
      else:
            # for evaluation
            out = self.stage1(x)
            for stage in self.stages:
                out = stage(self.activation(out))
            return out

if __name__ == '__main__':
    model = MS_ST_GCN(in_channels=3, n_features=64, n_classes=4, n_layers=9, n_stages=2)
    model.load_state_dict(torch.load('model_best.pth'))
    model.eval()

    model.to('cuda')
    sample = pd.read_pickle('demo3.pkl')
    kp, image_size = sample['pose_results'].copy(), sample['image_size']

    h, w = image_size

    kp[:, :, 0] = (kp[:, :, 0] - w / 2) / (w / 2)
    kp[:, :, 1] = (kp[:, :, 1] - h / 2) / (h / 2)

    # ft = []
    # for k in kp:
    #   ft.append(k.flatten())
    # ft = np.array(ft)
    # ft = torch.from_numpy(ft).t().float()

    ft = torch.from_numpy(kp).float()
    ft = ft.permute(2, 0, 1).contiguous()
    out = model(ft[None].to('cuda'))