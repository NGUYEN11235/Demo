import torch.nn as nn
import torch.nn.functional as F
import torch



class AttentionChannel(nn.Module):
    def __init__(self, out_channels, reduction_rate=2):
        super(AttentionChannel, self).__init__()
        rr = reduction_rate
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        se = x.mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        x = x * se2.unsqueeze(-1)
        return x


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channel, out_channels):
        super().__init__()

        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

        self.CA = AttentionChannel(out_channels)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        out = self.CA(out) + out
        return x + out


class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_channel,
        n_features,
        n_classes,
        n_layers):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x):
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageTCN(nn.Module):
  """
  Y. Abu Farha and J. Gall.
  MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
  In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019
  parameters used in originl paper:
      n_features: 64
      n_stages: 4
      n_layers: 10
  """

  def __init__(
      self,
      in_channel,
      n_features,
      n_classes,
      n_stages,
      n_layers):
      super().__init__()
      self.stage1 = SingleStageTCN(in_channel, n_features, n_classes, n_layers)

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

