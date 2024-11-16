import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import copy

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Some weights of*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Temporal1DConv(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Temporal1DConv, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_layers < 1:
            self.temporal_1Dconv1 = nn.Identity()
            self.temporal_1Dconv1 = nn.Identity()

        if num_layers >= 1:
            self.temporal_1Dconv1 = nn.Sequential(
                nn.Conv1d(self.input_size, self.hidden_size, kernel_size=5, stride=1, padding=0),
                nn.MaxPool1d(kernel_size=2, ceil_mode=False),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(inplace=True),
            )
            self.temporal_1Dconv2 = nn.Identity()

        if num_layers == 2:
            self.temporal_1Dconv2 = nn.Sequential(
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=0),
                nn.MaxPool1d(kernel_size=2, ceil_mode=False),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(inplace=True),
            )

    def forward(self, frame_feat):
        res = self.temporal_1Dconv1(frame_feat)
        res = self.temporal_1Dconv2(res)

        return res

if __name__ == '__main__':
    model = Temporal1DConv(100, 512)
    frame_feat = torch.randn(8, 100, 262144)
    feature_length = torch.tensor([100])
    out = model(frame_feat, feature_length)
    print(out['visual_features'].shape)
    print(out['conv_logits'].shape)
    print(out['feature_len'])
