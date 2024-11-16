import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import copy
from torchvision import models, transforms
from commons.t1dconv import Temporal1DConv
from commons.biLSTM import BiLSTMLayer
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Some weights of*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Conv2D_Conv1D_LSTM(nn.Module):
    def __init__(self, num_classes=100, hidden_size=512):
        super(Conv2D_Conv1D_LSTM, self).__init__()

        self.num_classes = num_classes

        self.conv_2d = models.resnet18(pretrained=True)
        self.conv_2d.fc = nn.Identity()
        self.conv_2d.avgpool = nn.Identity()
        self.fc = nn.Linear(512*7*7, hidden_size)

        self.temporal_conv1D = Temporal1DConv(hidden_size, hidden_size, num_classes, num_layers=2)
        self.temporal_model = BiLSTMLayer(input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(hidden_size*2, self.num_classes)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def forward(self, x, device):
        res = []
        for batch in x:
            frame_features = []
            prev_frame_path = ""
            prev_frame = None

            for frame in batch:
                if prev_frame_path == frame:
                    frame_features.append(prev_frame)
                    continue

                img = Image.open(frame)
                img = self.transform(img).unsqueeze(0).to(device)

                features = self.conv_2d(img)
                features = F.relu(self.fc(features))

                prev_frame_path = frame
                prev_frame = features
                frame_features.append(features)

            frame_features = torch.stack(frame_features, dim=1).permute(0, 2, 1)
            
            conv1d_features = self.temporal_conv1D(frame_features).permute(0, 2, 1)
            lstm_output = self.temporal_model(conv1d_features)

            temp_res = self.classifier(lstm_output)
            temp_res = F.log_softmax(temp_res, dim=-1)

            res.append(temp_res.squeeze(0))

        return torch.stack(res, dim=0).to(device)

if __name__ == '__main__':
    x = torch.randn(8, 100, 3, 224, 224).to("mps")
    vac = Conv2D_Conv1D_LSTM(num_classes=200, hidden_size=512).to("mps")
    output = vac(x)