import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import models, transforms
from commons.t1dconv import Temporal1DConv
from commons.biLSTM import BiLSTMLayer
from utils.data_generator_ArCSL import data_generator_ArCSL
import clip
from PIL import Image
from torch.nn import MultiheadAttention

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Some weights of*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


aggregation_settings = {
    "AGG_AVG" : 0,
    "AGG_ATTN" : 1,
}

class CLIP_Img_Encoder(nn.Module):
    def __init__(self, model_name, freeze=False, hidden_size=512):
        super(CLIP_Img_Encoder, self).__init__()

        self.freeze = freeze
        self.model, _ = clip.load(model_name, jit=False)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.model.encode_image(x).float()
        x = self.fc(x)

        return x # [batch_size, 512]

class CLIP_Vid_Encoder(nn.Module):
    def __init__(
            self, 
            hidden_size=512, 
            img_encoder_model_name="ViT-B/32",
            freeze_img_encoder=False,
            t1d_conv_layers=2, 
            lstm_layers=2,
            aggregation_method=aggregation_settings["AGG_ATTN"]
        ):
        super(CLIP_Vid_Encoder, self).__init__()

        self.img_encoder = CLIP_Img_Encoder(img_encoder_model_name, freeze=freeze_img_encoder, hidden_size=hidden_size)

        self.temporal_conv1D = Temporal1DConv(hidden_size, hidden_size, num_layers=t1d_conv_layers)
        self.temporal_model = BiLSTMLayer(input_size=hidden_size, hidden_size=hidden_size, num_layers=lstm_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.aggregation_method = aggregation_method
        if aggregation_method == aggregation_settings["AGG_ATTN"]:
            self.agg_net = nn.Linear(hidden_size, 1)
            
    def aggregate(self, x):
        if self.aggregation_method == aggregation_settings["AGG_AVG"]:
            x = x.squeeze(0) # [123, 512]
            return torch.mean(x, dim=0) # [512]
        elif self.aggregation_method == aggregation_settings["AGG_ATTN"]:
            x = x.squeeze(0) # [123, 512]
            weights = F.softmax(self.agg_net(x))
            return torch.sum(weights * x, dim=0) # [512]

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

                img = Image.open(frame) #[1, 3, 224, 224]
                img = self.transform(img).unsqueeze(0).to(device)

                features = self.img_encoder(img) #[1, 512]

                prev_frame_path = frame
                prev_frame = features
                frame_features.append(features)
            
            frame_features = torch.stack(frame_features, dim=1).permute(0, 2, 1) # [500, 1, 512] -> [1, 512, 500]
            conv1d_features = self.temporal_conv1D(frame_features).permute(0, 2, 1) # [1, 123, 512]
            lstm_output = self.temporal_model(conv1d_features) # [1, 123, 1024]

            temp_res = self.fc(lstm_output) # [1, 123, 512]
            temp_res_agg = self.aggregate(temp_res) # [512]

            res.append(temp_res_agg) # [512]

        return torch.stack(res, dim=0).to(device) # [batch_size, 512]

def test():
    data_gen = data_generator_ArCSL("./data/ArCSL", batch_size=2)
    model = CLIP_Vid_Encoder(hidden_size=256).to("mps")
    for batch in data_gen:
        frames = batch['frames']

        output = model(frames, "mps")
        print(output.shape)

        break
    