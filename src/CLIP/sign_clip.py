import torch
import torch.nn as nn
import torch.nn.functional as F
from CLIP.video_encoder import CLIP_Vid_Encoder, test
from CLIP.text_encoder import CLIP_Text_Encoder
from utils.data_generator_ArCSL import *

aggregation_settings = {
    "AGG_AVG" : 0,
    "AGG_ATTN" : 1,
}

class SignCLIP(nn.Module):
    def __init__(
            self, 
            img_encoder_model_name="ViT-B/32", 
            text_encoder_model_name="M-CLIP/XLM-Roberta-Large-Vit-B-32", 
            projection_input_size=512,
            hidden_size=512,
            freeze_img_encoder=False, 
            freeze_text_encoder=False, 
            t1d_conv_layers=2, 
            lstm_layers=2, 
            aggregation_method=aggregation_settings["AGG_ATTN"]
        ):
        super(SignCLIP, self).__init__()
        
        self.video_encoder = CLIP_Vid_Encoder(
            img_encoder_model_name=img_encoder_model_name, 
            freeze_img_encoder=freeze_img_encoder,
            t1d_conv_layers=t1d_conv_layers,
            lstm_layers=lstm_layers,
            aggregation_method=aggregation_method
        )
        
        self.text_encoder = CLIP_Text_Encoder(
            model_name=text_encoder_model_name,
            freeze=freeze_text_encoder
        )
        
        self.img_projection = nn.Linear(projection_input_size, hidden_size)
        self.text_projection = nn.Linear(projection_input_size, hidden_size)

    def forward(self, video_frames, text_inputs, device):
        video_embeddings = self.video_encoder(video_frames, device)  # [batch_size, hidden_size]
        text_embeddings = self.text_encoder(text_inputs, device)  # [batch_size, hidden_size]
        
        video_embeddings = F.normalize(self.img_projection(video_embeddings), p=2, dim=1)  # [batch_size, hidden_size]
        text_embeddings = F.normalize(self.text_projection(text_embeddings), p=2, dim=1)  # [batch_size, hidden_size]

        return video_embeddings, text_embeddings


def test():
    vision_text_model_pairs = [
        ("ViT-B/32", "M-CLIP/XLM-Roberta-Large-Vit-B-32", 512),
        ("ViT-L/14", "M-CLIP/XLM-Roberta-Large-Vit-L-14", 768),
    ]

    model = SignCLIP(
        img_encoder_model_name=vision_text_model_pairs[0][0],
        text_encoder_model_name=vision_text_model_pairs[0][1],
        projection_input_size=vision_text_model_pairs[0][2],
        hidden_size=256,
        freeze_img_encoder=False,
        freeze_text_encoder=False,
        t1d_conv_layers=2,
        lstm_layers=2,
        aggregation_method=aggregation_settings["AGG_AVG"]
    ).to("mps")

    data_gen = data_generator_ArCSL("./data/ArCSL", batch_size=1)
    for batch in data_gen:
        frames = batch['frames']
        text_inputs = ["Three blind horses listening to Mozart.", "Älgen är skogens konung!"]

        video_embeddings, text_embeddings = model(frames, text_inputs[0], "mps")

        print(video_embeddings.shape, text_embeddings.shape)

        break
