import os
import pandas as pd
import numpy as np
import random
import re
import os
import argparse
import torch
import torch.nn as nn
from CLIP.sign_clip import SignCLIP, test
from utils.data_generator_ArCSL import *


if __name__ == "__main__":
    vision_text_model_pairs = [
        ("ViT-B/32", "M-CLIP/XLM-Roberta-Large-Vit-B-32"),
        ("ViT-L/14", "M-CLIP/XLM-Roberta-Large-Vit-L-14"),
    ]

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', dest='data_path')
    # parser.add_argument('--model', dest='model', default="Conv2D_Conv1D_LSTM")
    # parser.add_argument('--device',dest='device', default='cuda')
    # parser.add_argument('--epochs',dest='epochs', default=40)
    # parser.add_argument('--batch_size',dest='batch_size', default=2)
    # parser.add_argument('--phase',dest='phase', default="train")
    # args = parser.parse_args()

    # main(args)
    test()