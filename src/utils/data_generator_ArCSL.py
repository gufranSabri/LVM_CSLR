import os
import warnings
import random
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch
from pprint import pprint
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_gloss_dict(data_path):
    gloss_dict = {}
    gloss_set_path = os.path.join(data_path, "gloss_set.txt")
    with open(gloss_set_path, 'r') as f:
        glosses = f.readlines()
        for i,g in enumerate(glosses):
            gloss_dict[g.replace("\n", "")] = i+1
            
    return gloss_dict

def total_vids_len(phase="train"):
    if phase == "train":
        return 500*10
    
    if phase == "valid":
        return 500*3
    
    if phase == "test":
        return 500*2

def data_generator_ArCSL(data_path, batch_size=16, phase="test", return_str = False):
    vids = []
    frames_path = os.path.join(data_path, "1st_500_frames")
    
    signers = [str(i).zfill(2) for i in range(10)]
    if phase == "valid":
        signers = [str(i).zfill(2) for i in range(10, 13)]
    if phase == "test":
        signers = [str(i).zfill(2) for i in range(13, 15)]
    signs = [str(i).zfill(4) for i in range(1, 501)]

    for signer in signers:
        for sign in signs:
            vids.append((signer,sign))

    random.shuffle(vids)

    gloss_dict = get_gloss_dict(data_path)
    ground_truth_gloss_path = os.path.join(data_path, "GroundTruth_gloss.txt")
    with open(ground_truth_gloss_path, 'r') as f:
        labels = f.readlines()
        labels = [label.split(" ") for label in labels]

        for i, label in enumerate(labels):
            labels[i] = [l.replace("\n", "") for l in label]
        
        if not return_str:
            for i, label in enumerate(labels):
                labels[i] = [gloss_dict[l.replace("\n", "")] for l in label]

    for i in range(0, len(vids), batch_size):
        batch = vids[i:i+batch_size]
        y = [labels[int(b[1])-1] for b in batch]
        
        frames_paths = [os.path.join(frames_path, signer, sign) for signer, sign in batch]
        frames = [[] for _ in range(batch_size)]

        if len(frames_paths) < batch_size:
            break
        for j in range(batch_size):
            for frame in os.listdir(frames_paths[j]):
                frames[j].append(os.path.join(frames_paths[j], frame))

            frames[j].sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            if len(frames[j]) < 500:
                pad = 500 - len(frames[j])
                frames[j] = frames[j] + [frames[j][-1]] * pad
        
        yield {"frames":frames, "y":y, "labels":labels}

if __name__ == "__main__":
    data_path = "./data/ArCSL"
    d = data_generator_ArCSL(data_path, return_str=True)

    for batch in d:
        pass