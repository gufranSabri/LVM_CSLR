import os
import pandas as pd
import numpy as np
import random
import datetime
import os
import argparse
import torch
import torch.nn as nn
from CLIP.sign_clip import SignCLIP, test
from utils.data_generator_ArCSL import *
from utils.scheduler import LinearDecayLR
from torch.utils.tensorboard import SummaryWriter

aggregation_settings = {
    "AGG_AVG" : 0,
    "AGG_ATTN" : 1,
}

vision_text_model_pairs = [
    ("ViT-B/32", "M-CLIP/XLM-Roberta-Large-Vit-B-32", 512),
    ("ViT-L/14", "M-CLIP/XLM-Roberta-Large-Vit-L-14", 768),
]

def main(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.device == 'mps':
        torch.mps.manual_seed(seed)
        torch.backends.mps.deterministic=True
        torch.backends.mps.benchmark = False
    elif args.device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    model = None
    if args.model == "B":
        model = SignCLIP(
            img_encoder_model_name=vision_text_model_pairs[0][0],
            text_encoder_model_name=vision_text_model_pairs[0][1],
            projection_input_size=vision_text_model_pairs[0][2],
            hidden_size=int(args.hidden_size),
            freeze_img_encoder= True if int(args.freeze_clip_modules) == 1 else False,
            freeze_text_encoder=True if int(args.freeze_clip_modules) == 1 else False,
            t1d_conv_layers=int(args.t1d_conv_layers),
            lstm_layers=int(args.lstm_layers),
            aggregation_method=aggregation_settings[args.agg_method]
        ).to(args.device)

    elif args.model == "L":
        model = SignCLIP(
            img_encoder_model_name=vision_text_model_pairs[1][0],
            text_encoder_model_name=vision_text_model_pairs[1][1],
            projection_input_size=vision_text_model_pairs[1][2],
            hidden_size=int(args.hidden_size),
            freeze_img_encoder=True if int(args.freeze_clip_modules) == 1 else False,
            freeze_text_encoder=True if int(args.freeze_clip_modules) == 1 else False,
            t1d_conv_layers=2,
            lstm_layers=2,
            aggregation_method=aggregation_settings[args.agg_method]
        ).to(args.device)
    
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f"SignCLIP_{args.model}_{args.lr}_{args.hidden_size}_f{args.freeze_clip_modules}_{args.agg_method}_b{args.batch_size}_{date}"
    writer = SummaryWriter(f"runs/{model_name}")

    if not os.path.exists("./models"):
        os.mkdir("./models")
        os.mkdir("./models/SignCLIP")        

    model = train(model, args, writer)
    model_path = f"./models/SignCLIP/{model_name}.tar"
    torch.save({
        "vid_encoder":model.video_encoder.state_dict(),
        "text_encoder":model.text_encoder.state_dict(),
        "img_projection":model.img_projection.state_dict(),
        "text_projection":model.text_projection.state_dict(),
        
    }, model_path)

def train(model, args, writer):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    
    lr_scheduler=LinearDecayLR(optimizer, int(args.epochs), int(int(args.epochs)*0.75))
    for epoch in range(int(args.epochs)):
        data_gen = data_generator_ArCSL(args.data_path, batch_size=args.batch_size, return_str=True)
        num_batches = total_vids_len("train") // args.batch_size
        training_loss = 0

        model.train()
        for _, batch in enumerate(tqdm(data_gen, desc=f"Epoch: {epoch+1}", total=num_batches)):
            frames = batch['frames']
            y = [" ".join(text) for text in batch['y']]

            video_embeddings, text_embeddings = model(frames, y, args.device)
            similarity_matrix = torch.matmul(video_embeddings, text_embeddings.T)
            labels = torch.arange(similarity_matrix.size(0)).to(args.device)

            loss_i = criterion(similarity_matrix, labels)  # Predicting text from video
            loss_t = criterion(similarity_matrix.T, labels)  # Predicting video from text
            loss = (loss_i + loss_t) / 2  # Bidirectional loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()

        writer.add_scalar("Loss/train", training_loss / num_batches, epoch)
        writer.add_scalar("Learning_Rate", {lr_scheduler.get_lr()}, epoch)

        print(f"Training Loss: {training_loss / num_batches:.4f}")
        print(f"Learning Rate: {lr_scheduler.get_lr()}\n")

        lr_scheduler.step()

        if epoch % 10 == 0:
            validate(model, args, writer, epoch)
    
    return model

def validate(model, args, writer, epoch):
    data_gen = data_generator_ArCSL(args.data_path, batch_size=args.batch_size, return_str=True, phase="test")
    num_batches = total_vids_len("test") // args.batch_size

    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        running_loss = 0
        for batch in tqdm(data_gen, desc="Validation", total=num_batches):
            frames = batch['frames']
            y = [" ".join(text) for text in batch['y']]

            video_embeddings, text_embeddings = model(frames, y, args.device)
            similarity_matrix = torch.matmul(video_embeddings, text_embeddings.T)
            labels = torch.arange(similarity_matrix.size(0)).to(args.device)
            loss_i = criterion(similarity_matrix, labels)  # Predicting text from video
            loss_t = criterion(similarity_matrix.T, labels)  # Predicting video from text
            loss = (loss_i + loss_t) / 2  # Bidirectional loss

            running_loss += loss.item()

        writer.add_scalar("Loss/Validation", running_loss/num_batches, epoch)

    print(f"Validation Loss: {running_loss/num_batches:.4f}")

def compute_metrics(embeddings1, embeddings2, ks=[1, 5, 10]):
    similarity_matrix = torch.matmul(embeddings1, embeddings2.T)
    num_samples = similarity_matrix.size(0)

    ranks = []
    for i in range(num_samples):
        scores = similarity_matrix[i]
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    recall_at_k = []
    for k in ks:
        recall_at_k.append(sum(r <= k for r in ranks) / num_samples)

    median_rank = torch.median(torch.tensor(ranks, dtype=torch.float)).item()
    return recall_at_k, median_rank

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default="./data/ArCSL")
    parser.add_argument('--model', dest='model', default="B")
    parser.add_argument('--lstm_layers', dest='lstm_layers', default="2")
    parser.add_argument('--t1d_conv_layers', dest='t1d_conv_layers', default="2")
    parser.add_argument('--hidden_size', dest='hidden_size', default="256")
    parser.add_argument('--freeze_clip_modules', dest='freeze_clip_modules', default="1")
    parser.add_argument('--lr', dest='lr', default="0.001")
    parser.add_argument('--agg_method', dest='agg_method', default="AGG_ATTN")
    parser.add_argument('--device',dest='device', default='cuda')
    parser.add_argument('--epochs',dest='epochs', default="20")
    parser.add_argument('--batch_size',dest='batch_size', default=4)
    args = parser.parse_args()

    main(args)
    