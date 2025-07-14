import os
import math
import torch
import random
import argparse
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import timm  # Vision Transformer library
import torch.nn.functional as F

from wavelets_vit import WaveletTransform3Channel
from data_imagine import get_dataset

class EEGViTClassifier(torch.nn.Module):
    """
    pkl → wavelet (3 bands) → ViT → classifier
    """
    def __init__(self, chans=122, samples=1651, cls=39, vit_model='vit_base_patch16_224', dropout=0.25):
        super().__init__()
        self.wavelet_encoder = WaveletTransform3Channel()
        self.vit = timm.create_model(vit_model, pretrained=False, num_classes=cls, in_chans=3)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, Chans, Samples)
        x = self.wavelet_encoder(x)  # (N, 3, Chans, Samples)
        x = self.dropout(x)
        # Resize to (N, 3, 224, 224) for ViT
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        out = self.vit(x)            # (N, cls)
        return out

class EEGDictDataset(Dataset):
    def __init__(self, data_dict, text_to_label_map):
        self.input_features = []
        self.labels = []
        for feature, text_label in zip(data_dict["input_features"], data_dict["labels"]):
            if text_label in text_to_label_map:
                self.input_features.append(feature)
                self.labels.append(text_to_label_map[text_label])

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return self.input_features[idx], label

def custom_collate(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    return inputs, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single-stream 3-band EEG ViT classification model.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr1', type=float, default=5e-4)
    parser.add_argument('--wd1', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_vit')
    parser.add_argument('--sub', type=str, default='01')
    parser.add_argument('--cls', type=int, default=39)
    parser.add_argument('--samples', type=int, default=1651)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--rand_guess', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='imagine_decode')
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    try:
        with open("./Chisco/json/textmaps.json", "r") as file:
            textmaps = json.load(file)
    except FileNotFoundError:
        print("Error: textmaps.json not found in ./Chisco/json/. Please ensure the file exists.")
        exit()

    all_data_dict = get_dataset(sub=args.sub)
    full_dataset = EEGDictDataset(all_data_dict, textmaps)
    if len(full_dataset) == 0:
        print("Error: No data was loaded. Check that the subject ID is correct and that text labels match the textmaps.json file.")
        exit()

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    trainset, validset = random_split(full_dataset, [train_size, valid_size])
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    print(f"Data loaded: {len(trainset)} training samples, {len(validset)} validation samples.")

    model = EEGViTClassifier(
        chans=122, samples=args.samples, cls=args.cls, vit_model=args.vit_model, dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(trainloader) * args.epoch)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"--- Starting Training on {device} ---")
    best_f1 = 0.0
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{args.epoch} | Avg Train Loss: {avg_train_loss:.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if len(all_preds) > 0 else 0
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_preds) > 0 else 0
        print(f"Epoch {epoch+1} Validation | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pt"))
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "checkpoint_best.pt"))
            print(f"*** New best model saved with F1-score: {best_f1:.4f} ***")

    print("--- Training Complete ---")







