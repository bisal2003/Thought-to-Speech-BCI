
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import random
# import argparse
# import json
# from torch.utils.data import DataLoader, Dataset, random_split
# from transformers import get_linear_schedule_with_warmup
# from sklearn.metrics import f1_score, confusion_matrix
# import timm
# from pathlib import Path
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns

# # --- 1. The NEW Dataset Class ---
# # This class loads individual pre-computed images.

# class PreprocessedEEGDataset(Dataset):
#     def __init__(self, subject_id, processed_data_root, text_to_label_map):
#         self.image_paths = []
#         self.labels = []
        
#         subject_path = processed_data_root / f"sub-{subject_id}"
#         images_dir = subject_path / "images"
#         label_file = subject_path / "labels.npy"

#         if not images_dir.exists() or not label_file.exists():
#             raise FileNotFoundError(f"Processed data not found for subject {subject_id} in {subject_path}. Please run preprocess_data.py first.")

#         # Get a sorted list of all individual image file paths
#         all_image_paths = sorted(list(images_dir.glob("*.npy")))
#         all_text_labels = np.load(label_file)

#         if len(all_image_paths) != len(all_text_labels):
#             raise ValueError("Mismatch between number of images and number of labels!")

#         # Map text labels to integer labels and store paths
#         for img_path, text_label in zip(all_image_paths, all_text_labels):
#             if text_label in text_to_label_map:
#                 self.image_paths.append(img_path)
#                 self.labels.append(text_to_label_map[text_label])

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # Load the image from its file path only when requested
#         image_path = self.image_paths[idx]
#         image = np.load(image_path)
        
#         image_tensor = torch.from_numpy(image).unsqueeze(0) # Add channel dimension -> (1, H, W)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return image_tensor, label

# # --- 2. The NEW, Simpler ViT Classifier ---
# # It no longer needs the WaveletImageTransform module.

# class EEGViTClassifier(torch.nn.Module):
#     def __init__(self, cls=39, vit_model='vit_tiny_patch16_224', dropout=0.25):
#         super().__init__()
#         # Create a ViT model. `in_chans=1` because our image is grayscale.
#         self.vit = timm.create_model(vit_model, pretrained=True, num_classes=cls, in_chans=1)
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, x):
#         # x is now the pre-computed image: (N, 1, H, W)
        
#         # 1. Resize the image to the size ViT expects (e.g., 224x224)
#         x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
#         # 2. Pass through dropout and ViT
#         x = self.dropout(x)
#         out = self.vit(x)
        
#         return out

# # --- 3. Main Training Loop ---

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a ViT model on pre-processed scalogram images.")
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--lr1', type=float, default=1e-4)
#     parser.add_argument('--wd1', type=float, default=0.01)
#     parser.add_argument('--epoch', type=int, default=50)
#     parser.add_argument('--batch', type=int, default=16)
#     parser.add_argument('--processed_data_root', type=str, default="./Chisco/derivatives/scalogram_images_1-8Hz", help="Root directory of pre-processed images.")
#     parser.add_argument('--checkpoint_path', type=str, default='checkpoints_vit_preprocessed')
#     parser.add_argument('--sub', type=str, default='01')
#     parser.add_argument('--cls', type=int, default=39)
#     parser.add_argument('--dropout', type=float, default=0.3)
#     parser.add_argument('--vit_model', type=str, default='vit_tiny_patch16_224')
#     # UPDATED: Made textmaps path an argument
#     parser.add_argument('--textmaps_path', type=str, default='./Chisco/json/textmaps.json', help="Path to the text-to-label mapping JSON file.")
#     args = parser.parse_args()
#     print(args)

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)

#     try:
#         # UPDATED: Use the argument for the path
#         with open(args.textmaps_path, "r") as file:
#             textmaps = json.load(file)
#     except FileNotFoundError:
#         print(f"Error: textmaps.json not found at {args.textmaps_path}")
#         exit()

#     # Load the pre-processed dataset
#     full_dataset = PreprocessedEEGDataset(args.sub, Path(args.processed_data_root), textmaps)
#     if len(full_dataset) == 0:
#         print("Error: No data was loaded. Check paths and ensure pre-processing was run.")
#         exit()

#     train_size = int(0.8 * len(full_dataset))
#     valid_size = len(full_dataset) - train_size
#     trainset, validset = random_split(full_dataset, [train_size, valid_size])
    
#     trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
#     validloader = DataLoader(validset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
#     print(f"Data loaded: {len(trainset)} training images, {len(validset)} validation images.")

#     model = EEGViTClassifier(cls=args.cls, vit_model=args.vit_model, dropout=args.dropout).to(device)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
#     scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(trainloader) * args.epoch)
#     criterion = torch.nn.CrossEntropyLoss()

#     print(f"--- Starting Training on {device} ---")
#     best_f1 = 0.0
#     for epoch in range(args.epoch):
#         model.train()
#         total_loss = 0
#         for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epoch}"):
#             images, labels = images.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             total_loss += loss.item()
        
#         avg_train_loss = total_loss / len(trainloader)
#         print(f"Epoch {epoch+1}/{args.epoch} | Avg Train Loss: {avg_train_loss:.4f}")

#         model.eval()
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for images, labels in validloader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 preds = torch.argmax(outputs, dim=1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if len(all_preds) > 0 else 0
#         macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_preds) > 0 else 0
#         print(f"Epoch {epoch+1} Validation | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

#         if macro_f1 > best_f1:
#             best_f1 = macro_f1
#             torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "checkpoint_best.pt"))
#             print(f"*** New best model saved with F1-score: {best_f1:.4f} ***")

#     print("--- Training Complete ---")


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import json
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix
import timm
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. The NEW Dataset Class ---
# This class loads individual pre-computed images.

class PreprocessedEEGDataset(Dataset):
    def __init__(self, subject_id, processed_data_root, text_to_label_map):
        self.image_paths = []
        self.labels = []
        
        subject_path = processed_data_root / f"sub-{subject_id}"
        images_dir = subject_path / "images"
        label_file = subject_path / "labels.npy"

        if not images_dir.exists() or not label_file.exists():
            raise FileNotFoundError(f"Processed data not found for subject {subject_id} in {subject_path}. Please run preprocess_data.py first.")

        # Get a sorted list of all individual image file paths
        all_image_paths = sorted(list(images_dir.glob("*.npy")))
        all_text_labels = np.load(label_file)

        if len(all_image_paths) != len(all_text_labels):
            raise ValueError("Mismatch between number of images and number of labels!")

        # Map text labels to integer labels and store paths
        for img_path, text_label in zip(all_image_paths, all_text_labels):
            if text_label in text_to_label_map:
                self.image_paths.append(img_path)
                self.labels.append(text_to_label_map[text_label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load the image from its file path only when requested
        image_path = self.image_paths[idx]
        image = np.load(image_path)
        
        image_tensor = torch.from_numpy(image).unsqueeze(0) # Add channel dimension -> (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image_tensor, label

# --- 2. The NEW, Simpler ViT Classifier ---
# It no longer needs the WaveletImageTransform module.

class EEGViTClassifier(torch.nn.Module):
    def __init__(self, cls=39, vit_model='vit_tiny_patch16_224', dropout=0.25):
        super().__init__()
        # Create a ViT model. `in_chans=1` because our image is grayscale.
        self.vit = timm.create_model(vit_model, pretrained=True, num_classes=cls, in_chans=1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x is now the pre-computed image: (N, 1, H, W)
        
        # 1. Resize the image to the size ViT expects (e.g., 224x224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 2. Pass through dropout and ViT
        x = self.dropout(x)
        out = self.vit(x)
        
        return out

# --- 3. Main Training Loop ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ViT model on pre-processed scalogram images.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr1', type=float, default=1e-4)
    parser.add_argument('--wd1', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--processed_data_root', type=str, default="./Chisco/derivatives/scalogram_images_1-8Hz", help="Root directory of pre-processed images.")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_vit_preprocessed')
    parser.add_argument('--sub', type=str, default='01')
    parser.add_argument('--cls', type=int, default=39)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--vit_model', type=str, default='vit_tiny_patch16_224')
    # UPDATED: Made textmaps path an argument
    parser.add_argument('--textmaps_path', type=str, default='./Chisco/json/textmaps.json', help="Path to the text-to-label mapping JSON file.")
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)

    try:
        # UPDATED: Use the argument for the path
        with open(args.textmaps_path, "r") as file:
            textmaps = json.load(file)
    except FileNotFoundError:
        print(f"Error: textmaps.json not found at {args.textmaps_path}")
        exit()

    # Load the pre-processed dataset
    full_dataset = PreprocessedEEGDataset(args.sub, Path(args.processed_data_root), textmaps)
    if len(full_dataset) == 0:
        print("Error: No data was loaded. Check paths and ensure pre-processing was run.")
        exit()

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    trainset, validset = random_split(full_dataset, [train_size, valid_size])
    
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Data loaded: {len(trainset)} training images, {len(validset)} validation images.")

    model = EEGViTClassifier(cls=args.cls, vit_model=args.vit_model, dropout=args.dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(trainloader) * args.epoch)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"--- Starting Training on {device} ---")
    best_f1 = 0.0
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epoch}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
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
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if len(all_preds) > 0 else 0
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_preds) > 0 else 0
        print(f"Epoch {epoch+1} Validation | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "checkpoint_best.pt"))
            print(f"*** New best model saved with F1-score: {best_f1:.4f} ***")

    print("--- Training Complete ---")
