# import os
# import math
# import torch
# import random
# import argparse
# import numpy as np
# import json
# from torch.utils.data import DataLoader, Dataset, random_split
# from transformers import get_linear_schedule_with_warmup
# from sklearn.metrics import f1_score

# # Use the new 3-band wavelet and EfficientNet-based CNN
# from eegcnn3bands import EEGEfficientNet, PositionalEncoding
# from wavelets_3bands import WaveletTransform3Channel
# from data_imagine import get_dataset

# class EEGclassification(torch.nn.Module):
#     """
#     Single-stream: pkl → wavelet (3 bands) → EfficientNet → Transformer → classifier
#     """
#     def __init__(self, chans=122, samples=1651, cls=39, dropout1=0.25, dropout2=0.25, layer=2, pooling='mean', effnet_version='b0'):
#         super().__init__()
#         self.wavelet_encoder = WaveletTransform3Channel()
#         self.eegcnn = EEGEfficientNet(chans=chans, samples=samples, out_dim=1280, version=effnet_version)
#         self.layer = layer
#         self.pooling = pooling
#         self.feature_dim = 1280  # EfficientNet-b0 output
#         if self.layer > 0:
#             self.poscode = PositionalEncoding(self.feature_dim, dropout=dropout2, max_len=samples)
#             self.encoder = torch.nn.TransformerEncoder(
#                 torch.nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=max(1, self.feature_dim//64), dim_feedforward=4*self.feature_dim, batch_first=True, dropout=dropout2),
#                 num_layers=self.layer)
#         self.linear = torch.nn.Linear(self.feature_dim, cls)

#     def forward(self, x, mask):
#         # x: (N, Chans, Samples)
#         x = self.wavelet_encoder(x)          # -> (N, 3, Chans, Samples)
#         # EfficientNet expects (N, 3, H, W) where H=Chans, W=Samples
#         features = self.eegcnn(x)            # -> (N, feature_dim)
#         # Optionally add positional encoding and transformer
#         if self.layer > 0:
#             features = features.unsqueeze(1)  # (N, 1, feature_dim)
#             features = self.poscode(features)
#             features = self.encoder(features, src_key_padding_mask=(mask.bool()==False))
#             features = features.squeeze(1)    # (N, feature_dim)
#         output = self.linear(features)
#         return output

# class EEGDictDataset(Dataset):
#     def __init__(self, data_dict, text_to_label_map):
#         self.input_features = []
#         self.labels = []
#         for feature, text_label in zip(data_dict["input_features"], data_dict["labels"]):
#             if text_label in text_to_label_map:
#                 self.input_features.append(feature)
#                 self.labels.append(text_to_label_map[text_label])

#     def __len__(self):
#         return len(self.input_features)

#     def __getitem__(self, idx):
#         mask = torch.ones(1651)  # Updated to match 'samples'
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return self.input_features[idx], label, mask

# def custom_collate(batch):
#     inputs, labels, masks = zip(*batch)
#     inputs = torch.stack(inputs, dim=0)
#     labels = torch.stack(labels, dim=0)
#     masks = torch.stack(masks, dim=0)
#     return inputs, labels, masks

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a single-stream 3-band EEG EfficientNet classification model.")
#     parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
#     parser.add_argument('--lr1', type=float, default=5e-4, help="Learning rate for the optimizer.")
#     parser.add_argument('--wd1', type=float, default=0.01, help="Weight decay for the optimizer.")
#     parser.add_argument('--epoch', type=int, default=100, help="Total number of training epochs.")
#     parser.add_argument('--batch', type=int, default=16, help="Batch size for training and validation. Lower if you have memory issues.")
#     parser.add_argument('--checkpoint_path', type=str, default='checkpoints_3band_rgb', help="Directory to save model checkpoints.")
#     parser.add_argument('--sub', type=str, default='01', help="Subject ID to train on.")
#     parser.add_argument('--cls', type=int, default=39, help="Number of output classes.")
#     parser.add_argument('--samples', type=int, default=1651, help="Number of samples per trial (timepoints).")
#     parser.add_argument('--pooling', type=str, default='mean', help="Pooling strategy: 'mean', 'max', or None.")
#     parser.add_argument('--layer', type=int, default=2, help="Number of Transformer layers.")
#     parser.add_argument('--dropout1', type=float, default=0.25, help="Dropout rate for the CNN.")
#     parser.add_argument('--dropout2', type=float, default=0.25, help="Dropout rate for the Transformer.")
#     parser.add_argument('--effnet_version', type=str, default='b0', help="EfficientNet version: 'b0' or 'b1'.")
#     parser.add_argument('--rand_guess', type=int, default=0)
#     parser.add_argument('--dataset', type=str, default='imagine_decode')
#     parser.add_argument('--subset_ratio', type=float, default=1.0)
#     args = parser.parse_args()
#     print(args)

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     if not os.path.exists(args.checkpoint_path):
#         os.makedirs(args.checkpoint_path)

#     try:
#         with open("./Chisco/json/textmaps.json", "r") as file:
#             textmaps = json.load(file)
#     except FileNotFoundError:
#         print("Error: textmaps.json not found in ./Chisco/json/. Please ensure the file exists.")
#         exit()

#     all_data_dict = get_dataset(sub=args.sub)
#     full_dataset = EEGDictDataset(all_data_dict, textmaps)
#     if len(full_dataset) == 0:
#         print("Error: No data was loaded. Check that the subject ID is correct and that text labels match the textmaps.json file.")
#         exit()

#     train_size = int(0.8 * len(full_dataset))
#     valid_size = len(full_dataset) - train_size
#     trainset, validset = random_split(full_dataset, [train_size, valid_size])
#     trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
#     validloader = DataLoader(validset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)
#     print(f"Data loaded: {len(trainset)} training samples, {len(validset)} validation samples.")

#     model = EEGclassification(
#         chans=122, samples=args.samples, cls=args.cls, dropout1=args.dropout1, 
#         dropout2=args.dropout2, layer=args.layer, pooling=args.pooling, effnet_version=args.effnet_version
#     ).to(device)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
#     scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(trainloader) * args.epoch)
#     criterion = torch.nn.CrossEntropyLoss()

#     print(f"--- Starting Training on {device} ---")
#     best_f1 = 0.0
#     for epoch in range(args.epoch):
#         model.train()
#         total_loss = 0
#         for inputs, labels, mask in trainloader:
#             inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs, mask)
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
#             for inputs, labels, mask in validloader:
#                 inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
#                 outputs = model(inputs, mask)
#                 preds = torch.argmax(outputs, dim=1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if len(all_preds) > 0 else 0
#         macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_preds) > 0 else 0
#         print(f"Epoch {epoch+1} Validation | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

#         torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pt"))
#         if macro_f1 > best_f1:
#             best_f1 = macro_f1
#             torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "checkpoint_best.pt"))
#             print(f"*** New best model saved with F1-score: {best_f1:.4f} ***")

#     print("--- Training Complete ---")

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

# Use the new 3-band wavelet and EfficientNet-based CNN
from eegcnn3bands import EEGEfficientNet
from wavelets_3bands import WaveletTransform3Channel
from data_imagine import get_dataset

class EEGclassification(torch.nn.Module):
    """
    Single-stream: pkl → wavelet (3 bands) → EfficientNet → classifier
    (Transformer removed, EfficientNet output used directly)
    """
    def __init__(self, chans=122, samples=1651, cls=39, dropout1=0.25, dropout2=0.25, layer=0, pooling='mean', effnet_version='b0'):
        super().__init__()
        self.wavelet_encoder = WaveletTransform3Channel()
        self.eegcnn = EEGEfficientNet(chans=chans, samples=samples, out_dim=1280, version=effnet_version)
        self.feature_dim = 1280  # EfficientNet-b0 output
        self.linear = torch.nn.Linear(self.feature_dim, cls)

    def forward(self, x, mask=None):
        # x: (N, Chans, Samples)
        x = self.wavelet_encoder(x)          # -> (N, 3, Chans, Samples)
        features = self.eegcnn(x)            # -> (N, feature_dim)
        output = self.linear(features)
        return output

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
    parser = argparse.ArgumentParser(description="Train a single-stream 3-band EEG EfficientNet classification model.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--lr1', type=float, default=5e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--wd1', type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument('--epoch', type=int, default=100, help="Total number of training epochs.")
    parser.add_argument('--batch', type=int, default=16, help="Batch size for training and validation. Lower if you have memory issues.")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_3band_rgb', help="Directory to save model checkpoints.")
    parser.add_argument('--sub', type=str, default='01', help="Subject ID to train on.")
    parser.add_argument('--cls', type=int, default=39, help="Number of output classes.")
    parser.add_argument('--samples', type=int, default=1651, help="Number of samples per trial (timepoints).")
    parser.add_argument('--pooling', type=str, default='mean', help="Pooling strategy: 'mean', 'max', or None.")
    parser.add_argument('--layer', type=int, default=0, help="Number of Transformer layers (ignored, always 0).")
    parser.add_argument('--dropout1', type=float, default=0.25, help="Dropout rate for the CNN.")
    parser.add_argument('--dropout2', type=float, default=0.25, help="Dropout rate for the Transformer (ignored).")
    parser.add_argument('--effnet_version', type=str, default='b0', help="EfficientNet version: 'b0' or 'b1'.")
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

    model = EEGclassification(
        chans=122, samples=args.samples, cls=args.cls, dropout1=args.dropout1, 
        dropout2=args.dropout2, layer=0, pooling=args.pooling, effnet_version=args.effnet_version
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