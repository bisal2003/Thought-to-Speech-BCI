
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import random
import argparse
import json
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import timm  # Vision Transformer library
from joblib import Parallel, delayed
from data_imagine import get_dataset

# --- 1. Wavelet Transform to Create a Single Image ---

def _create_channel_image(signal, scales, wavelet_name, sfreq):
    """
    Performs CWT on a single channel signal (1, Samples) and returns its magnitude.
    Output shape: (num_freqs, Samples)
    """
    coeffs, _ = pywt.cwt(signal, scales, wavelet_name, sampling_period=1.0/sfreq)
    return np.abs(coeffs)

class WaveletImageTransform(nn.Module):
    """
    A non-trainable module that converts a multi-channel EEG trial into a single,
    large 2D image by stacking the time-frequency maps of all channels.

    Input: (N, Chans, Samples)
    Output: (N, 1, Chans * num_freqs, Samples)
    """
    def __init__(self, sfreq=500, fmin=1, fmax=8, n_freqs=8, wavelet_name='morl', n_jobs=-1): # MODIFIED: Default frequencies
        super().__init__()
        self.sfreq = sfreq
        self.wavelet_name = wavelet_name
        self.n_jobs = n_jobs
        
        # Define frequencies based on the new range
        self.freqs = np.linspace(fmin, fmax, n_freqs)
        self.scales = pywt.central_frequency(wavelet_name) * sfreq / self.freqs
        
        # This layer is a fixed transformation, not meant to be trained
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Applies the CWT to each channel and stacks the results into a single image.
        """
        device = x.device
        batch_size, n_channels, n_samples = x.shape
        
        x_cpu = x.detach().cpu().numpy()
        
        # Process each trial in the batch
        final_images = []
        for i in range(batch_size):
            # For one trial, get all channel signals
            trial_signals = x_cpu[i, :, :] # (Chans, Samples)
            
            # In parallel, compute the CWT for each channel of the current trial
            channel_images = Parallel(n_jobs=self.n_jobs)(
                delayed(_create_channel_image)(
                    signal, self.scales, self.wavelet_name, self.sfreq
                ) for signal in trial_signals
            )
            
            # Stack the channel images vertically to form one large image
            # List of (n_freqs, n_samples) -> (Chans * n_freqs, n_samples)
            full_image = np.vstack(channel_images)
            final_images.append(full_image)
            
        # Stack all trial images into a single batch
        output_numpy = np.stack(final_images) # (N, Chans * n_freqs, Samples)
        
        # Add a channel dimension for the ViT (grayscale)
        output_numpy = np.expand_dims(output_numpy, axis=1) # (N, 1, H, W)
        
        output_tensor = torch.from_numpy(output_numpy).float().to(device)
        
        return output_tensor

# --- 2. Vision Transformer (ViT) Classifier ---

class EEGViTClassifier(torch.nn.Module):
    """
    Architecture:
    EEG pkl -> WaveletImageTransform -> ViT -> Classifier
    """
    def __init__(self, chans=122, samples=1651, cls=39, vit_model='vit_tiny_patch16_224', dropout=0.25, fmin=1, fmax=8, n_freqs=8): # MODIFIED: Default frequencies
        super().__init__()
        self.wavelet_encoder = WaveletImageTransform(fmin=fmin, fmax=fmax, n_freqs=n_freqs)
        
        # Create a ViT model. `in_chans=1` because our image is grayscale.
        self.vit = timm.create_model(vit_model, pretrained=True, num_classes=cls, in_chans=1)
        
        # Optional: Add a dropout layer for regularization
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, Chans, Samples)
        
        # 1. Create the stacked wavelet image
        # Output shape: (N, 1, Chans * n_freqs, Samples)
        x = self.wavelet_encoder(x)
        
        # 2. Resize the image to the size ViT expects (e.g., 224x224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 3. Pass through dropout and ViT
        x = self.dropout(x)
        out = self.vit(x) # (N, num_classes)
        
        return out

# --- 3. Dataset and Training Loop ---

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
    parser = argparse.ArgumentParser(description="Train a ViT model on stacked wavelet images of EEG data.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr1', type=float, default=1e-4)
    parser.add_argument('--wd1', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_vit_stacked')
    parser.add_argument('--sub', type=str, default='01')
    parser.add_argument('--cls', type=int, default=39)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--vit_model', type=str, default='vit_tiny_patch16_224', help="e.g., vit_tiny_patch16_224, vit_small_patch16_224")
    # ADDED: Arguments for frequency range
    parser.add_argument('--fmin', type=int, default=1, help="Minimum frequency for wavelet transform.")
    parser.add_argument('--fmax', type=int, default=8, help="Maximum frequency for wavelet transform.")
    parser.add_argument('--n_freqs', type=int, default=8, help="Number of frequency bins for wavelet transform.")
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
        print("Error: No data was loaded.")
        exit()

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    trainset, validset = random_split(full_dataset, [train_size, valid_size])
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    print(f"Data loaded: {len(trainset)} training samples, {len(validset)} validation samples.")

    model = EEGViTClassifier(
        chans=122, samples=1651, cls=args.cls, vit_model=args.vit_model, dropout=args.dropout,
        fmin=args.fmin, fmax=args.fmax, n_freqs=args.n_freqs # MODIFIED: Pass frequency args to model
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

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "checkpoint_best.pt"))
            print(f"*** New best model saved with F1-score: {best_f1:.4f} ***")

    print("--- Training Complete ---")
