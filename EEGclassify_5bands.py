# import os
# import math
# import torch
# import random
# import argparse
# import numpy as np
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from sklearn.metrics import f1_score

# # It is assumed that you have the following files in the same directory:
# # 1. wavelets_5bands.py: Contains the WaveletTransform5Channel class.
# # 2. eegcnn5bands.py: Contains the modified EEGcnn class that accepts `in_channels`.
# # 3. data_imagine.py: Contains the get_dataset function to load your data.
# from eegcnn5bands import EEGcnn, PositionalEncoding
# from wavelets_5bands import WaveletTransform5Channel
# from data_imagine import get_dataset

# class EEGclassification(torch.nn.Module):
#     """
#     Single-stream: pkl → wavelet (5 bands) → CNN → Transformer → classifier
#     """
#     def __init__(self, chans=125, timestamp=165, cls=39, dropout1=0.25, dropout2=0.25, layer=2, pooling='mean', size1=8, size2=8, feel1=125, feel2=25):
#         super().__init__()
#         self.wavelet_encoder = WaveletTransform5Channel()
#         num_bands = 5  # Always 5 after grouping
#         self.eegcnn = EEGcnn(Chans=chans, kernLength1=feel1, kernLength2=feel2, F1=size1, D=size2, F2=size1*size2, P1=2, P2=5, dropoutRate=dropout1, in_channels=num_bands)
#         self.layer = layer
#         if self.layer > 0:
#             self.poscode = PositionalEncoding(size1*size2, dropout=dropout2, max_len=timestamp)
#             self.encoder = torch.nn.TransformerEncoder(
#                 torch.nn.TransformerEncoderLayer(d_model=size1*size2, nhead=max(1, size1*size2//8), dim_feedforward=4*size1*size2, batch_first=True, dropout=dropout2),
#                 num_layers=self.layer)
#         self.pooling = pooling
#         self.linear = torch.nn.Linear(timestamp*(size1*size2) if pooling is None else size1*size2, cls)

#     def forward(self, x, mask):
#         x = self.wavelet_encoder(x)  # (N, 5, 125, S)
#         x = self.eegcnn(x).permute(0, 2, 1)  # (N, time, F2)
#         if self.layer > 0:
#             x = self.poscode(x)
#             x = self.encoder(x, src_key_padding_mask=(mask.bool()==False))
#         if self.pooling == "mean":
#             x = torch.mean(x, dim=1)
#         elif self.pooling == "max":
#             x = torch.max(x, dim=1)[0]
#         else:
#             x = x.reshape(x.shape[0], -1)
#         return self.linear(x)

# # --- Main Training Script ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train a single-stream 5-band EEG classification model.")
#     parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
#     parser.add_argument('--lr1', type=float, default=5e-4, help="Learning rate for the optimizer.")
#     parser.add_argument('--wd1', type=float, default=0.01, help="Weight decay for the optimizer.")
#     parser.add_argument('--epoch', type=int, default=100, help="Total number of training epochs.")
#     parser.add_argument('--batch', type=int, default=16, help="Batch size for training and validation. Lower if you have memory issues.")
#     parser.add_argument('--checkpoint_path', type=str, default='checkpoints_5band_rgb', help="Directory to save model checkpoints.")
#     parser.add_argument('--sub', type=str, default='01', help="Subject ID to train on.")
#     parser.add_argument('--cls', type=int, default=39, help="Number of output classes.")
#     # --- Model Hyperparameters ---
#     parser.add_argument('--timestamp', type=int, default=165, help="Number of timestamps after CNN processing.")
#     parser.add_argument('--pooling', type=str, default='mean', help="Pooling strategy: 'mean', 'max', or None.")
#     parser.add_argument('--layer', type=int, default=2, help="Number of Transformer layers.")
#     parser.add_argument('--dropout1', type=float, default=0.25, help="Dropout rate for the CNN.")
#     parser.add_argument('--dropout2', type=float, default=0.25, help="Dropout rate for the Transformer.")
#     args = parser.parse_args()
#     print(args)

#     # --- Setup ---
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Create checkpoint directory if it doesn't exist
#     if not os.path.exists(args.checkpoint_path):
#         os.makedirs(args.checkpoint_path)

#     # --- Data Loading ---
#     trainset = get_dataset(subject=args.sub, mode='train')
#     validset = get_dataset(subject=args.sub, mode='test')
#     trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
#     validloader = DataLoader(validset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
#     print(f"Data loaded: {len(trainset)} training samples, {len(validset)} validation samples.")

#     # --- Model, Optimizer, and Scheduler ---
#     model = EEGclassification(
#         chans=122, timestamp=args.timestamp, cls=args.cls, dropout1=args.dropout1, 
#         dropout2=args.dropout2, layer=args.layer, pooling=args.pooling
#     ).to(device)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
#     scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(trainloader) * args.epoch)
#     criterion = torch.nn.CrossEntropyLoss()

#     # --- Training Loop ---
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

#         # --- Validation ---
#         model.eval()
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for inputs, labels, mask in validloader:
#                 inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
#                 outputs = model(inputs, mask)
#                 preds = torch.argmax(outputs, dim=1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
        
#         accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
#         macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#         print(f"Epoch {epoch+1} Validation | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

#         # --- Save Checkpoint ---
#         # Save the model at the end of every epoch for safety and retraining.
#         torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pt"))
        
#         # Save a separate checkpoint for the best performing model based on F1 score.
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

# It is assumed that you have the following files in the same directory:
# 1. wavelets_5bands.py: Contains the WaveletTransform5Channel class.
# 2. eegcnn5bands.py: Contains the modified EEGcnn class that accepts `in_channels`.
# 3. data_imagine.py: Contains the get_dataset function to load your data.
from eegcnn5bands import EEGcnn, PositionalEncoding
from wavelets_5bands import WaveletTransform5Channel
from data_imagine import get_dataset

class EEGclassification(torch.nn.Module):
    """
    Single-stream: pkl → wavelet (5 bands) → CNN → Transformer → classifier
    """
    def __init__(self, chans=122, timestamp=165, cls=39, dropout1=0.25, dropout2=0.25, layer=2, pooling='mean', size1=8, size2=8, feel1=125, feel2=25):
        super().__init__()
        self.wavelet_encoder = WaveletTransform5Channel()
        num_bands = 5  # 5 frequency bands from the wavelet transform
        self.eegcnn = EEGcnn(Chans=chans, kernLength1=feel1, kernLength2=feel2, F1=size1, D=size2, F2=size1*size2, P1=2, P2=5, dropoutRate=dropout1, in_channels=num_bands)
        self.layer = layer
        if self.layer > 0:
            self.poscode = PositionalEncoding(size1*size2, dropout=dropout2, max_len=timestamp)
            self.encoder = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model=size1*size2, nhead=max(1, (size1*size2)//8), dim_feedforward=4*size1*size2, batch_first=True, dropout=dropout2),
                num_layers=self.layer)
        self.pooling = pooling
        self.linear = torch.nn.Linear(timestamp*(size1*size2) if pooling is None else size1*size2, cls)

    def forward(self, x, mask):
        # Input x: (N, Chans, Samples)
        x = self.wavelet_encoder(x)          # -> (N, 5, Chans, Samples)
        x = self.eegcnn(x).permute(0, 2, 1)  # -> (N, time, F2)
        
        if self.layer > 0:
            x = self.poscode(x)
            x = self.encoder(x, src_key_padding_mask=(mask.bool()==False))
        
        if self.pooling == "mean":
            # Masked mean pooling to correctly average over non-padded parts
            x = torch.sum(x * mask.unsqueeze(dim=2), dim=1) / torch.sum(mask, dim=1).unsqueeze(dim=1)
        elif self.pooling == "max":
            x = torch.max(x, dim=1)[0]
        else: # Flatten if no pooling is specified
            x = x.reshape(x.shape[0], -1)
            
        return self.linear(x)

class EEGDictDataset(Dataset):
    """
    Dataset class that takes the dictionary from get_dataset and a text-to-label map
    to produce integer-labeled samples for training.
    """
    def __init__(self, data_dict, text_to_label_map):
        self.input_features = []
        self.labels = []
        
        # Process the raw data from get_dataset
        for feature, text_label in zip(data_dict["input_features"], data_dict["labels"]):
            # Convert text label to integer ID. If a sentence is not in the map, skip it.
            if text_label in text_to_label_map:
                self.input_features.append(feature)
                self.labels.append(text_to_label_map[text_label])

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        # The mask is used by the transformer to ignore padding.
        # The CNN output length is 165 by default, so we create a mask of that size.
        mask = torch.ones(165)
        
        # The label is already an integer, just convert it to a tensor.
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return self.input_features[idx], label, mask

def custom_collate(batch):
    """
    Custom collate function to correctly stack tensors from the dataset into a batch.
    This is necessary to handle the items returned by EEGDictDataset.
    """
    inputs, labels, masks = zip(*batch)
    # Stack along a new batch dimension (dim=0)
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)
    return inputs, labels, masks

# --- Main Training Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single-stream 5-band EEG classification model.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--lr1', type=float, default=5e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--wd1', type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument('--epoch', type=int, default=100, help="Total number of training epochs.")
    parser.add_argument('--batch', type=int, default=16, help="Batch size for training and validation. Lower if you have memory issues.")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_5band_rgb', help="Directory to save model checkpoints.")
    parser.add_argument('--sub', type=str, default='01', help="Subject ID to train on.")
    parser.add_argument('--cls', type=int, default=39, help="Number of output classes.")
    # --- Model Hyperparameters ---
    parser.add_argument('--timestamp', type=int, default=165, help="Number of timestamps after CNN processing.")
    parser.add_argument('--pooling', type=str, default='mean', help="Pooling strategy: 'mean', 'max', or None.")
    parser.add_argument('--layer', type=int, default=2, help="Number of Transformer layers.")
    parser.add_argument('--dropout1', type=float, default=0.25, help="Dropout rate for the CNN.")
    parser.add_argument('--dropout2', type=float, default=0.25, help="Dropout rate for the Transformer.")
    parser.add_argument('--feel1', type=int, default=20, help="Kernel length for first CNN layer.")
    parser.add_argument('--feel2', type=int, default=10, help="Kernel length for second CNN layer.")
    # --- Compatibility Arguments (not used in this script but added to prevent errors) ---
    parser.add_argument('--rand_guess', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='imagine_decode')
    parser.add_argument('--subset_ratio', type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    # --- Setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # --- Data Loading ---
    # 1. Load the text-to-label mapping from the JSON file.
    try:
        with open("./Chisco/json/textmaps.json", "r") as file:
            textmaps = json.load(file)
    except FileNotFoundError:
        print("Error: textmaps.json not found in ./Chisco/json/. Please ensure the file exists.")
        exit()

    # 2. Load the raw data dictionary (features and string labels)
    all_data_dict = get_dataset(sub=args.sub)
    
    # 3. Create the full dataset, which will map text labels to integer IDs
    full_dataset = EEGDictDataset(all_data_dict, textmaps)
    
    if len(full_dataset) == 0:
        print("Error: No data was loaded. Check that the subject ID is correct and that text labels match the textmaps.json file.")
        exit()

    # 4. Split the dataset into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    trainset, validset = random_split(full_dataset, [train_size, valid_size])
    
    # 5. Create DataLoaders with the custom collate function
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    validloader = DataLoader(validset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    print(f"Data loaded: {len(trainset)} training samples, {len(validset)} validation samples.")

    # --- Model, Optimizer, and Scheduler ---
    model = EEGclassification(
        chans=122, timestamp=args.timestamp, cls=args.cls, dropout1=args.dropout1, 
        dropout2=args.dropout2, layer=args.layer, pooling=args.pooling,
        feel1=args.feel1, feel2=args.feel2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(trainloader) * args.epoch)
    criterion = torch.nn.CrossEntropyLoss()

    # --- Training Loop ---
    print(f"--- Starting Training on {device} ---")
    best_f1 = 0.0
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for inputs, labels, mask in trainloader:
            # Move batch of data to the selected device
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{args.epoch} | Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels, mask in validloader:
                inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
                outputs = model(inputs, mask)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) if len(all_preds) > 0 else 0
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_preds) > 0 else 0
        print(f"Epoch {epoch+1} Validation | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

        # --- Save Checkpoint ---
        # Save the model at the end of every epoch for safety and retraining.
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pt"))
        
        # Save a separate checkpoint for the best performing model based on F1 score.
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "checkpoint_best.pt"))
            print(f"*** New best model saved with F1-score: {best_f1:.4f} ***")

    print("--- Training Complete ---")







