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
# from sklearn.metrics import f1_score
# import timm
# from pathlib import Path
# from tqdm import tqdm
# from torchvision import transforms

# # --- 1. Data Augmentation Transforms ---
# # Custom transform for frequency masking (horizontal bars)
# class FrequencyMasking(nn.Module):
#     def __init__(self, freq_mask_param, iid_masks=True):
#         super().__init__()
#         self.freq_mask_param = freq_mask_param
#         self.iid_masks = iid_masks

#     def forward(self, spec):
#         if self.iid_masks:
#             # Each channel gets its own mask
#             for _ in range(spec.shape[0]):
#                 spec = self.apply_mask(spec)
#         else:
#             # All channels get the same mask
#             spec = self.apply_mask(spec)
#         return spec

#     def apply_mask(self, spec):
#         v = spec.shape[1] # Number of frequency bins
#         f = int(np.random.uniform(0.0, self.freq_mask_param))
#         if f == 0:
#             return spec
#         f0 = random.randint(0, v - f)
#         spec[:, f0:f0 + f, :] = 0
#         return spec

# # Custom transform for time masking (vertical bars)
# class TimeMasking(nn.Module):
#     def __init__(self, time_mask_param, iid_masks=True):
#         super().__init__()
#         self.time_mask_param = time_mask_param
#         self.iid_masks = iid_masks

#     def forward(self, spec):
#         if self.iid_masks:
#             for _ in range(spec.shape[0]):
#                 spec = self.apply_mask(spec)
#         else:
#             spec = self.apply_mask(spec)
#         return spec

#     def apply_mask(self, spec):
#         t = spec.shape[2] # Number of time steps
#         tau = int(np.random.uniform(0.0, self.time_mask_param))
#         if tau == 0:
#             return spec
#         t0 = random.randint(0, t - tau)
#         spec[:, :, t0:t0 + tau] = 0
#         return spec

# # --- 2. The Dataset Class with Augmentations ---
# class PreprocessedEEGDataset(Dataset):
#     def __init__(self, subject_id, processed_data_root, text_to_label_map, transform=None):
#         self.image_paths = []
#         self.labels = []
#         self.transform = transform
        
#         subject_path = processed_data_root / f"sub-{subject_id}"
#         images_dir = subject_path / "images"
#         label_file = subject_path / "labels.npy"

#         if not images_dir.exists() or not label_file.exists():
#             raise FileNotFoundError(f"Processed data not found for subject {subject_id} in {subject_path}. Please run preprocess_data.py first.")

#         all_image_paths = sorted(list(images_dir.glob("*.npy")))
#         all_text_labels = np.load(label_file)

#         if len(all_image_paths) != len(all_text_labels):
#             raise ValueError("Mismatch between number of images and number of labels!")

#         for img_path, text_label in zip(all_image_paths, all_text_labels):
#             if text_label in text_to_label_map:
#                 self.image_paths.append(img_path)
#                 self.labels.append(text_to_label_map[text_label])

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         image = np.load(self.image_paths[idx])
#         image_tensor = torch.from_numpy(image) # Already (bands, H, W)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
        
#         if self.transform:
#             image_tensor = self.transform(image_tensor)
            
#         return image_tensor, label

# # --- 3. The ViT Classifier ---
# class EEGViTClassifier(torch.nn.Module):
#     def __init__(self, cls=39, vit_model='vit_small_patch16_224', dropout=0.25):
#         super().__init__()
#         self.vit = timm.create_model(vit_model, pretrained=True, num_classes=cls, in_chans=3)  # or 5 for 5 bands
#         # The dropout in timm models is often part of the head, so we rely on that.
#         # If you need more dropout, you can add it here.
#         # self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, x):
#         # x is the pre-computed image: (N, 3, H, W)
#         # The ViT model in timm handles resizing internally if the input size is not native.
#         # Forcing a resize is still good practice to ensure consistency.
#         if x.shape[-2:] != (224, 224):
#             x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
#         out = self.vit(x)
#         return out

# # --- 4. Main Training and Hyperparameter Tuning Loop ---
# def run_training(args):
#     """Wraps the training logic to be callable for hyperparameter tuning."""
#     print(f"\n--- Starting Run with Params: {args} ---")
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Create a unique directory for this run's checkpoints
#     run_name = f"vit_{args.vit_model}_lr_{args.lr1}_wd_{args.wd1}_drop_{args.dropout}"
#     checkpoint_dir = Path(args.checkpoint_path) / run_name
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         with open(args.textmaps_path, "r") as file:
#             textmaps = json.load(file)
#     except FileNotFoundError:
#         print(f"Error: textmaps.json not found at {args.textmaps_path}")
#         return 0

#     # Define data augmentation pipelines
#     train_transform = transforms.Compose([
#         transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.5),
#         FrequencyMasking(freq_mask_param=15),
#         TimeMasking(time_mask_param=30),
#     ])
    
#     # Load dataset and apply transforms
#     full_dataset = PreprocessedEEGDataset(args.sub, Path(args.processed_data_root), textmaps)
    
#     # Split indices first
#     train_size = int(0.8 * len(full_dataset))
#     valid_size = len(full_dataset) - train_size
#     train_indices, valid_indices = random_split(range(len(full_dataset)), [train_size, valid_size])

#     # Create subsets
#     train_subset = torch.utils.data.Subset(full_dataset, train_indices)
#     valid_subset = torch.utils.data.Subset(full_dataset, valid_indices)

#     # Apply transforms by wrapping the subsets in a new dataset class if needed,
#     # or by modifying the original dataset to accept transforms.
#     # Here, we re-assign the transform attribute for the training set.
#     train_subset.dataset.transform = train_transform

#     trainloader = DataLoader(train_subset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
#     validloader = DataLoader(valid_subset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
#     # FIX: Corrected variable names from trainset/validset to train_subset/valid_subset
#     print(f"Data loaded: {len(train_subset)} training images, {len(valid_subset)} validation images.")

#     model = EEGViTClassifier(cls=args.cls, vit_model=args.vit_model).to(device)
    
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
#         print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {accuracy:.4f} | Val F1: {macro_f1:.4f}")

#         if macro_f1 > best_f1:
#             best_f1 = macro_f1
#             torch.save(model.state_dict(), os.path.join(checkpoint_dir, "checkpoint_best.pt"))
#             print(f"*** New best model saved with F1-score: {best_f1:.4f} ***")

#     print(f"--- Run Complete. Best F1: {best_f1:.4f} ---")
#     return best_f1


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train or tune a ViT model on pre-processed scalogram images.")
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--epoch', type=int, default=50)
#     parser.add_argument('--batch', type=int, default=32) # Increased batch size for efficiency
#     parser.add_argument('--processed_data_root', type=str, default="./Chisco/derivatives/scalogram_images_1-8Hz")
#     parser.add_argument('--checkpoint_path', type=str, default='checkpoints_vit_tuned')
#     parser.add_argument('--sub', type=str, default='01')
#     parser.add_argument('--cls', type=int, default=39)
#     parser.add_argument('--textmaps_path', type=str, default='./Chisco/json/textmaps.json')
    
#     # Hyperparameters for tuning
#     parser.add_argument('--lr1', type=float, default=1e-4)
#     parser.add_argument('--wd1', type=float, default=1e-2)
#     parser.add_argument('--dropout', type=float, default=0.1) # Note: ViT head has its own dropout
#     parser.add_argument('--vit_model', type=str, default='vit_small_patch16_224') # Switched to 'small'
    
#     # Flag to enable tuning mode
#     parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning mode.")
    
#     args = parser.parse_args()

#     if args.tune:
#         print("--- Hyperparameter Tuning Mode Enabled ---")
#         # Define the grid of hyperparameters to search
#         learning_rates = [5e-5, 1e-4, 2e-4]
#         weight_decays = [1e-2, 5e-3]
#         vit_models = ['vit_tiny_patch16_224', 'vit_small_patch16_224'] # Example models to try
        
#         best_overall_f1 = 0
#         best_params = {}

#         for lr in learning_rates:
#             for wd in weight_decays:
#                 for model_name in vit_models:
#                     # Update args for the current run
#                     args.lr1 = lr
#                     args.wd1 = wd
#                     args.vit_model = model_name
                    
#                     # Run training with the current set of hyperparameters
#                     current_f1 = run_training(args)
                    
#                     if current_f1 > best_overall_f1:
#                         best_overall_f1 = current_f1
#                         best_params = {'lr': lr, 'wd': wd, 'model': model_name}
#                         print(f"\n>>> New Best Overall F1: {best_overall_f1:.4f} with params {best_params} <<<\n")

#         print("\n--- Hyperparameter Tuning Complete ---")
#         print(f"Best F1 Score: {best_overall_f1:.4f}")
#         print(f"Best Hyperparameters: {best_params}")
#     else:
#         print("--- Single Training Mode Enabled ---")
#         run_training(args)


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
from sklearn.metrics import f1_score
import timm
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

# --- 1. Data Augmentation Transforms ---
# Custom transform for frequency masking (horizontal bars)
class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param, iid_masks=True):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks

    def forward(self, spec):
        if self.iid_masks:
            # Each channel gets its own mask
            for _ in range(spec.shape[0]):
                spec = self.apply_mask(spec)
        else:
            # All channels get the same mask
            spec = self.apply_mask(spec)
        return spec

    def apply_mask(self, spec):
        v = spec.shape[1] # Number of frequency bins
        f = int(np.random.uniform(0.0, self.freq_mask_param))
        if f == 0:
            return spec
        f0 = random.randint(0, v - f)
        spec[:, f0:f0 + f, :] = 0
        return spec

# Custom transform for time masking (vertical bars)
class TimeMasking(nn.Module):
    def __init__(self, time_mask_param, iid_masks=True):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.iid_masks = iid_masks

    def forward(self, spec):
        if self.iid_masks:
            for _ in range(spec.shape[0]):
                spec = self.apply_mask(spec)
        else:
            spec = self.apply_mask(spec)
        return spec

    def apply_mask(self, spec):
        t = spec.shape[2] # Number of time steps
        tau = int(np.random.uniform(0.0, self.time_mask_param))
        if tau == 0:
            return spec
        t0 = random.randint(0, t - tau)
        spec[:, :, t0:t0 + tau] = 0
        return spec

# --- 2. The Dataset Class with Augmentations ---
class PreprocessedEEGDataset(Dataset):
    def __init__(self, subject_id, processed_data_root, text_to_label_map, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        subject_path = processed_data_root / f"sub-{subject_id}"
        images_dir = subject_path / "images"
        label_file = subject_path / "labels.npy"

        if not images_dir.exists() or not label_file.exists():
            raise FileNotFoundError(f"Processed data not found for subject {subject_id} in {subject_path}. Please run preprocess_data.py first.")

        all_image_paths = sorted(list(images_dir.glob("*.npy")))
        all_text_labels = np.load(label_file)

        if len(all_image_paths) != len(all_text_labels):
            raise ValueError("Mismatch between number of images and number of labels!")

        for img_path, text_label in zip(all_image_paths, all_text_labels):
            if text_label in text_to_label_map:
                self.image_paths.append(img_path)
                self.labels.append(text_to_label_map[text_label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        image_tensor = torch.from_numpy(image) # Already (bands, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label

# --- 3. The ViT Classifier ---
class EEGViTClassifier(torch.nn.Module):
    def __init__(self, cls=39, vit_model='vit_small_patch16_224', dropout=0.25):
        super().__init__()
        self.vit = timm.create_model(vit_model, pretrained=True, num_classes=cls, in_chans=3)  # or 5 for 5 bands
        # The dropout in timm models is often part of the head, so we rely on that.
        # If you need more dropout, you can add it here.
        # self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x is the pre-computed image: (N, 3, H, W)
        # The ViT model in timm handles resizing internally if the input size is not native.
        # Forcing a resize is still good practice to ensure consistency.
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        out = self.vit(x)
        return out

# --- 4. Main Training and Hyperparameter Tuning Loop ---
def run_training(args):
    """Wraps the training logic to be callable for hyperparameter tuning."""
    print(f"\n--- Starting Run with Params: {args} ---")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a unique directory for this run's checkpoints
    run_name = f"vit_{args.vit_model}_lr_{args.lr1}_wd_{args.wd1}_drop_{args.dropout}"
    checkpoint_dir = Path(args.checkpoint_path) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.textmaps_path, "r") as file:
            textmaps = json.load(file)
    except FileNotFoundError:
        print(f"Error: textmaps.json not found at {args.textmaps_path}")
        return 0

    # Define data augmentation pipelines
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.5),
        FrequencyMasking(freq_mask_param=15),
        TimeMasking(time_mask_param=30),
    ])
    
    # Load dataset and apply transforms
    full_dataset = PreprocessedEEGDataset(args.sub, Path(args.processed_data_root), textmaps)
    
    # Split indices first
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_indices, valid_indices = random_split(range(len(full_dataset)), [train_size, valid_size])

    # Create subsets
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_subset = torch.utils.data.Subset(full_dataset, valid_indices)

    # Apply transforms by wrapping the subsets in a new dataset class if needed,
    # or by modifying the original dataset to accept transforms.
    # Here, we re-assign the transform attribute for the training set.
    train_subset.dataset.transform = train_transform

    trainloader = DataLoader(train_subset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    validloader = DataLoader(valid_subset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    # FIX: Corrected variable names from trainset/validset to train_subset/valid_subset
    print(f"Data loaded: {len(train_subset)} training images, {len(valid_subset)} validation images.")

    model = EEGViTClassifier(cls=args.cls, vit_model=args.vit_model).to(device)
    
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
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {accuracy:.4f} | Val F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "checkpoint_best.pt"))
            print(f"*** New best model saved with F1-score: {best_f1:.4f} ***")

    print(f"--- Run Complete. Best F1: {best_f1:.4f} ---")
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or tune a ViT model on pre-processed scalogram images.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32) # Increased batch size for efficiency
    parser.add_argument('--processed_data_root', type=str, default="./Chisco/derivatives/scalogram_images_3bands")  # or 5bands    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_vit_tuned')
    parser.add_argument('--sub', type=str, default='01')
    parser.add_argument('--cls', type=int, default=39)
    parser.add_argument('--textmaps_path', type=str, default='./Chisco/json/textmaps.json')
    
    # Hyperparameters for tuning
    parser.add_argument('--lr1', type=float, default=1e-4)
    parser.add_argument('--wd1', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.1) # Note: ViT head has its own dropout
    parser.add_argument('--vit_model', type=str, default='vit_small_patch16_224') # Switched to 'small'
    
    # Flag to enable tuning mode
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning mode.")
    
    args = parser.parse_args()

    if args.tune:
        print("--- Hyperparameter Tuning Mode Enabled ---")
        # Define the grid of hyperparameters to search
#         learning_rates = [5e-5, 1e-4, 2e-4]
#         weight_decays = [1e-2, 5e-3]
#         vit_models = ['vit_tiny_patch16_224', 'vit_small_patch16_224'] # Example models to try
        
#         best_overall_f1 = 0
#         best_params = {}

#         for lr in learning_rates:
#             for wd in weight_decays:
#                 for model_name in vit_models:
#                     # Update args for the current run
#                     args.lr1 = lr
#                     args.wd1 = wd
#                     args.vit_model = model_name
                    
#                     # Run training with the current set of hyperparameters
#                     current_f1 = run_training(args)
                    
#                     if current_f1 > best_overall_f1:
#                         best_overall_f1 = current_f1
#                         best_params = {'lr': lr, 'wd': wd, 'model': model_name}
#                         print(f"\n>>> New Best Overall F1: {best_overall_f1:.4f} with params {best_params} <<<\n")

#         print("\n--- Hyperparameter Tuning Complete ---")
#         print(f"Best F1 Score: {best_overall_f1:.4f}")
#         print(f"Best Hyperparameters: {best_params}")
    else:
        print("--- Single Training Mode Enabled ---")
        run_training(args)