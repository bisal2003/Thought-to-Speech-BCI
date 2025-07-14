import torch
import torch.nn as nn
import math

class EEGcnn(nn.Module):
    """
    A CNN architecture based on EEGNet, modified to accept multi-channel input 
    (e.g., 3 frequency bands as RGB).
    """
    def __init__(self, Chans=122, F1=8, D=2, F2=16, kernLength1=10, kernLength2=5, P1=2, P2=5, dropoutRate=0.5, in_channels=3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, F1, (1, kernLength1), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(dropoutRate)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, kernLength2), groups=D * F1, padding='same', bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(dropoutRate)
        )

    def forward(self, input):
        hidden = self.block1(input)
        hidden = self.block2(hidden)
        output = torch.squeeze(hidden, dim=2)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)



# import torch
# import torch.nn as nn
# import math
# from torchvision import models

# class EEGEfficientNet(nn.Module):
#     """
#     EfficientNet backbone for EEG 3-band (RGB) image input.
#     Input: (N, 3, Chans, Samples)
#     Output: (N, feature_dim)
#     """
#     def __init__(self, chans=122, samples=1651, out_dim=1280, version='b0'):
#         super().__init__()
#         # Load EfficientNet backbone (b0 is smallest, b7 is largest)
#         if version == 'b0':
#             self.backbone = models.efficientnet_b0(weights=None)
#         elif version == 'b1':
#             self.backbone = models.efficientnet_b1(weights=None)
#         else:
#             raise ValueError("Only b0 and b1 supported for now.")
#         # Change input conv to accept (3, chans, samples)
#         self.backbone.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
#         # Remove classifier head, keep only features
#         self.backbone.classifier = nn.Identity()
#         self.out_dim = out_dim

#     def forward(self, x):
#         # x: (N, 3, Chans, Samples)
#         # EfficientNet expects (N, 3, H, W), so Chans=H, Samples=W
#         features = self.backbone(x)  # (N, out_dim)
#         return features

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=1500):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(1), :]
#         return self.dropout(x)


