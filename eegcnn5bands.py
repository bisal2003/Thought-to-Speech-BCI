# MIT License
# 
# Copyright (c) 2024 Zihan Zhang, Yi Zhao, Harbin Institute of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import math

class EEGcnn(nn.Module):
    """
    A CNN architecture based on EEGNet, modified to accept multi-channel input 
    (e.g., 5 frequency bands).
    """
    # Add `in_channels` to the constructor to specify the number of input feature maps.
    def __init__(self, Chans=122, F1=8, D=2, F2=16, kernLength1=125, kernLength2=25, P1=2, P2=5, dropoutRate=0.5, in_channels=5):
        super().__init__()
        
        # This block performs the temporal and spatial convolutions.
        self.block1 = nn.Sequential(
            # The first convolution now accepts `in_channels` (e.g., 5) instead of 1.
            # It learns temporal patterns within each band and channel.
            nn.Conv2d(in_channels, F1, (1, kernLength1), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            
            # This is the spatial convolution. It learns patterns across the 122 EEG channels.
            nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(dropoutRate)
        )
        
        # This block further refines the features and reduces dimensionality.
        self.block2 = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, kernLength2), groups=D * F1, padding='same', bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(dropoutRate)
        )

    def forward(self, input):
        """
        Forward pass for the CNN.
        Args:
            input (torch.Tensor): An "image-like" tensor of shape (N, in_channels, Chans, Samples).
        Returns:
            torch.Tensor: A feature tensor of shape (N, F2, Samples_out).
        """
        # The input is already the correct 4D shape, so we no longer need to unsqueeze it.
        hidden = self.block1(input)
        hidden = self.block2(hidden)
        
        # Squeeze the height dimension (which is now 1) to get a 3D tensor for the Transformer.
        output = torch.squeeze(hidden, dim=2)
        return output

# The PositionalEncoding module remains unchanged.
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
        # x has shape (N, seq_len, d_model)
        # self.pe has shape (max_len, d_model)
        # We add the positional encoding to the input tensor.
        # The positional encoding `self.pe` will be broadcasted across the batch dimension.
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


