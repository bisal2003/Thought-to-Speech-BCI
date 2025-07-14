# Chisco_DeWave/
# ├── data/
# │   └── chisco_preprocessed/      # Your preprocessed EEG files (e.g., .pkl, .npy)
# ├── dewave/
# │   ├── __init__.py
# │   ├── encoder.py                # CNN/Conformer + Transformer encoder for EEG
# │   ├── vq.py                     # Vector Quantization (VQ-VAE) codex module
# │   ├── bart_decoder.py           # Wrapper for BART/LLM decoder
# │   ├── dataset.py                # Dataset loader for Chisco
# │   ├── train.py                  # Training loop (stage 1 & 2)
# │   ├── evaluate.py               # BLEU/ROUGE evaluation
# │   └── utils.py                  # Helper functions
# ├── scripts/
# │   ├── run_train.sh
# │   └── run_eval.sh
# ├── main.py                       # Entry point
# └── README.md

# Code Citations

## License: unknown
https://github.com/CMU-Robotics-Club/RobOrchestra/blob/f4d26b2a08ac7f78fff9f9e51ca69d001606d42f/Software/Test%20Code/transformer.py

```
):
        super
```


## License: unknown
https://github.com/LLNL/lbann/blob/1db91a2ba387c722c36cdc9a7f0ad325296c8965/applications/nlp/transformer/pytorch-reference/train_transformer_translation.py

```
):
        super
```


## License: MIT
https://github.com/fkwai/geolearn/blob/1130bc2a59339609ec4d7e4169e054b77797a79c/hydroDL/model/layers.py

```
):
        super
```


## License: unknown
https://github.com/CMU-Robotics-Club/RobOrchestra/blob/f4d26b2a08ac7f78fff9f9e51ca69d001606d42f/Software/Test%20Code/transformer.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000
```


## License: unknown
https://github.com/LLNL/lbann/blob/1db91a2ba387c722c36cdc9a7f0ad325296c8965/applications/nlp/transformer/pytorch-reference/train_transformer_translation.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000
```


## License: MIT
https://github.com/fkwai/geolearn/blob/1130bc2a59339609ec4d7e4169e054b77797a79c/hydroDL/model/layers.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000
```


## License: unknown
https://github.com/sabj17/MasterThesis/blob/3d87eb804829a081b1d64b42ed79e6e600e12165/src/embedders.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000
```


## License: unknown
https://github.com/yurkoi/transformersbase_nlp/blob/6ac3ad88ec15e47a32d1ece466abaae579f1dcd0/encoder-decoder/encoder_decoder.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000
```


## License: unknown
https://github.com/ManlioWu/ESTAG/blob/bc9a99ebfc1f2723cd5cb3655600868a02cae60a/models/model.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000
```


## License: unknown
https://github.com/CMU-Robotics-Club/RobOrchestra/blob/f4d26b2a08ac7f78fff9f9e51ca69d001606d42f/Software/Test%20Code/transformer.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term
```


## License: unknown
https://github.com/LLNL/lbann/blob/1db91a2ba387c722c36cdc9a7f0ad325296c8965/applications/nlp/transformer/pytorch-reference/train_transformer_translation.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term
```


## License: MIT
https://github.com/fkwai/geolearn/blob/1130bc2a59339609ec4d7e4169e054b77797a79c/hydroDL/model/layers.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term
```


## License: unknown
https://github.com/sabj17/MasterThesis/blob/3d87eb804829a081b1d64b42ed79e6e600e12165/src/embedders.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term
```


## License: unknown
https://github.com/yurkoi/transformersbase_nlp/blob/6ac3ad88ec15e47a32d1ece466abaae579f1dcd0/encoder-decoder/encoder_decoder.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term
```


## License: unknown
https://github.com/ManlioWu/ESTAG/blob/bc9a99ebfc1f2723cd5cb3655600868a02cae60a/models/model.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term
```


## License: unknown
https://github.com/CMU-Robotics-Club/RobOrchestra/blob/f4d26b2a08ac7f78fff9f9e51ca69d001606d42f/Software/Test%20Code/transformer.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```


## License: unknown
https://github.com/LLNL/lbann/blob/1db91a2ba387c722c36cdc9a7f0ad325296c8965/applications/nlp/transformer/pytorch-reference/train_transformer_translation.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```


## License: MIT
https://github.com/fkwai/geolearn/blob/1130bc2a59339609ec4d7e4169e054b77797a79c/hydroDL/model/layers.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```


## License: unknown
https://github.com/sabj17/MasterThesis/blob/3d87eb804829a081b1d64b42ed79e6e600e12165/src/embedders.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```


## License: unknown
https://github.com/yurkoi/transformersbase_nlp/blob/6ac3ad88ec15e47a32d1ece466abaae579f1dcd0/encoder-decoder/encoder_decoder.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```


## License: unknown
https://github.com/ManlioWu/ESTAG/blob/bc9a99ebfc1f2723cd5cb3655600868a02cae60a/models/model.py

```
):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```


## License: unknown
https://github.com/zerkvii/empathetic_dialogue_research/blob/25005fe1090a3b9ac637a9ef282debe5e385949e/test_ppl.py

```
seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda
```


## License: unknown
https://github.com/langfield/spred/blob/cda06089ff26dc46f9c6e47f07648dc7cbd20b31/spred/gpst/train.py

```
seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda
```


## License: unknown
https://github.com/zerkvii/empathetic_dialogue_research/blob/25005fe1090a3b9ac637a9ef282debe5e385949e/test_ppl.py

```
seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #
```


## License: unknown
https://github.com/langfield/spred/blob/cda06089ff26dc46f9c6e47f07648dc7cbd20b31/spred/gpst/train.py

```
seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #
```

