# Chisco 

**[Chisco: An EEG-based BCI dataset for decoding of imagined speech](https://www.nature.com/articles/s41597-024-04114-1)**

## **Abstract**
The rapid advancement of deep learning has enabled Brain-Computer Interfaces (BCIs) technology, particularly neural decoding techniques, to achieve higher accuracy and deeper levels of interpretation. Interest in decoding imagined speech has significantly increased because its concept akin to ``mind reading''. However, previous studies on decoding neural language have predominantly focused on brain activity patterns during human reading. The absence of imagined speech electroencephalography (EEG) datasets has constrained further research in this field. We present the *Chinese Imagined Speech Corpus* (Chisco), including over 20,000 sentences of high-density EEG recordings of imagined speech from healthy adults. Each subject's EEG data exceeds 900 minutes, representing the largest dataset per individual currently available for decoding neural language to date. Furthermore, the experimental stimuli include over 6,000 everyday phrases across 39 semantic categories, covering nearly all aspects of daily language. We believe that Chisco represents a valuable resource for the fields of BCIs, facilitating the development of more user-friendly BCIs.

## **Supplements**
In addition to the three participants mentioned in the paper, we collected and validated data from two additional participants. The data were acquired using the same experimental paradigm and are accessible via the same Chisco link.

## **Reproducing the Paper Results**

To reproduce the results from the paper, please follow the configurations provided below:

### **Model Configuration：**
```
python -u EEGclassify.py --rand_guess 0 --lr1 5e-4 --epoch 100 --layer 1 --pooling mean --dataset imagine_decode --sub "01" --cls 39 --dropout1 0.5 --dropout2 0.5 --feel1 20 --feel2 10 --subset_ratio 1
```
### **SBATCH Parameters:**
We ran our code on a SLURM cluster server. The following details may not be critical for reproducing the results presented in the paper, but they are provided here for reference if needed.
```bash
#!/bin/zsh
#SBATCH -p compute 
#SBATCH -N 1                                  # Request 1 node
#SBATCH --ntasks-per-node=1                   # 1 process per node
#SBATCH --cpus-per-task=4                     # Use 4 CPU cores per task
#SBATCH --gres=gpu:a100-pcie-80gb:1           # Request 1 A100 GPU
#SBATCH --mem=100G                            # Allocate 100GB memory
source ~/.zshrc
```

# Deep Learning for Imagined Speech Decoding from EEG Signals

This repository contains the code and research for a comprehensive exploration of deep learning architectures for decoding imagined speech from electroencephalography (EEG) signals. The project leverages the [Chisco dataset](https://www.nature.com/articles/s41597-024-04114-1) to develop and evaluate various models for EEG-based Brain-Computer Interfaces (BCI), with the goal of advancing communication tools for individuals with paralysis.

## Table of Contents
- [Project Goal](#project-goal)
- [Features](#features)
- [Architectures Explored](#architectures-explored)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Project Goal

The primary objective of this project is to investigate, implement, and systematically compare state-of-the-art deep learning models for classifying imagined speech from raw EEG data. By translating these neural signals into text, we aim to contribute to the development of more intuitive and effective communication technologies for people who have lost the ability to speak.

## Features

- **Multi-Architecture Exploration:** Implementation and training pipelines for a wide range of deep learning models.
- **Advanced Signal Processing:** Utilizes Wavelet Transforms and band-pass filtering to extract rich features from EEG signals.
- **Transfer Learning:** Leverages powerful, pre-trained computer vision models (EfficientNet, ViT) for EEG classification.
- **End-to-End Training:** Includes models like DeWave that learn directly from minimally processed EEG data.
- **Detailed Analysis:** Provides Jupyter notebooks for model evaluation, including accuracy metrics, confusion matrices, and codex visualizations.

## Architectures Explored

This project investigates several distinct architectural philosophies to understand their respective strengths for EEG decoding:

1.  **Wavelet-Based CNNs (`wavelets_*.py`, `eegcnn*.py`):** These models transform the 1D EEG time-series into 2D time-frequency scalograms using the Continuous Wavelet Transform (CWT). A CNN then learns to classify these "EEG images." We explore approaches that create channels from signal properties (magnitude/phase) and from neurophysiologically-relevant frequency bands (Delta, Theta, Alpha, Beta, Gamma).

2.  **Frequency Band-Based CNNs (`EEGclassify_*bands.py`):** This approach filters the raw EEG into canonical frequency bands and processes each band with a parallel CNN stream. This allows the model to learn specialized features for each brain rhythm before fusing them for a final classification.

3.  **EfficientNet (`effi_cnn.py`):** Reimagining EEG scalograms as images, this model adapts the highly efficient and powerful EfficientNet architecture. It uses compound scaling and advanced convolutional blocks to achieve state-of-the-art performance with fewer parameters.

4.  **Vision Transformer (ViT) (`EEG_VIT.py`, `VIT.ipynb`):** This model applies the Transformer architecture directly to patches of the EEG spectrogram. Its self-attention mechanism is uniquely suited to capturing the long-range dependencies and global network dynamics of the brain during cognitive tasks.

5.  **DeWave (`Dewave/`):** A bespoke, hybrid architecture that combines a CNN feature extractor, a Transformer encoder for temporal modeling, and a Vector Quantization (VQ) module. The VQ layer discretizes the learned representations into a "codex," improving noise robustness and interpretability.

## Dataset

This project uses the **Chisco (Chinese Imagined Speech Corpus)**, a large, publicly available dataset of high-density EEG recordings from healthy adults imagining over 6,000 everyday phrases. The dataset is essential for training and evaluating the deep learning models in this repository.

> The rapid advancement of deep learning has enabled Brain-Computer Interfaces (BCIs) technology, particularly neural decoding techniques, to achieve higher accuracy and deeper levels of interpretation... We believe that Chisco represents a valuable resource for the fields of BCIs, facilitating the development of more user-friendly BCIs.

For complete details, please refer to the original paper.

## Project Structure

Here is an overview of the key files and directories in this repository:

```
.
├── Dewave/               # Implementation of the DeWave model
├── checkpoints_*/         # Saved model checkpoints (ignored by Git)
├── dataset/              # Raw and preprocessed data (ignored by Git)
├── *.py                  # Core Python scripts for models, data processing, and training
│   ├── data_imagine.py   # Dataset loading and preprocessing
│   ├── wavelets_*.py     # Wavelet transform feature extraction
│   ├── eegcnn*.py        # CNN model definitions
│   ├── EEGclassify_*.py  # Training and classification scripts
│   └── EEG_VIT.py        # Vision Transformer model
├── *.ipynb               # Jupyter notebooks for analysis and visualization
├── .gitignore            # Specifies files and directories to be ignored by Git
└── README.md             # This file
```

## Getting Started

Follow these instructions to set up the project environment and run the models.

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for training)
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    (It is recommended to create a `requirements.txt` file with all dependencies)
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers scikit-learn pandas numpy matplotlib seaborn mne
    ```

4.  **Download the Dataset:**
    Download the Chisco dataset and place the preprocessed data in a `dataset/` directory (or update the paths in the training scripts).

## Usage

The training scripts can be run from the command line. For example, to train the 5-band classification model on subject "01":

```bash
python -u EEGclassify_5bands.py \
    --dataset imagine_decode \
    --sub "01" \
    --cls 39 \
    --lr1 5e-4 \
    --epoch 100 \
    --layer 1 \
    --pooling mean \
    --dropout1 0.5 \
    --dropout2 0.5
```

Each training script has its own set of command-line arguments. Please refer to the respective files for more details.

## Results

Detailed analysis, model performance metrics, confusion matrices, and visualizations can be found in the Jupyter notebooks (`*.ipynb`). The `test.ipynb` notebook, in particular, contains a comprehensive report comparing the performance and characteristics of all explored architectures.

## Citation

If you use the Chisco dataset in your research, please cite the original publication:

```bibtex
@article{Zhang2024,
  author = {Zihan Zhang and Xiao Ding and Yu Bao and Yi Zhao and Xia Liang and Bing Qin and Ting Liu},
  title = {Chisco: An EEG-based BCI dataset for decoding of imagined speech},
  journal = {Scientific Data},
  volume = {11},
  number = {1},
  pages = {1265},
  year = {2024},
  month = {November},
  doi = {10.1038/s41597-024-04114-1},
  url = {https://doi.org/10.1038/s41597-024-04114-1},
  issn = {2052-4463}
}
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).
