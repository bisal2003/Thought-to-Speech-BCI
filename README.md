# Decoding Imagined Speech with Deep Learning

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
[![torch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![transformers](https://img.shields.io/badge/🤗%20Transformers-ffc107?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/index)

This repository contains a comprehensive exploration of deep learning architectures for decoding imagined speech from electroencephalography (EEG) signals. Using the **[Chisco dataset](https://www.nature.com/articles/s41597-024-04114-1)**, this project develops and evaluates various models for Brain-Computer Interfaces (BCI), with the goal of advancing communication tools for individuals with paralysis.

## 🚀 Project Goal

The primary objective is to investigate, implement, and systematically compare state-of-the-art deep learning models for classifying imagined speech from raw EEG data. By translating these neural signals into text, we aim to contribute to the development of more intuitive and effective communication technologies for people who have lost the ability to speak.

## ✨ Key Features

- **Multi-Architecture Pipeline:** Implements a wide range of models, from CNNs to Transformers.
- **Advanced Signal Processing:** Utilizes Wavelet Transforms and band-pass filtering to extract rich, meaningful features.
- **Transfer Learning:** Leverages powerful, pre-trained vision models (EfficientNet, ViT) for EEG classification.
- **End-to-End Learning:** Includes models like DeWave that learn directly from minimally processed data.
- **In-depth Analysis:** Provides Jupyter notebooks for model evaluation, including accuracy metrics, confusion matrices, and visualizations.

---

## 🧠 Architectures Explored

This project investigates several distinct architectural philosophies. Below is an overview of the current models. This section is designed to be easily updated as new approaches are developed.

*(**Tip:** You can create a diagram of the overall system using a free tool like [diagrams.net](https://app.diagrams.net/), export it as a `.png`, add it to the repository, and embed it here.)*
```
[//]: # (Placeholder for Architecture Diagram)
<p align="center">
  <img src="path/to/your/architecture_diagram.png" width="800" alt="System Architecture Diagram">
</p>
```

---

### 1. Wavelet-Based CNNs
- **Concept:** Transforms the 1D EEG time-series into 2D time-frequency scalograms using the Continuous Wavelet Transform (CWT). This creates a rich "image" where the x-axis is time, the y-axis is frequency, and the pixel intensity is the signal's power. A CNN then learns to classify these images.
- **Implementations:**
    - **3-Band Model:** Creates channels from signal properties (magnitude, phase, and magnitude difference).
    - **5-Band Model:** Creates channels from neurophysiologically-relevant frequency bands (Delta, Theta, Alpha, Beta, Gamma).
- **Code:** `wavelets_*.py`, `eegcnn*.py`

### 2. Frequency Band-Based CNNs
- **Concept:** Filters the raw EEG into canonical frequency bands and processes each band with a parallel CNN stream. This allows the model to learn specialized features for each brain rhythm before fusing them for a final classification. It's computationally efficient and highly interpretable.
- **Code:** `EEGclassify_*bands.py`

### 3. EfficientNet
- **Concept:** Reimagines EEG scalograms as images and adapts the highly efficient and powerful EfficientNet architecture. It uses compound scaling and advanced convolutional blocks to achieve state-of-the-art performance with fewer parameters, making it ideal for potential deployment.
- **Code:** `effi_cnn.py`

### 4. Vision Transformer (ViT)
- **Concept:** Applies the Transformer architecture directly to patches of the EEG spectrogram. Its self-attention mechanism is uniquely suited to capturing the long-range dependencies and global network dynamics of the brain, modeling how different regions and frequencies interact over time.
- **Code:** `EEG_VIT.py`, `VIT.ipynb`

### 5. DeWave
- **Concept:** A bespoke, hybrid architecture combining a CNN feature extractor, a Transformer encoder for temporal modeling, and a Vector Quantization (VQ) module. The VQ layer discretizes the learned representations into a "codex," improving noise robustness and interpretability.
- **Code:** `Dewave/`

### 6. *Your Next Approach Here*
- **Concept:** *(Describe the core idea of your new model.)*
- **Code:** *(Link to the relevant script(s).)*

---

## 📊 The Chisco Dataset

This project uses the **Chisco (Chinese Imagined Speech Corpus)**, a large, public dataset of high-density EEG recordings.

> The absence of imagined speech electroencephalography (EEG) datasets has constrained further research in this field. We present the *Chinese Imagined Speech Corpus* (Chisco), including over 20,000 sentences of high-density EEG recordings of imagined speech from healthy adults... We believe that Chisco represents a valuable resource for the fields of BCIs, facilitating the development of more user-friendly BCIs.

<details>
  <summary><b>Click to see original paper reproduction details</b></summary>
  
  ### **Model Configuration:**
  ```
  python -u EEGclassify.py --rand_guess 0 --lr1 5e-4 --epoch 100 --layer 1 --pooling mean --dataset imagine_decode --sub "01" --cls 39 --dropout1 0.5 --dropout2 0.5 --feel1 20 --feel2 10 --subset_ratio 1
  ```
  ### **SBATCH Parameters:**
  The original authors ran their code on a SLURM cluster. These details are provided for reference.
  ```bash
  #!/bin/zsh
  #SBATCH -p compute 
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --cpus-per-task=4
  #SBATCH --gres=gpu:a100-pcie-80gb:1
  #SBATCH --mem=100G
  source ~/.zshrc
  ```
</details>

## ⚙️ Getting Started: A Detailed Walkthrough

This section provides a detailed, step-by-step guide to setting up your environment and running the project.

### Step 1: Clone the Repository
First, you need to download the project files from GitHub. Open your terminal, navigate to the directory where you want to store the project, and run the following command:

```bash
# This command downloads the project to a new folder called "Chisco"
git clone <your-repository-url>

# Navigate into the newly created project directory
cd Chisco
```

### Step 2: Set Up a Virtual Environment
It is a best practice to use a virtual environment to manage project-specific dependencies. This prevents conflicts between different projects.

```bash
# Create a virtual environment named ".venv" in the project directory
python -m venv .venv

# Activate the virtual environment
# On macOS and Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```
After activation, you will see `(.venv)` at the beginning of your terminal prompt.

### Step 3: Install Required Libraries
Install all the necessary Python libraries using pip. The main libraries are PyTorch for deep learning, Transformers for model components, and various others for data handling and plotting.

```bash
# Install the core deep learning library
pip install torch torchvision torchaudio

# Install other essential libraries
pip install transformers scikit-learn pandas numpy matplotlib seaborn mne
```
**Pro Tip:** After installing your packages, you can create a `requirements.txt` file to make it easy for others (and your future self) to replicate the environment:
```bash
pip freeze > requirements.txt
# To install from this file later, you would just run: pip install -r requirements.txt
```

### Step 4: Download and Prepare the Dataset
The models in this repository are trained on the Chisco dataset.

1.  **Download:** Access the dataset via the link in the [original paper](https://www.nature.com/articles/s41597-024-04114-1). You will need to download the EEG data files.
2.  **Organize:** Create a `dataset/` directory in the root of the project. Inside, you should place the preprocessed data according to the structure expected by the data loaders (e.g., `dataset/derivatives/preprocessed_pkl/sub-01/...`).
3.  **Check Paths:** Verify the data paths used in the training scripts (e.g., in `data_imagine.py`) and adjust them if your structure is different.

## Usage: Training and Evaluating Models

The training scripts are designed to be run from the command line with various arguments to control the experiment.

### Running a Training Script
Here is an example of how to train the `EEGclassify_5bands.py` model.

```bash
python -u EEGclassify_5bands.py \
    --dataset imagine_decode \
    --sub "01" \
    --cls 39 \
    --lr1 5e-4 \
    --epoch 100
```

### Understanding the Arguments
- `-u`: This standard Python flag ensures that the output is unbuffered, so you see logs in real-time.
- `--dataset`: Specifies the dataset to use (e.g., `imagine_decode`).
- `--sub`: The subject ID from the Chisco dataset to train on (e.g., `"01"`).
- `--cls`: The number of classes (imagined words) to classify.
- `--lr1`: The initial learning rate for the optimizer.
- `--epoch`: The total number of training epochs.

Each script has its own set of tunable hyperparameters. To see all available options for a script, you can add a `--help` flag or inspect the `argparse` section within the script file.

### Monitoring and Checkpoints
- **Logs:** Training progress, including epoch number, loss, and accuracy, will be printed to the terminal.
- **Checkpoints:** The models are configured to save their state periodically during training. These checkpoints are saved in directories like `checkpoints_5band/`. You can use these files to resume training or for later evaluation without having to retrain.

## 📈 Results and Analysis

The quantitative and qualitative results of the model evaluations can be explored in the Jupyter notebooks (`*.ipynb`).

- **`test.ipynb`:** Contains a comprehensive, high-level report comparing the performance, strengths, and weaknesses of the different architectures explored in this project.
- **`Dewave/dewave.ipynb`:** Provides specific analysis for the DeWave model, including visualizations of the learned "codex" indices.
- **Other Notebooks (`VIT.ipynb`, `wavelets.ipynb`):** Offer focused analysis for their respective models.

To view them, run Jupyter Lab in your terminal:
```bash
jupyter lab
```

## 📜 Citation

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
  doi = {10.1038/s41597-024-04114-1}
}
```

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
