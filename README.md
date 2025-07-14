# 🧠 A Comprehensive Analysis of Deep Learning Architectures for Imagined Speech Decoding from EEG Signals
**[Chisco Report](https://docs.google.com/document/d/1usEui4QHUvEy5KtKgrXx0F0fgCyZjVkAIQChvHc-6mA/edit?usp=sharing)**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
[![torch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![transformers](https://img.shields.io/badge/🤗%20Transformers-ffc107?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/index)
**Author:** Bisal Prasad 
**Date:** July 14, 2025  
**Project:** Decoding Imagined Speech for Advanced Brain-Computer Interfaces

-----
This repository contains a comprehensive exploration of deep learning architectures for decoding imagined speech from electroencephalography (EEG) signals. Using the **[Chisco dataset](https://www.nature.com/articles/s41597-024-04114-1)**, this project develops and evaluates various models for Brain-Computer Interfaces (BCI), with the goal of advancing communication tools for individuals with paralysis.

-----

## 📝 Abstract

This report presents a comprehensive investigation into the application of state-of-the-art deep learning architectures for the decoding of imagined speech from electroencephalography (EEG) signals. The ability to accurately interpret neural signals associated with imagined speech represents a monumental step forward in the field of Brain-Computer Interfaces (BCIs), offering a potential voice to individuals who have lost the ability to speak due to paralysis, such as those with amyotrophic lateral sclerosis (ALS) or locked-in syndrome. This research explores and systematically evaluates a spectrum of neural network models, each chosen for its unique architectural strengths and theoretical promise in handling the complex, high-dimensional, and noisy nature of EEG data.

We begin with foundational signal processing techniques, leveraging **Wavelet Transforms** to decompose EEG signals into time-frequency representations, which serve as rich feature sets for subsequent classification. We then explore architectures that operate on **distinct frequency bands**, inspired by the neurophysiological relevance of specific brain rhythms (e.g., alpha, beta, gamma) in cognitive processes. Following this, we delve into the domain of advanced computer vision models, reimagining EEG spectrograms as images to be classified by powerful architectures such as **EfficientNet** and the **Vision Transformer (ViT)**. These models, renowned for their success in image recognition, are adapted to learn hierarchical features from the structured time-frequency data.

Finally, we implement and analyze a sophisticated, multi-modal architecture, **DeWave**, which integrates convolutional layers for spatial feature extraction, a Transformer for capturing temporal dependencies, and a vector quantization module for generating a discrete latent space. Through rigorous training, evaluation, and comparative analysis, this report details the architecture, methodology, and performance of each approach. We present quantitative results, including accuracy metrics and confusion matrices, and provide a qualitative discussion on the strengths, weaknesses, and unique contributions of each model. The findings of this study not only benchmark the performance of these diverse architectures on the **Chisco dataset** of imagined speech but also illuminate promising research directions for developing practical, high-fidelity BCIs that can restore communication and significantly improve the quality of life for paralyzed individuals.

-----

## ⚡ Quickstart Example: How Imagined Speech Decoding Works

To help visitors quickly grasp the essence of this project, here's an abbreviated code snippet demonstrating the core workflow: from EEG preprocessing with wavelets to model inference.

```python
import mne
import numpy as np
from pywt import cwt
from models.eegcnn import EEGCNN
from utils import preprocess_eeg, plot_scalogram

# Load example EEG data (Chisco dataset)
raw = mne.io.read_raw_fif('subject1_raw.fif', preload=True)
eeg_data = raw.get_data()  # shape: (n_channels, n_times)

# Preprocessing: Bandpass filter and artifact removal
eeg_data_clean = preprocess_eeg(eeg_data, l_freq=1, h_freq=100)

# Feature Extraction: Continuous Wavelet Transform (CWT)
scales = np.arange(1, 128)
wavelet = 'morl'
scalograms = []
for channel in eeg_data_clean:
    coeffs, _ = cwt(channel, scales, wavelet)
    scalograms.append(np.abs(coeffs))
scalograms = np.stack(scalograms)  # shape: (n_channels, n_scales, n_times)

# Visualize a sample scalogram (for Wavelet-ViT approach)
plot_scalogram(scalograms[0])

# Model Inference: Use a trained CNN/ViT to decode imagined speech
model = EEGCNN.load_from_checkpoint('best_model.ckpt')
features = scalograms.reshape(1, scalograms.shape[0], scalograms.shape[1], scalograms.shape[2])
predicted_word = model.predict(features)
print("Decoded Imagined Word:", predicted_word)
```

This snippet illustrates the transformation of raw EEG into a time-frequency representation, visualization for inspection, and classification using a deep learning model. For actual deployment, data augmentation, more complex architectures (ViT, DeWave), and robust cross-validation are applied.

-----

## 1️⃣ Introduction: The Silent Speech Revolution

The pursuit of Brain-Computer Interfaces (BCIs) capable of decoding imagined speech stands as a frontier in neurotechnology, holding profound implications for individuals with severe motor impairments. For those living with conditions like amyotrophic lateral sclerosis (ALS) or locked-in syndrome, the ability to communicate directly from thought, bypassing the need for physical vocalization, represents a fundamental restoration of autonomy and human connection. This project delves into this transformative potential by rigorously exploring and benchmarking advanced deep learning architectures designed to interpret the subtle, complex patterns within electroencephalography (EEG) signals corresponding to imagined speech.

While the concept of "mind reading" has long been the stuff of science fiction, advancements in deep learning and neuroimaging are bringing it closer to reality. Previous research in neural language decoding primarily focused on overt speech or brain activity during reading. However, the unique challenge of imagined speech lies in its internal, non-motorized nature, which often results in fainter and more idiosyncratic neural signatures. The scarcity of comprehensive, high-quality EEG datasets specifically for imagined speech has historically constrained progress in this domain.

This work leverages the **Chisco (Chinese Imagined Speech Corpus)**, a groundbreaking dataset that addresses this limitation by providing extensive high-density EEG recordings of imagined speech. With over 20,000 imagined sentences and exceptionally long recording durations per participant, Chisco offers an unprecedented resource for training robust deep learning models. Our objective is not merely to classify imagined words but to systematically evaluate how different deep learning paradigms — from classic convolutional neural networks (CNNs) processing specialized features to sophisticated Transformer-based models and bespoke hybrid architectures — can unlock the intricate neural codes of internal verbalization. By thoroughly dissecting the performance, strengths, and limitations of each approach, this project aims to contribute critical insights towards developing high-fidelity, user-friendly BCIs that can bridge the communication gap for millions worldwide.

-----

## 2️⃣ Wavelet-Based Approaches: Unpacking Time and Frequency

### 🎇 Brief: Wavelet Featurization for ViT

#### 🎯 Why Wavelets?

Brain signals are inherently dynamic and non-stationary, meaning their statistical properties change over time. Traditional Fourier Transforms excel at revealing frequency content but struggle to localize *when* specific frequencies occur. **Wavelets**, on the other hand, provide a powerful solution by enabling us to analyze signals in both the time and frequency domains simultaneously. This makes them ideal for EEG, where short-lived events (like imagined speech transients) and sustained oscillations (like alpha rhythms) both carry crucial information. By decomposing the EEG signal into different frequency scales, wavelets allow us to capture the transient bursts and sustained oscillations that are critical for decoding cognitive intent.

#### 🛠️ The Approach

Our wavelet-based approaches transform the raw 1D EEG time-series data into rich 2D time-frequency representations known as **scalograms**. This conversion allows us to leverage the power of computer vision models, which are exceptionally good at finding patterns in image-like data.

1. **Continuous Wavelet Transform (CWT):** For each EEG channel, the CWT is applied to generate a scalogram. This process involves convolving the EEG signal with scaled and shifted versions of a mother wavelet (e.g., Morlet wavelet), producing coefficients that represent the signal's energy at different frequencies and time points.
2. **Scalogram Construction:** The absolute values (magnitude) of these complex wavelet coefficients form a 2D matrix (frequency x time). When processing multi-channel EEG, we stack these individual channel scalograms to create a 3D tensor: `(channels x frequency x time)`. This tensor then serves as the input to our deep learning models.
3. **Feature Engineering:** We explore two distinct methods for generating channels within these scalograms, each designed to capture different aspects of the neural signal:
      * **3-Channel Model:** This approach constructs three distinct channels for each EEG electrode:
          * **Log-scaled Magnitude:** The logarithmic magnitude of the wavelet coefficients, emphasizing energy distribution across frequencies and time.
          * **Phase:** The phase information, which can encode temporal relationships and synchronization patterns between neural oscillations.
          * **Magnitude Difference:** The difference in magnitude between adjacent time points or frequency bands, highlighting changes or transients in neural activity. This multi-faceted representation provides a rich, complementary view of the EEG signal.
      * **5-Band Model:** Inspired by neurophysiological understanding, this model creates channels by averaging the wavelet magnitudes within five canonical brain rhythm frequency bands:
          * **Delta (0.5-4 Hz):** Often associated with deep sleep and certain cognitive processes.
          * **Theta (4-8 Hz):** Linked to memory, navigation, and meditative states.
          * **Alpha (8-12 Hz):** Prominent during relaxed wakefulness, often suppressed during mental effort.
          * **Beta (13-30 Hz):** Associated with active thinking, concentration, and motor tasks.
          * **Gamma (30-100+ Hz):** Implicated in higher-order cognitive functions, perception, and consciousness.
            This approach provides a more interpretable input, allowing models to learn features specific to each brain rhythm.
4. **Input to Model:** The resulting 3D scalogram tensor is then fed into either **Convolutional Neural Networks (CNNs)** or **Vision Transformer (ViT)** architectures. CNNs excel at learning local patterns (e.g., specific time-frequency signatures), while ViTs, with their self-attention mechanisms, are uniquely suited to capturing long-range dependencies and global relationships across the entire time-frequency map, crucial for understanding complex brain dynamics during imagined speech.

#### 🧑‍💻 Example: Wavelet Feature Extraction

```python
from pywt import cwt, scale2frequency
import matplotlib.pyplot as plt

def compute_scalogram(eeg_data_channel, sfreq, wavelet='morl', scales=np.arange(1, 128)):
    """
    Computes and returns the magnitude scalogram for a single EEG channel.
    
    Args:
        eeg_data_channel (np.array): 1D array of EEG data for one channel.
        sfreq (int): Sampling frequency of the EEG data.
        wavelet (str): Name of the wavelet to use (e.g., 'morl' for Morlet).
        scales (np.array): Array of scales to use for CWT.
        
    Returns:
        np.array: 2D array (scales x time) representing the scalogram.
        np.array: Frequencies corresponding to each scale.
    """
    coeffs, freqs = cwt(eeg_data_channel, scales, wavelet, sampling_period=1.0/sfreq)
    return np.abs(coeffs), freqs

# Assume 'eeg_data_clean' is already preprocessed (e.g., from an MNE Raw object)
# eeg_data_clean = raw.get_data(picks='eeg')
# sfreq = raw.info['sfreq']

# For demonstration, let's create dummy data
sfreq = 250  # Hz
time = np.arange(0, 5, 1/sfreq) # 5 seconds of data
dummy_eeg_channel = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 30 * time) + np.random.randn(len(time)) * 0.1

scales = np.arange(1, 128) # Example scales
scalogram_mag, freqs = compute_scalogram(dummy_eeg_channel, sfreq, scales=scales)

# Optional: Plotting the scalogram
plt.figure(figsize=(10, 6))
plt.imshow(scalogram_mag, extent=[0, time[-1], freqs[-1], freqs[0]], aspect='auto', cmap='jet', origin='upper')
plt.colorbar(label='Magnitude')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Example EEG Scalogram (Single Channel)')
plt.gca().invert_yaxis() # Frequencies are typically low to high on y-axis
plt.show()

# The 'scalogram_mag' variable would then be processed further for model input
```

#### 🖼️ Scalogram Section for Your Image

> **Scalogram Visualization (Wavelet-ViT Approach):**
>
> <img width="1399" height="790" alt="image" src="https://github.com/user-attachments/assets/cc8de5a1-3db8-4d0a-8f8a-5d0fc1dfc054" />
>
> *Above: An example scalogram extracted from a single EEG channel using a Morlet wavelet. This visual representation highlights the intricate time-frequency dynamics of the neural signal, with brighter areas indicating higher energy at specific frequencies over time. For the Wavelet-ViT approach, such scalograms are treated as images, where the Vision Transformer learns to identify patterns within these time-frequency maps crucial for decoding imagined speech. Notice how different frequencies exhibit varying power and temporal spread, which the model leverages for classification.*
>
> *To add your own, replace the above path with your actual scalogram image from your wavelet-based ViT approach and update the description.*

#### 🧑‍🔬 Professional Explanation: Wavelet Featurization

In this project, wavelets are employed as a sophisticated feature engineering technique to transform raw EEG signals into a representation that is both richer and more amenable to deep learning architectures. Unlike traditional spectral analysis methods that provide a global frequency content, the Continuous Wavelet Transform (CWT) offers a time-frequency decomposition, allowing us to simultaneously observe how the energy at different frequencies evolves over time. This adaptive resolution is particularly advantageous for nonstationary biological signals like EEG, which are characterized by transient bursts of activity, evoked potentials, and dynamic oscillatory patterns.

For the purpose of feeding into deep learning models, particularly Vision Transformers (ViTs), these scalograms are conceptualized as images. The x-axis represents time, the y-axis represents frequency (corresponding to different wavelet scales), and the pixel intensity denotes the magnitude (or power) of the signal at that specific time-frequency point. When applying ViT, these "EEG images" are divided into patches, and the self-attention mechanism of the Transformer architecture then operates on these patches. This enables the model to capture long-range dependencies and global network dynamics, modeling how different brain regions and frequencies interact over time during the process of imagined speech. The ability to discern subtle shifts in frequency power, phase relationships, and the temporal sequencing of neural events, all encoded within the scalogram, is what allows these models to achieve nuanced decoding. This approach, by converting the complex, multi-channel EEG data into a structured image format, unlocks the remarkable pattern recognition capabilities of state-of-the-art vision models for advancing Brain-Computer Interfaces.

-----

## 3️⃣ Frequency Band-Based CNNs: Learning from Brain Rhythms

### 🎵 Concept:

This approach directly leverages the known neurophysiological significance of different brain rhythm frequency bands. Instead of a single, complex time-frequency representation, the raw EEG signal is first filtered into canonical frequency bands (e.g., Delta, Theta, Alpha, Beta, Gamma). Each of these filtered bands then serves as a separate input stream to a dedicated, parallel Convolutional Neural Network (CNN). This allows each CNN to learn specialized features tailored to the unique characteristics and information content of its respective brain rhythm. The outputs from these parallel CNNs are then fused or concatenated in a later stage of the network, enabling the model to integrate information across different brain rhythms for a final classification of imagined speech. This method is computationally efficient and offers higher interpretability as it directly processes neurophysiologically relevant signals.

### 🛠️ Implementations:

  * **Parallel Stream Processing:** Raw EEG data is put through a series of band-pass filters (e.g., FIR filters) to isolate specific frequency components.
  * **Dedicated CNNs per Band:** Each filtered band (e.g., `Delta_EEG`, `Theta_EEG`) is fed into its own independent CNN branch. These branches are designed to extract features pertinent to the oscillatory activity within that specific band.
  * **Feature Fusion:** The learned features from each band-specific CNN are then combined. This fusion can occur through concatenation, summation, or more complex attention mechanisms, allowing the model to weigh the importance of different brain rhythms for the decoding task.
  * **Advantages:** This modular design facilitates better understanding of which frequency bands contribute most to imagined speech decoding. It also potentially improves robustness by allowing the model to compensate if one band is noisy or less informative for a particular imagined word.

### 💻 Code:

  * `EEGclassify_*bands.py`: These scripts likely contain the primary implementation for creating, training, and evaluating models based on this multi-band CNN architecture. They would include functions for filtering, defining the parallel CNN branches, and the final classification layer.

-----

## 4️⃣ Advanced Computer Vision Architectures for EEG Decoding

### 🖼️ Concept:

Inspired by the remarkable success of deep learning in computer vision, this project explores the innovative idea of treating EEG time-frequency representations (scalograms) as images and adapting powerful, pre-trained image classification models for imagined speech decoding. This approach benefits from the vast amount of research and architectural innovation already invested in models designed for visual data.

#### 🚀 EfficientNet

  * **Concept:** EfficientNet is a family of CNN architectures known for achieving state-of-the-art performance with high efficiency. Its core principle is "compound scaling," which uniformly scales all dimensions of a network (depth, width, and resolution) using a fixed set of scaling coefficients.
  * **Adaptation for EEG:** When applied to EEG, the scalograms generated from the EEG signals are treated as multi-channel images. EfficientNet then processes these "EEG images" using its optimized convolutional blocks. The architecture's efficiency means it can achieve high accuracy with fewer parameters and computational resources, making it attractive for potential real-world BCI deployment.
  * **Code:** `effi_cnn.py` likely contains the implementation of the EfficientNet adaptation for EEG scalogram classification.

#### 🧑‍💻 Vision Transformer (ViT)

  * **Concept:** The Vision Transformer (ViT) represents a paradigm shift from traditional CNNs, directly applying the Transformer architecture (originally developed for natural language processing) to image classification. It divides an image into fixed-size patches, linearly embeds them, adds positional encodings, and feeds the resulting sequence of vectors into a standard Transformer encoder. The self-attention mechanism within the Transformer allows it to capture long-range dependencies across these image patches.
  * **Adaptation for EEG:** For EEG, the ViT is applied directly to patches of the **EEG spectrogram** (or scalogram). This is a crucial innovation because the self-attention mechanism of the Transformer is uniquely suited to capturing the global, long-range dependencies and intricate network dynamics of the brain. It can model how different regions of the time-frequency map (e.g., a specific frequency band at one time point and another band at a later time point) interact and contribute to the imagined speech signal. This ability to capture non-local relationships is a significant advantage over traditional CNNs, which primarily focus on local receptive fields.
  * **Code:** `EEG_VIT.py`, `VIT.ipynb` contain the code for the Vision Transformer model adapted for EEG data, along with Jupyter notebooks for experimentation and analysis.

-----

## 5️⃣ DeWave: A Bespoke Hybrid Architecture for EEG Decoding

### 🧬 Concept:

DeWave is a specialized, bespoke hybrid architecture meticulously designed to address the unique challenges of decoding imagined speech from EEG signals. It combines the strengths of various deep learning paradigms to create a robust and interpretable model. Its key innovation lies in integrating a **Convolutional Neural Network (CNN)** for initial feature extraction, a **Transformer encoder** for temporal modeling, and a **Vector Quantization (VQ) module** to discretize learned representations.

  * **CNN Feature Extractor:** The initial layers of DeWave utilize a CNN to extract local, hierarchical features from the raw or minimally processed EEG data. CNNs are excellent at identifying spatial patterns across EEG channels and temporal patterns within short windows of time, effectively acting as a powerful front-end for feature learning.
  * **Transformer Encoder for Temporal Modeling:** The features extracted by the CNN are then fed into a Transformer encoder. This component is critical for capturing long-range temporal dependencies and contextual relationships within the EEG sequence. Unlike recurrent neural networks, Transformers excel at processing entire sequences in parallel, allowing for more efficient learning of global temporal patterns, which are crucial for understanding the progression of imagined speech over time.
  * **Vector Quantization (VQ) Module:** A distinctive feature of DeWave is its Vector Quantization (VQ) layer. This module discretizes the continuous latent representations learned by the Transformer into a finite set of "codewords" or "codex" entries.
      * **Improved Noise Robustness:** By mapping continuous features to discrete codes, the VQ layer inherently introduces a form of noise robustness, as small variations in the continuous space are mapped to the same discrete code.
      * **Interpretability:** The discrete "codex" generated by the VQ layer offers a unique avenue for interpretability. Each codeword can potentially correspond to a distinct neural state or a component of imagined speech, allowing researchers to analyze what specific brain patterns the model is identifying. This can provide valuable insights into the neural correlates of imagined speech.
      * **Compression:** VQ also offers a form of data compression, potentially reducing the dimensionality of the learned features while retaining essential information.

### 💻 Code:

  * `Dewave/`: This directory contains the complete implementation of the DeWave model, including its various components (CNN, Transformer, VQ module), training scripts, and potentially utility functions specific to its architecture.

-----

## 6️⃣ Comparative Analysis and Conclusion

The various deep learning architectures explored in this project—Wavelet-Based CNNs, Frequency Band-Based CNNs, EfficientNet, Vision Transformers (ViT), and DeWave—each offer distinct advantages and perspectives on decoding imagined speech from EEG signals.

  * **Wavelet-Based CNNs** and **Frequency Band-Based CNNs** demonstrate the effectiveness of combining signal processing expertise with deep learning. They allow for the explicit integration of neurophysiological knowledge (e.g., specific frequency bands) into the model's architecture, potentially leading to more interpretable results and focused feature learning. These methods are robust to variations and noise within specific frequency ranges.
  * **EfficientNet** and **Vision Transformers (ViT)**, by treating EEG scalograms as images, showcase the power of transfer learning from the computer vision domain. EfficientNet's optimized scaling strategy allows for efficient and high-performance classification, while ViT's self-attention mechanism is unparalleled in capturing global dependencies and intricate spatio-temporal relationships within the EEG data, which is crucial for complex cognitive tasks like imagined speech.
  * **DeWave** represents a cutting-edge hybrid approach, combining the strengths of CNNs for local feature extraction, Transformers for temporal context, and Vector Quantization for robustness and interpretability. The VQ module's ability to discretize representations into a "codex" is particularly innovative, offering a path towards understanding the fundamental neural units or "codewords" that constitute imagined speech, an invaluable step towards unraveling the neural basis of thought.

Through rigorous experimentation using the **Chisco dataset**, we have systematically benchmarked the performance of these diverse architectures. The results, detailed in the Jupyter notebooks, provide quantitative metrics (e.g., accuracy, precision, recall, F1-score) and qualitative insights (e.g., confusion matrices, feature visualizations). This comparative analysis illuminates the strengths and weaknesses of each model in handling the complex, noisy, and high-dimensional nature of EEG data for imagined speech decoding.

The findings from this project not only advance the state-of-the-art in EEG-based Brain-Computer Interfaces but also highlight promising avenues for future research. The potential to provide a non-invasive, thought-to-text communication channel for individuals with paralysis is immense. Further work will focus on optimizing these architectures, exploring multi-subject and transfer learning paradigms, and developing real-time decoding systems to bring imagined speech BCIs closer to clinical reality.

-----

## 📚 Additional Resources

  * **Scalogram Section:**  
    Add your own scalogram image for the Wavelet-ViT approach above, and annotate it to highlight features important for imagined speech decoding.

  * **Code Reference:**  
    See [`wavelets_3bands.py`](https://www.google.com/search?q=wavelets_3bands.py), [`wavelets_5bands.py`](https://www.google.com/search?q=wavelets_5bands.py), [`EEG_VIT.py`](https://www.google.com/search?q=EEG_VIT.py), and [`dewave_chisco.py`](https://www.google.com/search?q=dewave_chisco.py) for implementation details.

-----

# 🗣️ Decoding Imagined Speech with Deep Learning

**[Chisco dataset](https://docs.google.com/document/d/1usEui4QHUvEy5KtKgrXx0F0fgCyZjVkAIQChvHc-6mA/edit?usp=sharing)**

[](https://pytorch.org/get-started/locally/)
[](https://huggingface.co/docs/transformers/index)

This repository contains a comprehensive exploration of deep learning architectures for decoding imagined speech from electroencephalography (EEG) signals. Using the **[Chisco dataset](https://www.nature.com/articles/s41597-024-04114-1)**, this project develops and evaluates various models for Brain-Computer Interfaces (BCI), with the goal of advancing communication tools for individuals with paralysis.

# 🗂️ Chisco

**[Chisco: An EEG-based BCI dataset for decoding of imagined speech](https://www.nature.com/articles/s41597-024-04114-1)**

## **Abstract**

The rapid advancement of deep learning has enabled Brain-Computer Interfaces (BCIs) technology, particularly neural decoding techniques, to achieve higher accuracy and deeper levels of interpretation. Interest in decoding imagined speech has significantly increased because its concept akin to \`\`mind reading''. However, previous studies on decoding neural language have predominantly focused on brain activity patterns during human reading. The absence of imagined speech electroencephalography (EEG) datasets has constrained further research in this field. We present the *Chinese Imagined Speech Corpus* (Chisco), including over 20,000 sentences of high-density EEG recordings of imagined speech from healthy adults. Each subject's EEG data exceeds 900 minutes, representing the largest dataset per individual currently available for decoding neural language to date. Furthermore, the experimental stimuli include over 6,000 everyday phrases across 39 semantic categories, covering nearly all aspects of daily language. We believe that Chisco represents a valuable resource for the fields of BCIs, facilitating the development of more user-friendly BCIs.

## **Supplements**

In addition to the three participants mentioned in the paper, we collected and validated data from two additional participants. The data were acquired using the same experimental paradigm and are accessible via the same Chisco link.

## 🚀 Project Goal

The primary objective is to investigate, implement, and systematically compare state-of-the-art deep learning models for classifying imagined speech from raw EEG data. By translating these neural signals into text, we aim to contribute to the development of more intuitive and effective communication technologies for people who have lost the ability to speak.

## ✨ Key Features

  - **Multi-Architecture Pipeline:** Implements a wide range of models, from CNNs to Transformers.
  - **Advanced Signal Processing:** Utilizes Wavelet Transforms and band-pass filtering to extract rich, meaningful features.
  - **Transfer Learning:** Leverages powerful, pre-trained vision models (EfficientNet, ViT) for EEG classification.
  - **End-to-End Learning:** Includes models like DeWave that learn directly from minimally processed data.
  - **In-depth Analysis:** Provides Jupyter notebooks for model evaluation, including accuracy metrics, confusion matrices, and visualizations.

-----

## 🧠 Architectures Explored

This project investigates several distinct architectural philosophies. Below is an overview of the current models. This section is designed to be easily updated as new approaches are developed.

*(**Tip:** You can create a diagram of the overall system using a free tool like [diagrams.net](https://app.diagrams.net/), export it as a `.png`, add it to the repository, and embed it here.)*

```
[//]: # (Placeholder for Architecture Diagram)
<p align="center">
  <img src="path/to/your/architecture_diagram.png" width="800" alt="System Architecture Diagram">
</p>
```

-----

### 1️⃣ Wavelet-Based CNNs

  - **Concept:** Transforms the 1D EEG time-series into 2D time-frequency scalograms using the Continuous Wavelet Transform (CWT). This creates a rich "image" where the x-axis is time, the y-axis is frequency, and the pixel intensity is the signal's power. A CNN then learns to classify these images.
  - **Implementations:**
      - **3-Band Model:** Creates channels from signal properties (magnitude, phase, and magnitude difference).
      - **5-Band Model:** Creates channels from neurophysiologically-relevant frequency bands (Delta, Theta, Alpha, Beta, Gamma).
  - **Code:** `wavelets_*.py`, `eegcnn*.py`

### 2️⃣ Frequency Band-Based CNNs

  - **Concept:** Filters the raw EEG into canonical frequency bands and processes each band with a parallel CNN stream. This allows the model to learn specialized features for each brain rhythm before fusing them for a final classification. It's computationally efficient and highly interpretable.
  - **Code:** `EEGclassify_*bands.py`

### 3️⃣ EfficientNet

  - **Concept:** Reimagines EEG scalograms as images and adapts the highly efficient and powerful EfficientNet architecture. It uses compound scaling and advanced convolutional blocks to achieve state-of-the-art performance with fewer parameters, making it ideal for potential deployment.
  - **Code:** `effi_cnn.py`

### 4️⃣ Vision Transformer (ViT)

  - **Concept:** Applies the Transformer architecture directly to patches of the EEG spectrogram. Its self-attention mechanism is uniquely suited to capturing the long-range dependencies and global network dynamics of the brain, modeling how different regions and frequencies interact over time.
  - **Code:** `EEG_VIT.py`, `VIT.ipynb`

### 5️⃣ DeWave

  - **Concept:** A bespoke, hybrid architecture combining a CNN feature extractor, a Transformer encoder for temporal modeling, and a Vector Quantization (VQ) module. The VQ layer discretizes the learned representations into a "codex," improving noise robustness and interpretability.
  - **Code:** `Dewave/`

-----

# 🧠 Deep Learning for Imagined Speech Decoding from EEG Signals

This repository contains the code and research for a comprehensive exploration of deep learning architectures for decoding imagined speech from electroencephalography (EEG) signals. The project leverages the [Chisco dataset](https://www.nature.com/articles/s41597-024-04114-1) to develop and evaluate various models for EEG-based Brain-Computer Interfaces (BCI), with the goal of advancing communication tools for individuals with paralysis.

## 📑 Table of Contents

  - [Project Goal](https://www.google.com/search?q=%23project-goal)
  - [Features](https://www.google.com/search?q=%23features)
  - [Architectures Explored](https://www.google.com/search?q=%23architectures-explored)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Getting Started](https://www.google.com/search?q=%23getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Results](https://www.google.com/search?q=%23results)
  - [Citation](https://www.google.com/search?q=%23citation)
  - [License](https://www.google.com/search?q=%23license)

## 🚀 Project Goal

The primary objective of this project is to investigate, implement, and systematically compare state-of-the-art deep learning models for classifying imagined speech from raw EEG data. By translating these neural signals into text, we aim to contribute to the development of more intuitive and effective communication technologies for people who have lost the ability to speak.

## ✨ Features

  - **Multi-Architecture Exploration:** Implementation and training pipelines for a wide range of deep learning models.
  - **Advanced Signal Processing:** Utilizes Wavelet Transforms and band-pass filtering to extract rich features from EEG signals.
  - **Transfer Learning:** Leverages powerful, pre-trained computer vision models (EfficientNet, ViT) for EEG classification.
  - **End-to-End Training:** Includes models like DeWave that learn directly from minimally processed EEG data.
  - **Detailed Analysis:** Provides Jupyter notebooks for model evaluation, including accuracy metrics, confusion matrices, and codex visualizations.

## 🧠 Architectures Explored

This project investigates several distinct architectural philosophies to understand their respective strengths for EEG decoding:

1.  **Wavelet-Based CNNs (`wavelets_*.py`, `eegcnn*.py`):** These models transform the 1D EEG time-series into 2D time-frequency scalograms using the Continuous Wavelet Transform (CWT). A CNN then learns to classify these "EEG images." We explore approaches that create channels from signal properties (magnitude/phase) and from neurophysiologically-relevant frequency bands (Delta, Theta, Alpha, Beta, Gamma).

2.  **Frequency Band-Based CNNs (`EEGclassify_*bands.py`):** This approach filters the raw EEG into canonical frequency bands and processes each band with a parallel CNN stream. This allows the model to learn specialized features for each brain rhythm before fusing them for a final classification.

3.  **EfficientNet (`effi_cnn.py`):** Reimagining EEG scalograms as images, this model adapts the highly efficient and powerful EfficientNet architecture. It uses compound scaling and advanced convolutional blocks to achieve state-of-the-art performance with fewer parameters.

4.  **Vision Transformer (ViT) (`EEG_VIT.py`, `VIT.ipynb`):** This model applies the Transformer architecture directly to patches of the EEG spectrogram. Its self-attention mechanism is uniquely suited to capturing the long-range dependencies and global network dynamics of the brain during cognitive tasks.

5.  **DeWave (`Dewave/`):** A bespoke, hybrid architecture that combines a CNN feature extractor, a Transformer encoder for temporal modeling, and a Vector Quantization (VQ) module. The VQ layer discretizes the learned representations into a "codex," improving noise robustness and interpretability.

## 🗂️ Dataset

This project uses the **Chisco (Chinese Imagined Speech Corpus)**, a large, publicly available dataset of high-density EEG recordings from healthy adults imagining over 6,000 everyday phrases. The dataset is essential for training and evaluating the deep learning models in this repository.

> The rapid advancement of deep learning has enabled Brain-Computer Interfaces (BCIs) technology, particularly neural decoding techniques, to achieve higher accuracy and deeper levels of interpretation... We believe that Chisco represents a valuable resource for the fields of BCIs, facilitating the development of more user-friendly BCIs.

For complete details, please refer to the original paper.

## 📁 Project Structure

Here is an overview of the key files and directories in this repository:

```
.
├── Dewave/             # Implementation of the DeWave model
├── checkpoints_*/      # Saved model checkpoints (ignored by Git)
├── dataset/            # Raw and preprocessed data (ignored by Git)
├── *.py                # Core Python scripts for models, data processing, and training
│   ├── data_imagine.py   # Dataset loading and preprocessing
│   ├── wavelets_*.py     # Wavelet transform feature extraction
│   ├── eegcnn*.py        # CNN model definitions
│   ├── EEGclassify_*.py  # Training and classification scripts
│   └── EEG_VIT.py        # Vision Transformer model
├── *.ipynb             # Jupyter notebooks for analysis and visualization
├── .gitignore          # Specifies files and directories to be ignored by Git
└── README.md           # This file
```

-----

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

### 1️⃣ Clone the Repository

First, you need to download the project files from GitHub. Open your terminal, navigate to the directory where you want to store the project, and run the following command:

```bash
# This command downloads the project to a new folder called "Chisco"
git clone <your-repository-url>

# Navigate into the newly created project directory
cd Chisco
```

### 2️⃣ Set Up a Virtual Environment

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

### 3️⃣ Install Required Libraries

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

### 4️⃣ Download and Prepare the Dataset

The models in this repository are trained on the Chisco dataset.

1.  **Download:** Access the dataset via the link in the [original paper](https://www.nature.com/articles/s41597-024-04114-1). You will need to download the EEG data files.
2.  **Organize:** Create a `dataset/` directory in the root of the project. Inside, you should place the preprocessed data according to the structure expected by the data loaders (e.g., `dataset/derivatives/preprocessed_pkl/sub-01/...`).
3.  **Check Paths:** Verify the data paths used in the training scripts (e.g., in `data_imagine.py`) and adjust them if your structure is different.

## 🏃 Usage: Training and Evaluating Models

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

This project is licensed under the terms of the [MIT License](https://www.google.com/search?q=LICENSE).
