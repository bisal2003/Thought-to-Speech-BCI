import torch
import torch.nn as nn
import pywt
import numpy as np
from joblib import Parallel, delayed

def _transform_one_channel(signal, scales, wavelet_name, freqs, sfreq):
    """
    Transforms a single channel signal into 3 frequency band signals (Delta, Theta, Alpha).
    The process is:
    1. Perform CWT to get a time-frequency representation.
    2. Calculate power (magnitude) from the complex coefficients.
    3. Define 25 fine-grained sub-bands from 0-100 Hz (4 Hz each).
    4. Average power within each sub-band.
    5. Group the 25 sub-bands into 3 main neurological bands (Delta, Theta, Alpha).
    """
    coeffs, _ = pywt.cwt(signal, scales, wavelet_name, sampling_period=1.0/sfreq)
    power = coeffs  # Use magnitude for image-like input

    band_edges = np.array([0, 4, 8, 12])  # Delta:0-4, Theta:4-8, Alpha:8-12 Hz
    sub_band_powers = []
    for i in range(len(band_edges) - 1):
        fmin, fmax = band_edges[i], band_edges[i+1]
        freq_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if len(freq_indices) > 0:
            mean_power = np.mean(power[freq_indices, :], axis=0)
            sub_band_powers.append(mean_power)
        else:
            sub_band_powers.append(np.zeros_like(signal))
    sub_band_powers = np.array(sub_band_powers) # Shape: (3, n_samples)
    return sub_band_powers

class WaveletTransform3Channel(nn.Module):
    """
    Converts a raw EEG signal batch into a 3-channel tensor representing Delta, Theta, Alpha bands.
    Input: (N, Chans, Samples)
    Output: (N, 3, Chans, Samples)
    """
    def __init__(self, sfreq=500, wavelet_name='morl', n_jobs=-1):
        super().__init__()
        self.sfreq = sfreq
        self.wavelet_name = wavelet_name
        self.n_jobs = n_jobs
        self.freqs = np.linspace(1, 12, 30)  # Only up to 12 Hz for 3 bands
        self.scales = pywt.central_frequency(self.wavelet_name) * self.sfreq / self.freqs
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        device = x.device
        batch_size, n_channels, n_samples = x.shape
        x_cpu = x.detach().cpu().numpy()
        signals_to_process = x_cpu.reshape(-1, n_samples)
        processed_bands_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one_channel)(
                signal, self.scales, self.wavelet_name, self.freqs, self.sfreq
            ) for signal in signals_to_process
        )
        output_numpy = np.stack(processed_bands_list) # (N*C, 3, S)
        output_numpy = output_numpy.reshape(batch_size, n_channels, 3, n_samples)
        output_numpy = output_numpy.transpose(0, 2, 1, 3) # (N, 3, C, S)
        output_tensor = torch.from_numpy(output_numpy).float().to(device)
        return output_tensor
