# import torch
# import torch.nn as nn
# import pywt
# import numpy as np
# from joblib import Parallel, delayed

# def _transform_one_channel(signal, scales, wavelet_name, freqs, sfreq):
#     """
#     Transforms a single channel signal into 5 frequency band signals,
#     preserving both real and imaginary parts (no abs, no mean).
#     Output: (10, n_samples) = 5 bands × 2 (real, imag)
#     """
#     coeffs, _ = pywt.cwt(signal, scales, wavelet_name, sampling_period=1.0/sfreq)
#     band_edges = np.arange(0, 101, 4)
#     sub_band_coeffs = []
#     for i in range(len(band_edges) - 1):
#         fmin, fmax = band_edges[i], band_edges[i+1]
#         freq_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
#         if len(freq_indices) > 0:
#             # Instead of mean, just take the first frequency in the band (or you could stack all)
#             band_coeff = coeffs[freq_indices[0], :]
#             sub_band_coeffs.append(band_coeff)
#         else:
#             sub_band_coeffs.append(np.zeros(signal.shape, dtype=np.complex64))
#     sub_band_coeffs = np.array(sub_band_coeffs) # (25, n_samples)

#     # Group into 5 bands (no mean, just concatenate real/imag)
#     delta_band = sub_band_coeffs[0]
#     theta_band = sub_band_coeffs[1]
#     alpha_band = sub_band_coeffs[2]
#     beta_band = sub_band_coeffs[3]
#     gamma_band = sub_band_coeffs[8]

#     # Stack real and imag for each band
#     bands_real_imag = np.stack([
#         delta_band.real, delta_band.imag,
#         theta_band.real, theta_band.imag,
#         alpha_band.real, alpha_band.imag,
#         beta_band.real, beta_band.imag,
#         gamma_band.real, gamma_band.imag
#     ], axis=0)  # (10, n_samples)
#     return bands_real_imag


# class WaveletTransform5Channel(nn.Module):
#     """
#     A non-trainable PyTorch module that converts a raw EEG signal batch 
#     into a 5-channel "image-like" tensor representing 5 frequency bands.
    
#     Input: (N, Chans, Samples)
#     Output: (N, 5, Chans, Samples) -> 5 is the number of frequency bands.
#     """
#     def __init__(self, sfreq=500, wavelet_name='morl', n_jobs=-1):
#         super().__init__()
#         self.sfreq = sfreq
#         self.wavelet_name = wavelet_name
#         self.n_jobs = n_jobs
        
#         # Define a high-resolution list of frequencies to analyze for the CWT.
#         # This covers the full range from 1 Hz to 100 Hz.
#         self.freqs = np.linspace(1, 100, 150)
#         # Convert frequencies to scales for the CWT function
#         self.scales = pywt.central_frequency(self.wavelet_name) * self.sfreq / self.freqs
        
#         # This layer is a fixed transformation, not meant to be trained.
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         """
#         Applies the wavelet transformation in parallel.
#         """
#         device = x.device
#         batch_size, n_channels, n_samples = x.shape
        
#         # Move tensor to CPU and convert to NumPy for pywt and joblib
#         x_cpu = x.detach().cpu().numpy()
        
#         # Flatten the batch and channel dimensions to create a list of 1D signals
#         # that can be processed in parallel. Shape changes from (N, C, S) to (N*C, S).
#         signals_to_process = x_cpu.reshape(-1, n_samples)
        
#         # Use joblib to run the transformation on all signals across multiple CPU cores.
#         processed_bands_list = Parallel(n_jobs=self.n_jobs)(
#             delayed(_transform_one_channel)(
#                 signal, self.scales, self.wavelet_name, self.freqs, self.sfreq
#             ) for signal in signals_to_process
#         )
        
#         # The result is a list of (5, n_samples) arrays.
#         # Stack them into a single numpy array.
#         output_numpy = np.stack(processed_bands_list) # Shape: (N*C, 5, S)
        
#         # Reshape back to include batch and channel dimensions.
#         # (N*C, 5, S) -> (N, C, 5, S)
#         output_numpy = output_numpy.reshape(batch_size, n_channels, 10, n_samples)
        
#         # Transpose to get the desired output shape (N, 5, C, S) for the CNN.
#         output_numpy = output_numpy.transpose(0, 2, 1, 3)
        
#         # Convert back to a tensor on the original device.
#         output_tensor = torch.from_numpy(output_numpy).float().to(device)
        
#         return output_tensor





import torch
import torch.nn as nn
import pywt
import numpy as np
from joblib import Parallel, delayed

def _transform_one_channel(signal, scales, wavelet_name, freqs, sfreq):
    """
    Transforms a single channel signal into 5 frequency band signals.
    The process is:
    1. Perform CWT to get a time-frequency representation.
    2. Calculate power (magnitude) from the complex coefficients.
    3. Define 25 fine-grained sub-bands from 0-100 Hz (4 Hz each).
    4. Average power within each sub-band.
    5. Group the 25 sub-bands into 5 main neurological bands.
    """
    # 1. Perform CWT. The output `coeffs` is complex-valued.
    coeffs, _ = pywt.cwt(signal, scales, wavelet_name, sampling_period=1.0/sfreq)
    
    # 2. Calculate power by taking the absolute value (magnitude). This is standard practice
    # to get a real-valued representation of energy for the CNN.
    power = coeffs # Shape: (n_scales, n_samples)

    # 3. Define 25 sub-bands (0-4, 4-8, ..., 96-100 Hz) and calculate their average power.
    band_edges = np.arange(0, 101, 4)
    sub_band_powers = []
    for i in range(len(band_edges) - 1):
        fmin, fmax = band_edges[i], band_edges[i+1]
        # Find frequency indices for the current sub-band
        freq_indices = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if len(freq_indices) > 0:
            # Average power across the frequencies in this sub-band
            mean_power = np.mean(power[freq_indices, :], axis=0)
            sub_band_powers.append(mean_power)
        else:
            # If no frequencies from our CWT analysis fall in this range, append a zero signal
            sub_band_powers.append(np.zeros_like(signal))
    
    sub_band_powers = np.array(sub_band_powers) # Shape: (25, n_samples)

    # 4. Group into 5 main bands by averaging the relevant sub-bands
    # Delta: 0-4 Hz (uses sub-band 0)
    # Theta: 4-8 Hz (uses sub-band 1)
    # Alpha: 8-12 Hz (uses sub-band 2)
    # Beta: 12-32 Hz (uses sub-bands 3, 4, 5, 6, 7)
    # Gamma: 32-100 Hz (uses sub-bands 8 to 24)
    delta_band = sub_band_powers[0]
    theta_band = sub_band_powers[1]
    alpha_band = sub_band_powers[2]
    beta_band = np.mean(sub_band_powers[3:8, :], axis=0)
    gamma_band = np.mean(sub_band_powers[8:, :], axis=0)

    # 5. Stack the 5 final band signals.
    return np.stack([delta_band, theta_band, alpha_band, beta_band, gamma_band], axis=0)


class WaveletTransform5Channel(nn.Module):
    """
    A non-trainable PyTorch module that converts a raw EEG signal batch 
    into a 5-channel "image-like" tensor representing 5 frequency bands.
    
    Input: (N, Chans, Samples)
    Output: (N, 5, Chans, Samples) -> 5 is the number of frequency bands.
    """
    def __init__(self, sfreq=500, wavelet_name='morl', n_jobs=-1):
        super().__init__()
        self.sfreq = sfreq
        self.wavelet_name = wavelet_name
        self.n_jobs = n_jobs
        
        # Define a high-resolution list of frequencies to analyze for the CWT.
        # This covers the full range from 1 Hz to 100 Hz.
        self.freqs = np.linspace(1, 100, 150)
        # Convert frequencies to scales for the CWT function
        self.scales = pywt.central_frequency(self.wavelet_name) * self.sfreq / self.freqs
        
        # This layer is a fixed transformation, not meant to be trained.
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Applies the wavelet transformation in parallel.
        """
        device = x.device
        batch_size, n_channels, n_samples = x.shape
        
        # Move tensor to CPU and convert to NumPy for pywt and joblib
        x_cpu = x.detach().cpu().numpy()
        
        # Flatten the batch and channel dimensions to create a list of 1D signals
        # that can be processed in parallel. Shape changes from (N, C, S) to (N*C, S).
        signals_to_process = x_cpu.reshape(-1, n_samples)
        
        # Use joblib to run the transformation on all signals across multiple CPU cores.
        processed_bands_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one_channel)(
                signal, self.scales, self.wavelet_name, self.freqs, self.sfreq
            ) for signal in signals_to_process
        )
        
        # The result is a list of (5, n_samples) arrays.
        # Stack them into a single numpy array.
        output_numpy = np.stack(processed_bands_list) # Shape: (N*C, 5, S)
        
        # Reshape back to include batch and channel dimensions.
        # (N*C, 5, S) -> (N, C, 5, S)
        output_numpy = output_numpy.reshape(batch_size, n_channels, 5, n_samples)
        
        # Transpose to get the desired output shape (N, 5, C, S) for the CNN.
        output_numpy = output_numpy.transpose(0, 2, 1, 3)
        
        # Convert back to a tensor on the original device.
        output_tensor = torch.from_numpy(output_numpy).float().to(device)
        
        return output_tensor

