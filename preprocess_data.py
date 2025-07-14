


# import os
# import pickle
# import numpy as np
# import pywt
# from pathlib import Path
# import argparse
# from tqdm import tqdm

# def create_scalogram_image(signal_matrix, scales, wavelet_name, sfreq):
#     """
#     Creates a single stacked scalogram image from a multi-channel EEG signal.
#     """
#     list_of_scalograms = []
#     for channel_signal in signal_matrix:
#         coeffs, _ = pywt.cwt(channel_signal, scales, wavelet_name, sampling_period=1.0/sfreq)
#         scalogram = np.abs(coeffs)
#         list_of_scalograms.append(scalogram)
    
#     stacked_scalogram = np.vstack(list_of_scalograms)
#     return stacked_scalogram

# def process_subject(subject_id, raw_data_root, output_root, scales, wavelet_name, sfreq):
#     """
#     Processes all .pkl files for a single subject, creating scalogram images
#     and saving them with labels read directly from the .pkl files.
#     """
#     print(f"--- Processing Subject: {subject_id} ---")
    
#     subject_raw_path = raw_data_root / f"sub-{subject_id}"
#     subject_output_path = output_root / f"sub-{subject_id}"
#     subject_output_path.mkdir(parents=True, exist_ok=True)
    
#     # Find all pkl files for the subject
#     pkl_files = sorted(list(subject_raw_path.glob("**/*_eeg.pkl")))
    
#     if not pkl_files:
#         print(f"Warning: No .pkl files found for subject {subject_id} in {subject_raw_path}")
#         return

#     all_images = []
#     all_labels = []

#     for pkl_path in tqdm(pkl_files, desc=f"Subject {subject_id}"):
#         with open(pkl_path, "rb") as f:
#             trials = pickle.load(f)
        
#         for trial in trials:
#             # CORRECTED: Get the label from the 'text' key inside the pkl
#             label = trial['text'].strip()
#             signal_matrix = np.squeeze(trial['input_features'])
            
#             # Create the image
#             image = create_scalogram_image(signal_matrix, scales, wavelet_name, sfreq)
            
#             all_images.append(image)
#             all_labels.append(label)

#     # Save all images and labels for the subject
#     if all_images:
#         np.save(subject_output_path / "images.npy", np.array(all_images, dtype=np.float32))
#         np.save(subject_output_path / "labels.npy", np.array(all_labels))
#         print(f"Saved {len(all_images)} images and labels for subject {subject_id}.")
#     else:
#         print(f"No data processed for subject {subject_id}.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Pre-process EEG data into scalogram images.")
#     parser.add_argument('--raw_data_root', type=str, default="./Chisco/derivatives/preprocessed_pkl", help="Root directory of raw .pkl files.")
#     parser.add_argument('--output_root', type=str, default="./Chisco/derivatives/scalogram_images_1-8Hz", help="Directory to save processed images.")
#     parser.add_argument('--subjects', nargs='+', default=["01", "02", "03"], help="List of subject IDs to process.")
#     parser.add_argument('--fmin', type=int, default=1, help="Minimum frequency for CWT.")
#     parser.add_argument('--fmax', type=int, default=8, help="Maximum frequency for CWT.")
#     parser.add_argument('--n_freqs', type=int, default=8, help="Number of frequency bins for CWT.")
#     args = parser.parse_args()

#     # --- Define CWT Parameters ---
#     SFREQ = 500
#     WAVELET_NAME = 'morl'
#     freqs = np.linspace(args.fmin, args.fmax, args.n_freqs)
#     scales = pywt.central_frequency(WAVELET_NAME) * SFREQ / freqs

#     # Process each subject
#     for sub_id in args.subjects:
#         process_subject(
#             subject_id=sub_id,
#             raw_data_root=Path(args.raw_data_root),
#             output_root=Path(args.output_root),
#             scales=scales,
#             wavelet_name=WAVELET_NAME,
#             sfreq=SFREQ
#         )
    
#     print("\n--- Pre-processing complete! ---")





import os
import pickle
import numpy as np
import pywt
from pathlib import Path
import argparse
from tqdm import tqdm

def create_scalogram_image(signal_matrix, scales, wavelet_name, sfreq):
    """Creates a single stacked scalogram image from a multi-channel EEG signal."""
    list_of_scalograms = []
    for channel_signal in signal_matrix:
        coeffs, _ = pywt.cwt(channel_signal, scales, wavelet_name, sampling_period=1.0/sfreq)
        scalogram = np.abs(coeffs)
        list_of_scalograms.append(scalogram)
    return np.vstack(list_of_scalograms)

def process_subject(subject_id, raw_data_root, output_root, scales, wavelet_name, sfreq):
    """
    Processes all .pkl files for a subject, saving each trial's image and label individually
    to avoid running out of memory.
    """
    print(f"--- Processing Subject: {subject_id} ---")
    
    subject_raw_path = raw_data_root / f"sub-{subject_id}"
    subject_output_path = output_root / f"sub-{subject_id}"
    # Create a dedicated folder for the images
    images_output_dir = subject_output_path / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    pkl_files = sorted(list(subject_raw_path.glob(f"**/*task-imagine*_eeg.pkl")))
    
    if not pkl_files:
        print(f"Warning: No 'task-imagine' .pkl files found for subject {subject_id} in {subject_raw_path}")
        return
    
    count=0

    all_labels = []
    trial_counter = 0
    for pkl_path in tqdm(pkl_files, desc=f"Subject {subject_id}"):
        with open(pkl_path, "rb") as f:
            trials = pickle.load(f)
        sentence=0
        print(count)
        count+=1
        for trial in trials:
            label = trial['text'].strip()
            signal_matrix = np.squeeze(trial['input_features'])
            print(sentence)
            sentence+=1
            # Create the image
            image = create_scalogram_image(signal_matrix, scales, wavelet_name, sfreq)
            
            # **MEMORY-EFFICIENT SAVE**: Save image immediately
            image_filename = f"trial_{trial_counter:04d}.npy"
            np.save(images_output_dir / image_filename, image.astype(np.float32))
            
            # Collect the label (this uses very little memory)
            all_labels.append(label)
            trial_counter += 1

    # Save the list of all labels once at the end
    if all_labels:
        np.save(subject_output_path / "labels.npy", np.array(all_labels))
        print(f"Saved {len(all_labels)} individual images and one labels file for subject {subject_id}.")
    else:
        print(f"No data processed for subject {subject_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process EEG data into scalogram images.")
    parser.add_argument('--raw_data_root', type=str, default="./Chisco/derivatives/preprocessed_pkl", help="Root directory of raw .pkl files.")
    parser.add_argument('--output_root', type=str, default="./Chisco/derivatives/scalogram_images_1-8Hz", help="Directory to save processed images.")
    parser.add_argument('--subjects', nargs='+', default=["01", "02", "03"], help="List of subject IDs to process.")
    parser.add_argument('--fmin', type=int, default=1, help="Minimum frequency for CWT.")
    parser.add_argument('--fmax', type=int, default=8, help="Maximum frequency for CWT.")
    parser.add_argument('--n_freqs', type=int, default=8, help="Number of frequency bins for CWT.")
    args = parser.parse_args()

    SFREQ = 500
    WAVELET_NAME = 'morl'
    freqs = np.linspace(args.fmin, args.fmax, args.n_freqs)
    scales = pywt.central_frequency(WAVELET_NAME) * SFREQ / freqs

    for sub_id in args.subjects:
        process_subject(
            subject_id=sub_id,
            raw_data_root=Path(args.raw_data_root),
            output_root=Path(args.output_root),
            scales=scales,
            wavelet_name=WAVELET_NAME,
            sfreq=SFREQ
        )
    
    print("\n--- Pre-processing complete! ---")