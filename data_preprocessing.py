# src/utils/data_preprocessing.py
import numpy as np
from scipy.signal import resample, butter, iirnotch, filtfilt

def preprocess_signal(signal, fs, dataset_name):
    """
    Applies common preprocessing steps to the signal.
    - Removes DC offset.
    - Applies dataset-specific processing (e.g., resampling, filtering).

    Args:
        signal (np.array): The raw signal data.
        fs (float): Original sampling frequency of the signal.
        dataset_name (str): Name of the dataset ('bonn' or 'hauz').

    Returns:
        tuple: (np.array: The preprocessed signal, float: The effective sampling frequency after preprocessing).
    """
    signal = signal - np.mean(signal) # Remove DC offset

    effective_fs = fs # Initialize with original fs

    if dataset_name == "bonn":
        # Standard Bandpass Filter (e.g., 0.5-30 Hz for EEG)
        lowcut = 0.5
        highcut = 30.0
        nyquist = 0.5 * fs
        # Ensure frequency bounds are valid
        low = max(0.01, lowcut / nyquist) # Min bound to avoid issues with very low frequencies
        high = min(0.99, highcut / nyquist) # Max bound to avoid issues near Nyquist
        order = 4 # Filter order

        b, a = butter(order, [low, high], btype='band')
        signal = filtfilt(b, a, signal)

        # Notch Filter for powerline noise (e.g., 50 Hz in India/Europe, 60 Hz in US)
        # Assuming 50 Hz for your region (India)
        notch_freq = 50.0
        Q = 30.0 # Quality factor
        if notch_freq < nyquist: # Only apply if notch frequency is below Nyquist
            b_notch, a_notch = iirnotch(notch_freq, Q, fs)
            signal = filtfilt(b_notch, a_notch, signal)
        else:
            print(f"Warning: Notch frequency {notch_freq} Hz is >= Nyquist frequency {nyquist} Hz for Bonn. Skipping notch filter.")

        # No resampling for Bonn (stays at 173.61 Hz)
        effective_fs = fs

    elif dataset_name == "hauz":
        # Downsample to 250 Hz if current fs is not 250
        desired_fs = 250.0 # This matches your updated configs.config
        if fs != desired_fs:
            desired_length = int(len(signal) * desired_fs / fs)
            signal = resample(signal, desired_length)
            effective_fs = desired_fs # Update effective fs after resampling

    return signal, effective_fs