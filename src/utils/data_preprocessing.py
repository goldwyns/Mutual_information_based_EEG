# src/utils/data_preprocessing.py
import numpy as np
from scipy.signal import resample, butter, iirnotch, filtfilt

def preprocess_signal(signal, fs, dataset_name, target_fs=None):
    """
    Applies common preprocessing steps to the signal.
    - Removes DC offset.
    - Applies dataset-specific processing (e.g., resampling, filtering).
    - Optionally resamples to a target sampling frequency for consistency.

    Args:
        signal (np.array): The raw signal data.
        fs (float): Original sampling frequency of the signal.
        dataset_name (str): Name of the dataset ('bonn' or 'hauz').
        target_fs (float, optional): Desired final sampling frequency. If None, dataset-specific defaults apply.

    Returns:
        tuple: (np.array: The preprocessed signal, float: The effective sampling frequency after preprocessing).
    """
    # --- Remove DC offset ---
    signal = signal - np.mean(signal)

    # Effective sampling freq (initially same as input)
    effective_fs = fs

    if dataset_name == "bonn":
        # --- Bandpass filter (0.5–30 Hz typical for EEG) ---
        lowcut = 0.5
        highcut = 30.0
        nyquist = 0.5 * fs
        low = max(0.01, lowcut / nyquist)
        high = min(0.99, highcut / nyquist)
        order = 4
        b, a = butter(order, [low, high], btype="band")
        signal = filtfilt(b, a, signal)

        # --- Notch filter at 50 Hz (if valid) ---
        notch_freq = 50.0
        Q = 30.0
        if notch_freq < nyquist:
            b_notch, a_notch = iirnotch(notch_freq, Q, fs)
            signal = filtfilt(b_notch, a_notch, signal)

    elif dataset_name == "hauz":
        # By default Hauz is 250 Hz, but we allow override below
        desired_fs = 250.0
        if fs != desired_fs and target_fs is None:
            desired_length = int(len(signal) * desired_fs / fs)
            signal = resample(signal, desired_length)
            effective_fs = desired_fs

    # --- Final uniform resampling if target_fs is requested ---
    if target_fs is not None and effective_fs != target_fs:
        desired_length = int(len(signal) * target_fs / effective_fs)
        signal = resample(signal, desired_length)
        effective_fs = target_fs

    return signal, effective_fs


def fix_segment_length(signal, target_len):
    """
    Ensures that the EEG segment has exactly `target_len` samples.
    - If longer, it trims the segment.
    - If shorter, it zero-pads at the end.

    Args:
        signal (np.array): Input EEG signal segment.
        target_len (int): Desired fixed length.

    Returns:
        np.array: Signal of shape (target_len,)
    """
    cur_len = len(signal)

    if cur_len > target_len:
        # Trim from the center
        start = (cur_len - target_len) // 2
        return signal[start:start + target_len]

    elif cur_len < target_len:
        # Pad with zeros at the end
        pad_width = target_len - cur_len
        return np.pad(signal, (0, pad_width), mode="constant")

    else:
        # Already correct length
        return signal
