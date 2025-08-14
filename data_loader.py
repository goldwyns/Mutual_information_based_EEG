# src/utils/data_loader.py
import scipy.io as sio
import numpy as np
import os

def load_signal_from_mat(filepath, data_key_in_mat_file=None):
    """
    Loads a signal from a .mat file. Attempts to find the data key dynamically
    if data_key_in_mat_file is not provided or not found.
    Args:
        filepath (str): Path to the .mat file.
        data_key_in_mat_file (str, optional): The expected key (variable name) inside the .mat file.
                                            If None or not found, tries to infer.
    Returns:
        np.array: The loaded signal data, squeezed to remove singleton dimensions.
    """
    try:
        mat_data = sio.loadmat(filepath)

        signal = None
        # 1. Try the provided key first
        if data_key_in_mat_file and data_key_in_mat_file in mat_data:
            signal = mat_data[data_key_in_mat_file].squeeze()
        else:
            # 2. If provided key fails or is None, try common keys or infer
            available_keys = [k for k in mat_data.keys() if not k.startswith('__')]

            if 'data' in available_keys:
                signal = mat_data['data'].squeeze()
            elif 'interictal' in available_keys:
                signal = mat_data['interictal'].squeeze()
            elif 'preictal' in available_keys:
                signal = mat_data['preictal'].squeeze()
            elif available_keys:
                signal = mat_data[available_keys[-1]].squeeze()
            else:
                print(f"Warning: No suitable data key found in {filepath}. Available keys: {mat_data.keys()}")
                return None

        if signal is not None:
            if signal.ndim > 1 and min(signal.shape) > 1:
                print(f"Warning: {filepath} contains multi-dimensional data {signal.shape} after squeeze. Taking the first channel/row.")
                signal = signal[0] if signal.shape[0] < signal.shape[1] else signal[:, 0]
                signal = signal.flatten()
            elif signal.ndim > 1:
                signal = signal.flatten()
            return signal
        else:
            return None

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def load_signal_from_txt(filepath):
    """
    Loads a signal from a .txt or .TXT file.
    Assumes a space-separated or tab-separated values file.
    Args:
        filepath (str): Path to the .txt file.
    Returns:
        np.array: The loaded signal data.
    """
    try:
        signal = np.loadtxt(filepath, dtype=np.float32)
        return signal
    except Exception as e:
        print(f"Error processing {filepath} as text file: {e}")
        return None