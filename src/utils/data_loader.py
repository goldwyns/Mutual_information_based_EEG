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
        np.array: The loaded signal data, squeezed and flattened to 1D.
    """
    try:
        mat_data = sio.loadmat(filepath)

        signal = None
        # 1. Try the provided key first
        if data_key_in_mat_file and data_key_in_mat_file in mat_data:
            signal = mat_data[data_key_in_mat_file].squeeze()
        else:
            # 2. Otherwise try common epilepsy dataset keys
            available_keys = [k for k in mat_data.keys() if not k.startswith('__')]

            preferred_keys = ['data', 'interictal', 'preictal', 'ictal']
            for key in preferred_keys:
                if key in available_keys:
                    signal = mat_data[key].squeeze()
                    break

            # 3. If still nothing, fall back to last available key
            if signal is None and available_keys:
                signal = mat_data[available_keys[-1]].squeeze()
                print(f"[INFO] Using fallback key '{available_keys[-1]}' for {filepath}")

            if signal is None:
                print(f"[WARNING] No suitable data key found in {filepath}. Available keys: {available_keys}")
                return None

        # 4. Ensure signal is 1D
        if signal.ndim > 1:
            if min(signal.shape) > 1:
                print(f"[WARNING] {filepath} contains multi-dimensional data {signal.shape}. Flattening all channels.")
            signal = signal.flatten()

        return signal

    except Exception as e:
        print(f"[ERROR] Could not process {filepath}: {e}")
        return None


def load_signal_from_txt(filepath):
    """
    Loads a signal from a .txt or .TXT file.
    Assumes comma-separated values (CSV-like format) or space/tab-separated.
    Handles both 1D and 2D (multi-channel) data by flattening all channels.
    Args:
        filepath (str): Path to the .txt file.
    Returns:
        np.array: The loaded signal data, flattened to a 1D array.
    """
    try:
        # Added delimiter=',' to handle comma-separated values
        signal = np.loadtxt(filepath, dtype=np.float32, delimiter=',') 
        
        # If the loaded signal is 2D (e.g., [time_points, channels] or [channels, time_points])
        if signal.ndim > 1:
            print(f"Info: Loaded multi-channel .txt data from {filepath} with shape {signal.shape}. Flattening all channels.")
            signal = signal.flatten() # Flatten all channels into a single 1D array
        
        return signal
    except Exception as e:
        print(f"Error processing {filepath} as text file: {e}")
        return None