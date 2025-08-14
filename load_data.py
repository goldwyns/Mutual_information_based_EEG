# In src/utils/load_data.py

import os
import numpy as np
import scipy.io as sio
# Moved this import to the top as good practice
from .features import extract_features

def load_bonn_dataset(base_paths, label_map):
    features = []
    labels = []
    # Iterate only through the label_names that are present in the current label_map
    for label_name in label_map.keys():
        path = base_paths.get(label_name)
        if path is None:
            print(f"Warning: Path for label '{label_name}' not found in base_paths. Skipping.")
            continue

        for file in os.listdir(path):
            if file.endswith(".txt"):
                data = np.loadtxt(os.path.join(path, file))
                features.append(data) # Append the 1D signal
                labels.append(label_map[label_name])
    # IMPORTANT: Return a list of 1D arrays or a 2D array where each row is a sample
    # np.array(features) will create a (num_samples, signal_length) array if lengths are consistent
    return np.array(features), np.array(labels)

def load_hauz_dataset(base_paths, label_map):
    features = []
    labels = []
    # Iterate only through the label_names that are present in the current label_map
    for label_name in label_map.keys():
        path = base_paths.get(label_name)
        if path is None:
            print(f"Warning: Path for label '{label_name}' not found in base_paths. Skipping.")
            continue

        for file in os.listdir(path):
            if file.endswith(".mat"):
                mat = sio.loadmat(os.path.join(path, file))
                key = list(mat.keys())[-1]
                data = mat[key].squeeze()
                features.append(data) # Append the 1D signal
                labels.append(label_map[label_name])
    # IMPORTANT: Return a list of 1D arrays or a 2D array where each row is a sample
    return np.array(features), np.array(labels)


def load_and_extract_features(dataset_path, dataset_name, label_map, fs):
    if dataset_name == "bonn":
        raw_data_segments, labels = load_bonn_dataset(dataset_path, label_map)
    elif dataset_name == "hauz":
        raw_data_segments, labels = load_hauz_dataset(dataset_path, label_map)
    else:
        raise ValueError("Unknown dataset")

    extracted_features_list = []
    # Iterate over each individual 1D signal segment
    for signal_segment in raw_data_segments: # <--- CRITICAL CHANGE HERE
        # Pass each 1D signal_segment to extract_features
        features_for_segment = extract_features(signal_segment, fs)
        extracted_features_list.append(features_for_segment)

    # Convert the list of feature arrays into a single 2D NumPy array
    return np.array(extracted_features_list), np.array(labels)