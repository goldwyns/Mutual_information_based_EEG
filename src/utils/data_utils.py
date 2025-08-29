import os
import numpy as np
from features import extract_features_from_signal
from utils.signal_loader import load_eeg_signal  # You will need to create this
from tqdm import tqdm

def load_and_extract_features(dataset_name, classification_task, base_paths, label_map, fs):
    """
    Loads EEG data, extracts features, and returns features and labels.
    """
    features = []
    labels = []

    class_dirs = label_map.keys()

    print(f"✅ Loading task: {dataset_name} - {classification_task}")
    for class_name in class_dirs:
        class_path = base_paths[class_name]
        class_label = label_map[class_name]

        signal_files = [
            os.path.join(class_path, fname)
            for fname in os.listdir(class_path)
            if fname.endswith(".txt") or fname.endswith(".edf")
        ]

        for file_path in tqdm(signal_files, desc=f"Processing {class_name}"):
            try:
                signal = load_eeg_signal(file_path, fs)  # Implement for .txt/.edf support
                feature_vector = extract_features_from_signal(signal, fs)
                features.append(feature_vector)
                labels.append(class_label)
            except Exception as e:
                print(f"⚠️ Skipped file {file_path} due to error: {e}")

    features = np.array(features)
    labels = np.array(labels)

    print(f"✅ Loaded {len(features)} samples.")
    return features, labels
