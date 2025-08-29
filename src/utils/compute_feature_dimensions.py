# src/utils/compute_features_dimensions.py

import numpy as np
from src.utils.features import extract_features
from src.utils.fix_segment import fix_segment_length  # if you put it in separate file

def compute_features_dimensions():
    fs = 200
    target_length = 400  # 2 seconds at 200Hz
    
    # Create a dummy random segment
    dummy_segment = np.random.randn(target_length)
    fixed_segment = fix_segment_length(dummy_segment, target_length=target_length)
    
    # Extract features
    features = extract_features(fixed_segment, fs=fs)
    
    print(f"Feature vector shape: {features.shape}")
    return features.shape[0]

if __name__ == "__main__":
    dim = compute_features_dimensions()
    print(f"Total Features = {dim}")

