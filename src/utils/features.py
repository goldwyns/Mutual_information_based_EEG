# src/utils/features.py
import numpy as np
from scipy.signal import welch, stft # Added stft for STFT features
from scipy.stats import skew, kurtosis
import pywt # Added PyWavelets for wavelet features


def compute_pseudo_covariance(segment):
    """Computes pseudo-covariance."""
    segment = np.asarray(segment).flatten()
    if len(segment) == 0: return 0.0
    return np.mean(np.abs(segment) * np.sign(segment))

def compute_shannon_entropy(segment):
    """Computes Shannon Entropy."""
    segment = np.asarray(segment).flatten()
    if len(segment) == 0: return 0.0
    
    # Handle segments with all zeros to prevent division by zero or log(0)
    total_abs_sum = np.sum(np.abs(segment))
    if total_abs_sum == 0:
        return 0.0 # If all values are zero, entropy is 0

    probabilities = np.abs(segment) / total_abs_sum
    # Add a small epsilon to log argument to avoid log(0) for any zero probabilities
    return -np.sum(probabilities * np.log2(probabilities + 1e-8))

def compute_inter_energy_kurtosis_covariance(segment):
    """Computes a combined metric of kurtosis and variance."""
    segment = np.asarray(segment).flatten()
    # Kurtosis requires at least 4 samples
    if len(segment) < 4: return 0.0
    return kurtosis(segment) * np.var(segment)

def extract_wavelet_features(segment, wavelet='db4', level=5):
    """Extracts features based on DWT coefficients (mean absolute value)."""
    segment = np.asarray(segment).flatten()
    if len(segment) == 0: 
        # Return zeros matching expected output size if segment is empty
        return np.zeros(level + 1) # Assumes level+1 coeffs (approx + details)

    # Determine maximum possible decomposition level for the given segment length
    max_level = pywt.dwt_max_level(len(segment), pywt.Wavelet(wavelet).dec_len)
    actual_level = min(level, max_level)

    if actual_level <= 0: # If segment is too short for meaningful decomposition
        # Return zeros if no decomposition can occur, maintaining expected size
        return np.zeros(level + 1)
    
    coeffs = pywt.wavedec(segment, wavelet, level=actual_level)
    wavelet_features_list = [np.mean(np.abs(c)) for c in coeffs]
    
    # Pad with zeros if actual_level was less than requested 'level' to maintain feature vector size
    if len(wavelet_features_list) < (level + 1):
        wavelet_features_list.extend([0.0] * ((level + 1) - len(wavelet_features_list)))
    
    return np.array(wavelet_features_list)

def compute_stft_features(segment, fs, window='hann'):
    """Computes mean magnitude of STFT frequency bins."""
    segment = np.asarray(segment).flatten()
    # STFT requires at least 2 samples, typically more for meaningful results
    if len(segment) < 2: 
        # Return zeros of an approximate expected size, crucial for consistent feature vector length
        # For nperseg=len(segment), the number of frequency bins is int(len(segment)/2) + 1
        return np.zeros(int(len(segment)/2)+1 if len(segment)>0 else 1)
    
    # --- nperseg FIX APPLIED HERE ---
    nperseg_dynamic = len(segment) 
    
    try:
        # noverlap set to 50% of nperseg, common practice
        f, t, Zxx = stft(segment, fs, window=window, nperseg=nperseg_dynamic, noverlap=nperseg_dynamic // 2)
        
        # If Zxx is empty (e.g., very short segment or STFT internal error), return zeros
        if Zxx.size == 0: 
            return np.zeros(int(nperseg_dynamic/2)+1)
            
        stft_features = np.mean(np.abs(Zxx), axis=1) # Mean magnitude across time bins for each freq bin
        
        # Optional: Normalization - often helpful for SNN inputs
        if np.max(stft_features) > 0:
            stft_features = stft_features / (np.max(stft_features) + 1e-8)
        
        return stft_features
    except ValueError as e:
        print(f"STFT Error for segment length {len(segment)}: {e}")
        return np.zeros(int(nperseg_dynamic/2)+1) # Return zeros on error


def compute_spectral_energy_ratios(segment, fs):
    """Computes power ratios in standard EEG bands relative to total power."""
    segment = np.asarray(segment).flatten()
    if len(segment) < 2: return np.zeros(5) # Return zeros for 5 bands if segment too short

    # --- nperseg FIX APPLIED HERE ---
    nperseg_dynamic = len(segment)
    
    try:
        # noverlap set to 50% of nperseg, common practice
        f, Pxx = welch(segment, fs, nperseg=nperseg_dynamic, noverlap=nperseg_dynamic // 2)
    except ValueError as e:
        print(f"Welch Error for segment length {len(segment)}: {e}")
        return np.zeros(5) # Return zeros on error for 5 bands

    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
             "beta": (13, 30), "gamma": (30, 70)}
    total_power = np.sum(Pxx)

    if total_power == 0:
        return np.zeros(len(bands)) # Return array of zeros if no power

    energy_ratios = np.array([np.sum(Pxx[(f >= low) & (f < high)]) / total_power
                              for band, (low, high) in bands.items()])
    return energy_ratios


def fix_segment_length(segment, target_length=400):
    """
    Ensures all EEG segments have the same length (for uniform feature extraction).
    
    Args:
        segment (np.array): Raw EEG segment.
        target_length (int): Desired length in samples (e.g., 400 for 2s at 200Hz).
    
    Returns:
        np.array: Fixed-length segment (truncated or zero-padded).
    """
    segment = np.asarray(segment).flatten()
    if len(segment) > target_length:
        return segment[:target_length]  # truncate
    elif len(segment) < target_length:
        return np.pad(segment, (0, target_length - len(segment)), 'constant')  # zero-pad
    else:
        return segment

# --- Main Feature Extraction Function ---
def extract_features(segment, fs=250, wavelet='db4', wavelet_level=5, stft_window='hann'):
    """
    Extracts a comprehensive set of time-domain, frequency-domain,
    wavelet, and spectral ratio features from an EEG segment.
    
    Args:
        segment (np.array): The EEG segment (1D array).
        fs (int): Sampling frequency of the EEG data.
        wavelet (str): Wavelet to use for DWT features (e.g., 'db4').
        wavelet_level (int): Decomposition level for wavelet features.
        stft_window (str): Window function for STFT (e.g., 'hann').

    Returns:
        np.array: A concatenated array of all extracted features.
    """
    segment = np.asarray(segment).flatten()

    # if len(segment) < 2:
    #     print(f"Warning: Segment too short for robust feature extraction (length={len(segment)}). Returning zeros.")
    #     # The calculation of the return size below is incorrect if segment length varies.
    #     # It would need to be a fixed size, which contradicts dynamic segment length.
    #     expected_stft_features_length = int(segment.shape[0] / 2) + 1 if segment.shape[0] > 0 else 1
    #     return np.zeros(8 + 3 + 7 + (wavelet_level + 1) + expected_stft_features_length + 5)
    # --- END REMOVE ---

    features = []

    # --- Original Time-domain features ---
    features.append(np.mean(segment))
    features.append(np.std(segment))
    features.append(np.max(segment))
    features.append(np.min(segment))
    features.append(np.ptp(segment)) # Peak-to-peak
    features.append(np.sqrt(np.mean(segment**2))) # RMS
    features.append(skew(segment))
    features.append(kurtosis(segment))

    # --- ADDING NEW TIME-DOMAIN FEATURES ---
    features.append(compute_pseudo_covariance(segment))
    features.append(compute_shannon_entropy(segment))
    features.append(compute_inter_energy_kurtosis_covariance(segment))

    # --- Frequency-domain features (from your original code, now with nperseg fix applied) ---
    nperseg_val = len(segment) # Use the actual segment length for nperseg
    # Welch requires nperseg >= 4 for stable results, though 2 is minimum for execution
    # Ensure nperseg_val is at least 2 for basic spectrum calculation
    if nperseg_val < 2: # Changed from 4 to 2, as 2 is the theoretical minimum for PSD
        # If segment is less than 2, cannot compute welch or stft, other functions will return 0.0 or zeros
        # This case is actually implicitly handled by individual feature functions
        # For simplicity, if len(segment) is 0 or 1, many features will be 0.0.
        # It's better to let `compute_spectral_energy_ratios` handle this for its return.
        pass # Do nothing, let the try-except handle potential errors in welch
    
    try:
        f, Pxx = welch(segment, fs=fs, nperseg=nperseg_val, noverlap=nperseg_val // 2) # Added noverlap for consistency
        
        # Define frequency bands (example bands for EEG, adjust as needed)
        delta_band = (0.5, 4)
        theta_band = (4, 8)
        alpha_band = (8, 12)
        beta_band = (12, 30)
        gamma_band = (30, 80) 

        def bandpower(freqs, psd, band):
            idx_band = np.where((freqs >= band[0]) & (freqs <= band[1]))
            if len(idx_band[0]) > 0:
                return np.trapz(psd[idx_band], freqs[idx_band])
            return 0.0

        features.append(bandpower(f, Pxx, delta_band))
        features.append(bandpower(f, Pxx, theta_band))
        features.append(bandpower(f, Pxx, alpha_band))
        features.append(bandpower(f, Pxx, beta_band))
        features.append(bandpower(f, Pxx, gamma_band))

        # Total power (from all available frequencies)
        features.append(np.trapz(Pxx, f))

        # Peak frequency (frequency with highest power)
        if len(Pxx) > 0:
            features.append(f[np.argmax(Pxx)])
        else:
            features.append(0.0)

    except ValueError as e:
        print(f"Error during Welch PSD calculation: {e}. Segment length: {len(segment)}, fs: {fs}, nperseg: {nperseg_val}. Returning zeros for freq-domain features.")
        # Append zeros for all frequency-domain features if calculation fails
        for _ in range(7): # 5 bands + total power + peak freq
            features.append(0.0)

    # --- ADDING WAVELET FEATURES ---
    wavelet_features = extract_wavelet_features(segment, wavelet=wavelet, level=wavelet_level)
    features.extend(wavelet_features) # Use extend for arrays to flatten them into the main list

    # --- ADDING STFT FEATURES ---
    stft_features = compute_stft_features(segment, fs=fs, window=stft_window)
    features.extend(stft_features)

    # --- ADDING SPECTRAL ENERGY RATIOS (normalized band powers) ---
    spectral_ratios = compute_spectral_energy_ratios(segment, fs=fs)
    features.extend(spectral_ratios)

    # Ensure all features are floats, handle potential NaNs and Infs
    features_array = np.array(features, dtype=np.float32)
    features_array[np.isnan(features_array)] = 0.0 # Replace NaN with 0
    features_array[np.isinf(features_array)] = 0.0 # Replace Inf with 0

    return features_array