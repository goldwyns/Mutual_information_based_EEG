import os
import scipy.io as sio
import numpy as np

import os
import numpy as np
import scipy.io
from tqdm import tqdm

def load_signal_from_mat(filepath, key):
    try:
        mat_data = scipy.io.loadmat(filepath)
        signal = mat_data[key].squeeze()
        return signal
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def prepare_dataloader(features, labels, test_size=0.2, batch_size=32):
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from antropy import spectral_entropy, svd_entropy


def extract_features(signal, fs=256):
    features = []

    # --- Time-Domain Features ---
    features.append(np.mean(signal))
    features.append(np.std(signal))
    features.append(np.var(signal))
    features.append(skew(signal))
    features.append(kurtosis(signal))
    features.append(np.max(signal))
    features.append(np.min(signal))
    features.append(np.median(signal))
    features.append(np.percentile(signal, 25))  # Q1
    features.append(np.percentile(signal, 75))  # Q3

    # --- Frequency-Domain Features ---
    freqs, psd = welch(signal, fs)
    features.append(np.sum(psd))                         # Total Power
    features.append(np.sum(psd[freqs < 4]))              # Delta
    features.append(np.sum(psd[(freqs >= 4) & (freqs < 8)]))  # Theta
    features.append(np.sum(psd[(freqs >= 8) & (freqs < 12)])) # Alpha
    features.append(np.sum(psd[(freqs >= 12) & (freqs < 30)]))# Beta
    features.append(np.sum(psd[(freqs >= 30)]))               # Gamma

    # --- Entropy Features ---
    features.append(spectral_entropy(signal, sf=fs, method='welch'))
    features.append(svd_entropy(signal, order=3, delay=1, normalize=True))

    return np.array(features)
from scipy.signal import resample

def preprocess_signal(signal, fs, dataset):
    signal = signal - np.mean(signal)  # Remove DC
    if dataset == "bonn":
        # Already sampled at 173.61 Hz
        return signal
    elif dataset == "hauz":
        # Downsample to 250 if needed
        if fs != 250:
            desired_length = int(len(signal) * 250 / fs)
            signal = resample(signal, desired_length)
        return signal
    else:
        return signal
import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

def load_and_extract_features(dataset_path, dataset_name, label_map, fs):
    all_features = []
    all_labels = []

    print(f"\n--- Loading from: {dataset_path} ---")

    for label_folder, label in label_map.items():
        folder_path = os.path.join(dataset_path, label_folder)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue

        print(f"\n🔍 Processing folder: {label_folder}")
        for file in tqdm(os.listdir(folder_path), desc=f"{label_folder} files"):
            file_path = os.path.join(folder_path, file)

            if not file.endswith(".mat"):
                continue

            try:
                mat = loadmat(file_path)
                key = label_folder.lower()  # Assume same as folder name
                if key not in mat:
                    print(f"⚠️ Key '{key}' not found in {file}")
                    continue

                signal = mat[key].squeeze()
                if signal.ndim != 1:
                    print(f"⚠️ Non-1D signal in {file}, skipping")
                    continue

                # Apply preprocessing
                preprocessed = preprocess_signal(signal, fs, dataset=dataset_name)
                feats = extract_features(preprocessed, fs)

                if feats is None or len(feats) == 0:
                    print(f"⚠️ Empty features from: {file}")
                    continue

                all_features.append(feats)
                all_labels.append(label)

            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")

    if not all_features:
        raise ValueError("❌ No features extracted. Please check file keys, preprocessing, and feature functions.")

    print(f"\n✅ Loaded {len(all_features)} samples.")
    return np.array(all_features), np.array(all_labels)
import torch.nn.functional as F

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=20):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    print(f"\nAccuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    return acc, cm, report
# TemporalSNNClassifier integrated with your existing EEG pipeline
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

class TemporalSNNClassifier(nn.Module):
    def __init__(self, input_features_dim, hidden_neurons, num_classes, T, lif_params):
        super(TemporalSNNClassifier, self).__init__()
        self.T = T
        self.input_features_dim = input_features_dim

        beta = lif_params.get("beta", 0.9)
        threshold = lif_params.get("threshold", 1.0)
        spike_grad = surrogate.fast_sigmoid()

        # Layers
        self.fc1 = nn.Linear(input_features_dim, hidden_neurons)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_neurons, num_classes)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        for step in range(self.T):
            cur_input = x  # You can add noise, jitter, or encode here

            cur_input = self.fc1(cur_input)
            spk1, mem1 = self.lif1(cur_input, mem1)

            cur_input = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur_input, mem2)

            spk2_rec.append(spk2)

        spk2_rec = torch.stack(spk2_rec, dim=0)  # Shape: [T, batch, classes]
        out = spk2_rec.sum(dim=0)  # Summing over time

        return out


# Integration Example:

def initialize_model(model_type, input_size, num_classes, model_params, lif_params):
    if model_type == 'TemporalSNNClassifier':
        model = TemporalSNNClassifier(
            input_features_dim=input_size,
            hidden_neurons=model_params.get('hidden_neurons', 128),
            num_classes=num_classes,
            T=model_params.get('simulation_timesteps', 25),
            lif_params=lif_params
        )
        print(f"Initialized TemporalSNNClassifier with input_size: {input_size}, num_classes: {num_classes}")
    else:
        raise ValueError("Unsupported model type.")
    return model

# Usage:
# model_params = {"hidden_neurons": 128, "simulation_timesteps": 25}
# lif_params = {"beta": 0.9, "threshold": 1.0}
# model = initialize_model("TemporalSNNClassifier", input_size=features.shape[1], num_classes=2, model_params=model_params, lif_params=lif_params)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SNN_LIF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNN_LIF, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = nn.LeakyReLU()  # Approximating LIF behavior
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)  # LIF approx
        x = self.fc2(x)
        return x
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
features, labels = load_and_extract_features(
    dataset_path=r"D:\RESEARCH\DATABASE\Neurology_Sleep_Centre_Hauz Khas",
    dataset_name="hauz",
    label_map={"ictal": 1, "interictal": 0},
    fs=256  # adjust for your dataset
)

# Prepare loaders
train_loader, test_loader = prepare_dataloader(features, labels)

# Choose model
#model = SNN_LIF(input_size=features.shape[1], hidden_size=128, output_size=2)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#criterion = nn.CrossEntropyLoss()
#model.to(device)

model_params = {"hidden_neurons": 128, "simulation_timesteps": 25}
lif_params = {"beta": 0.9, "threshold": 1.0}
model = initialize_model("TemporalSNNClassifier", input_size=features.shape[1], num_classes=2, model_params=model_params, lif_params=lif_params)

model.to(device)


# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train
train_model(model, train_loader, optimizer, criterion, device)

# Evaluate
evaluate_model(model, test_loader, device)
