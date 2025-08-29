import torch
import time
import psutil

# Correct relative imports inside src/
from models.snn_lif import SNN_LIF
from models.snn_cnn_lif import SNN_CNN_LIF
from models.temporal_snn import TemporalSNNClassifier

from src.utils.model_initializer import initialize_model
from src.utils.evaluator import evaluate_model
from src.utils.trainer import train_model
from src.utils.logger import log_experiment_result as log_results

from src.utils.data_loader import prepare_dataloader
from src.utils.load_data import load_and_extract_features # New import for this function

# If config is outside src/, do this:
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.config import DATASET_PATHS, SAMPLING_FREQ, LABEL_MAPPINGS, TASKS_TO_RUN

import torch
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hardcode a single task to mimic test.py ---
dataset = "hauz"
classification = "ictal_interictal" # Or whatever matches your label_map key
label_map = {"ictal": 1, "interictal": 0} # Direct map for hauz
fs = 256 # SAMPLING_FREQ[dataset]

# THIS IS THE FIX: Get the dictionary of paths for the "hauz" dataset from your config
base_path_dict = DATASET_PATHS[dataset] # This will now be a dictionary, not a string

print(f"\n🔁 Running Task: {classification.upper()}")

# Load and extract features
features, labels = load_and_extract_features(
    dataset_path=base_path_dict, # Pass the dictionary here
    dataset_name=dataset,
    label_map=label_map,
    fs=fs
)

# Prepare data loaders
train_loader, test_loader = prepare_dataloader(features, labels)

# --- Initialize and run TemporalSNNClassifier directly, no loop ---
model_type = "TemporalSNNClassifier" # Hardcode for single run
print(f"\n⚙️ Model: {model_type}")

model_params = {"hidden_neurons": 128, "simulation_timesteps": 25}
lif_params = {
    "beta": 0.9,
    "threshold": 1.0,
    "simulation_timesteps": model_params["simulation_timesteps"]
}

input_size = features.shape[1]
num_classes = len(set(labels))

model = initialize_model(model_type, input_size, num_classes, model_params, lif_params)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# --- TEMPORARY DEBUGGING BLOCK: Inspect data and model output ---
try:
    first_batch_data, first_batch_targets = next(iter(train_loader))
    first_batch_data = first_batch_data.to(device)
    first_batch_targets = first_batch_targets.to(device)

    #print(f"DEBUG: first_batch_data.requires_grad: {first_batch_data.requires_grad}")
    #print(f"DEBUG: first_batch_data.dtype: {first_batch_data.dtype}")
    #print(f"DEBUG: first_batch_data.shape: {first_batch_data.shape}")

    temp_outputs = model(first_batch_data)
    temp_loss = criterion(temp_outputs, first_batch_targets)
    #print(f"DEBUG: temp_outputs.requires_grad: {temp_outputs.requires_grad}")
    #print(f"DEBUG: temp_loss.requires_grad: {temp_loss.requires_grad}")
    temp_loss.backward()
    #print("DEBUG: Single batch backward pass successful.")
except Exception as e:
    #print(f"DEBUG: Single batch backward pass FAILED with error: {e}")
    raise # <--- Keep this 'raise' to see the full traceback

# --- No resource tracking, no debug block here ---

#train_model(model, train_loader, optimizer, criterion, device)

#accuracy, cm, report = evaluate_model(model, test_loader, device)

# No result logging or CSV saving for now

# import pandas as pd # Comment out
# pd.DataFrame(all_results).to_csv("results/experiment_summary.csv", index=False) # Comment out
# print("\n✅ All experiments complete. Results saved to results/experiment_summary.csv") # Comment out

model_type = "SNN_LIF" # Change model type here
print(f"\n⚙️ Model: {model_type}")

# SNN_LIF typically takes hidden_size (or hidden_neurons) and LIF parameters
# Make sure these parameters align with how your SNN_LIF model is defined in src/models/snn_lif.py
model_params = {"hidden_size": 128} # Or "hidden_neurons", depending on your SNN_LIF.__init__
lif_params = {
    "beta": 0.9,
    "threshold": 1.0,
    # SNN_LIF usually doesn't need "simulation_timesteps" directly in lif_params,
    # as it's typically a static (non-recurrent) SNN.
    # If your SNN_LIF takes `simulation_timesteps` in its __init__, include it.
}


input_size = features.shape[1]
num_classes = len(set(labels))

# Initialize the SNN_LIF model
model = initialize_model(model_type, input_size, num_classes, model_params, lif_params)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# --- Remove the TEMPORARY DEBUGGING BLOCK for now to reduce clutter ---
# try:
#     first_batch_data, first_batch_targets = next(iter(train_loader))
#     first_batch_data = first_batch_data.to(device)
#     first_batch_targets = first_batch_targets.to(device)

#     print(f"DEBUG: first_batch_data.requires_grad: {first_batch_data.requires_grad}")
#     print(f"DEBUG: first_batch_data.dtype: {first_batch_data.dtype}")
#     print(f"DEBUG: first_batch_data.shape: {first_batch_data.shape}")

#     temp_outputs = model(first_batch_data)
#     temp_loss = criterion(temp_outputs, first_batch_targets)
#     print(f"DEBUG: temp_outputs.requires_grad: {temp_outputs.requires_grad}")
#     print(f"DEBUG: temp_loss.requires_grad: {temp_loss.requires_grad}")
#     temp_loss.backward()
#     print("DEBUG: Single batch backward pass successful.")
# except Exception as e:
#     print(f"DEBUG: Single batch backward pass FAILED with error: {e}")
#     raise
# --- END TEMPORARY DEBUGGING BLOCK ---

#train_model(model, train_loader, optimizer, criterion, device)

#accuracy, cm, report = evaluate_model(model, test_loader, device)

# src/main_pipeline.py (relevant section)

# ... (previous code for data loading and preparation) ...

# --- Switch to SNN_CNN_LIF ---
model_type = "SNN_CNN_LIF" # Change model type here
print(f"\n⚙️ Model: {model_type}")

# SNN_CNN_LIF needs 'conv_out' and 'hidden_size' (and maybe input_size for its internal calculations)
model_params = {
    "conv_out": 32,  # Example value, adjust as needed
    "hidden_size": 128, # Example value, adjust as needed
}
lif_params = {
    "beta": 0.9,
    "threshold": 1.0,
    # SNN_CNN_LIF, as currently structured, doesn't use "simulation_timesteps" directly in its forward pass
}


input_size = features.shape[1]
num_classes = len(set(labels))

# Initialize the SNN_CNN_LIF model
# Ensure your initialize_model in src/utils/model_initializer.py handles SNN_CNN_LIF correctly
model = initialize_model(model_type, input_size, num_classes, model_params, lif_params)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

train_model(model, train_loader, optimizer, criterion, device)

accuracy, cm, report = evaluate_model(model, test_loader, device)

# ... (rest of your main_pipeline.py code) ...
