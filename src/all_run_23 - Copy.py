import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn # For CrossEntropyLoss
import torch.optim as optim # For Adam optimizer
import json
from datetime import datetime
import optuna

# Get the absolute path to the directory containing all_run.py (which is src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path to the project root directory (one level up from src/)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to sys.path to make 'configs' package and new utils importable
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your configuration and utility functions
import configs.config
from src.utils.trainer import train_model
from src.utils.evaluator import evaluate_model
from src.utils.model_initializer import initialize_model
from src.utils.features import extract_features
from src.utils.data_loader import load_signal_from_mat, load_signal_from_txt
from src.utils.data_preprocessing import preprocess_signal
from src.utils.feature_selection import select_features_mutual_info, select_features_rfe # Make sure these are properly imported

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Global Configuration (Now primarily from configs.config) ---
GLOBAL_SEED = 42 # Base seed for reproducibility
# Load robust evaluation parameters from config
N_RUNS = configs.config.ROBUST_EVAL_CONFIG["num_runs"]
N_SPLITS = configs.config.ROBUST_EVAL_CONFIG["n_splits"]
RANDOM_STATE_OFFSET = configs.config.ROBUST_EVAL_CONFIG["random_state_offset"]

# --- Results Output Configuration ---
RESULTS_DIR = os.path.join(project_root, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE_PREFIX = "robust_evaluation_results"
RESULTS_SAVE_FILE = os.path.join(RESULTS_DIR, f"{RESULTS_FILE_PREFIX}_latest.json") # File for incremental saving/resuming

# --- Early Stopping Configuration ---
EARLY_STOPPING_PARAMS = configs.config.EARLY_STOPPING_PARAMS

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Optional: for deterministic algorithms (might slow down training)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def load_data_and_create_labels(dataset_paths, label_mapping, original_sampling_frequency, dataset_name):
    """
    Loads EEG data from .mat or .txt files, applies preprocessing, extracts features.
    Assumes each file or files within a directory correspond to a specific class.
    Args:
        dataset_paths (dict): Dictionary mapping class identifiers (e.g., 's', 'z')
                              to file paths (e.g., 'data/bonn/s.mat') or directory paths (e.g., 'data/hauz/ictal').
        label_mapping (dict): Dictionary mapping class identifiers to numerical labels.
        original_sampling_frequency (float): Original sampling frequency of the EEG data as per config.
        dataset_name (str): The name of the dataset (e.g., 'bonn', 'hauz').
    Returns:
        tuple: (features_np, labels_np) as numpy arrays (converted to torch.Tensors later).
    """
    all_features = []
    all_labels = []

    print(f"Loading real EEG data for tasks based on: {dataset_paths}")

    files_to_process_with_info = [] # Store tuples of (filepath, class_identifier, label)

    for class_identifier, path_entries in dataset_paths.items():
        # IMPORTANT: Use the provided label_mapping directly here, as it's already filtered/transformed
        # by the calling function (the main loop).
        if class_identifier not in label_mapping:
            print(f"Warning: Class identifier '{class_identifier}' from path '{path_entries}' not found in label mapping provided to load_data_and_create_labels. Skipping.")
            continue

        label = label_mapping[class_identifier]

        # Ensure path_entries is always iterable
        if not isinstance(path_entries, list):
            path_entries = [path_entries]

        for path_entry in path_entries:
            if os.path.isdir(path_entry):
                print(f"Detected directory: {path_entry}. Searching for EEG files inside.")
                found_files_in_dir = False
                for root, _, files in os.walk(path_entry):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        file_ext = os.path.splitext(file_path)[1].lower()

                        if file_ext in ['.mat', '.txt', '.TXT']:
                            # --- IMPORTANT: Panwar specific file naming logic goes HERE ---
                            # This logic now correctly determines the *actual* class from filename
                            # and uses it to look up the label in the passed `label_mapping`.
                            if dataset_name == "panwar":
                                base_name = os.path.splitext(file_name)[0]
                                current_class_identifier = None
                                if base_name.startswith('E') or base_name.startswith('TrainE'):
                                    current_class_identifier = "epileptic"
                                elif base_name.startswith('H') or base_name.startswith('TrainH'):
                                    current_class_identifier = "healthy"
                                else:
                                    print(f"Warning: Unrecognized Panwar file prefix for {file_name}. Skipping.")
                                    continue # Skip this file
                                
                                # Use the Panwar-specific class identifier to get the label
                                if current_class_identifier not in label_mapping:
                                    print(f"Warning: Panwar class identifier '{current_class_identifier}' from file '{file_name}' not found in the provided label mapping {label_mapping}. Skipping.")
                                    continue
                                current_label = label_mapping[current_class_identifier]
                                files_to_process_with_info.append((file_path, current_class_identifier, current_label))
                                found_files_in_dir = True # At least one recognized file found
                            else:
                                # For other datasets, use the class_identifier from the dataset_paths mapping directly
                                files_to_process_with_info.append((file_path, class_identifier, label))
                                found_files_in_dir = True
                
                if not found_files_in_dir: # Check if no valid files were added from this directory
                    print(f"Warning: No .mat or .txt files (or recognized Panwar files) found in directory {path_entry}. Ensure files are present and extensions/naming are correct.")
                    
            elif os.path.isfile(path_entry):
                file_ext = os.path.splitext(path_entry)[1].lower()
                if file_ext in ['.mat', '.txt', '.TXT']:
                    # --- IMPORTANT: Panwar specific file naming logic for direct files, though less likely for Panwar ---
                    if dataset_name == "panwar":
                        # This scenario is less likely for Panwar given its folder structure,
                        # but keeping it for completeness if file_path_entry was a single file.
                        base_name = os.path.splitext(os.path.basename(path_entry))[0]
                        current_class_identifier = None
                        if base_name.startswith('E') or base_name.startswith('TrainE'):
                            current_class_identifier = "epileptic"
                        elif base_name.startswith('H') or base_name.startswith('TrainH'):
                            current_class_identifier = "healthy"
                        else:
                            print(f"Warning: Unrecognized Panwar file prefix for {os.path.basename(path_entry)}. Skipping.")
                            continue
                        
                        if current_class_identifier not in label_mapping:
                            print(f"Warning: Panwar class identifier '{current_class_identifier}' from file '{os.path.basename(path_entry)}' not found in the provided label mapping {label_mapping}. Skipping.")
                            continue
                        current_label = label_mapping[current_class_identifier]
                        files_to_process_with_info.append((path_entry, current_class_identifier, current_label))
                    else:
                        files_to_process_with_info.append((path_entry, class_identifier, label))
                else:
                    print(f"Warning: Unsupported file extension '{file_ext}' for direct file {path_entry}. Only .mat, .txt, and .TXT are supported. Skipping.")
            else:
                print(f"Warning: Path '{path_entry}' is neither a file nor a directory. Skipping.")

    if not files_to_process_with_info:
        print("Error: No supported EEG files found based on dataset_paths. Please check paths and file extensions.")
        # Fallback to dummy data
        dummy_input_size = configs.config.DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["input_size"]
        num_samples = 20
        dummy_num_classes = len(label_mapping) if label_mapping else 2
        print(f"Returning dummy data (shape {num_samples}x{dummy_input_size}) to allow script to run.")
        return np.random.randn(num_samples, dummy_input_size).astype(np.float32), \
               np.random.randint(0, dummy_num_classes, num_samples).astype(np.int64)

    print(f"Found {len(files_to_process_with_info)} EEG files to load.")

    # Calculate segment length based on desired duration in seconds and actual sampling frequency
    # We round to the nearest integer for sample count
    segment_length = int(round(configs.config.SEGMENT_DURATION_SECONDS * original_sampling_frequency))
    
    # Ensure minimum segment length (e.g., for STFT to produce enough bins)
    # A segment length of 1 for STFT would give only 1 feature (DC component), so a minimum of 2 or more is wise.
    if segment_length < 2: 
        print(f"Warning: Calculated segment length {segment_length} for {dataset_name} at {original_sampling_frequency}Hz is too small for feature extraction. Defaulting to a safe minimum of 178 samples.")
        segment_length = 178 # Fallback to existing working length if calculation yields too small a number
    
    print(f"Calculated segment length for {dataset_name} ({original_sampling_frequency}Hz) is {segment_length} samples ({configs.config.SEGMENT_DURATION_SECONDS} seconds).")


    for filepath, class_identifier_for_file, label_for_file in tqdm(files_to_process_with_info, desc="Loading and processing EEG files"):
        signal_data = None
        file_ext = os.path.splitext(filepath)[1].lower()

        data_key_for_mat = 'data'
        if dataset_name == "hauz": # Specific for Hauz, assuming it might not have 'data' key always
            data_key_for_mat = None

        if file_ext == '.mat':
            signal_data = load_signal_from_mat(filepath, data_key_for_mat)
        elif file_ext == '.txt' or file_ext == '.TXT': # Explicitly check for .TXT as well
            signal_data = load_signal_from_txt(filepath)

        if signal_data is not None:
            if signal_data.ndim > 1:
                signal_data = signal_data.flatten()

            processed_signal, effective_sampling_frequency = preprocess_signal(
                signal_data, original_sampling_frequency, dataset_name
            )

            extracted_features_for_file = []
            num_segments = len(processed_signal) // segment_length

            if num_segments == 0:
                print(f"Warning: Processed signal from {filepath} is too short ({len(processed_signal)} samples) for segment length {segment_length}. Skipping feature extraction for this file.")
                continue

            for i in range(num_segments):
                segment = processed_signal[i * segment_length : (i + 1) * segment_length]
                if len(segment) < 2:
                    print(f"Warning: Segment {i} from {filepath} is too short after segmentation. Skipping.")
                    continue

                features_for_segment = extract_features(segment, effective_sampling_frequency)
                extracted_features_for_file.append(features_for_segment)

            if not extracted_features_for_file:
                print(f"Warning: No valid segments found or features extracted from {filepath}. Skipping this file.")
                continue

            current_features = np.vstack(extracted_features_for_file)
            current_labels = np.full(current_features.shape[0], label_for_file, dtype=np.int64) # Use label_for_file

            all_features.append(current_features)
            all_labels.append(current_labels)
        else:
            print(f"Skipping {filepath} due to data loading error or unsupported format.")

    if not all_features:
        print("Error: No real features loaded after attempting to process all files. Check file contents and data keys.")
        dummy_input_size = configs.config.DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["input_size"]
        num_samples = 20
        dummy_num_classes = len(label_mapping) if label_mapping else 2
        print(f"Returning dummy data (shape {num_samples}x{dummy_input_size}) to allow script to run, but real data loading failed.")
        return np.random.randn(num_samples, dummy_input_size).astype(np.float32), \
               np.random.randint(0, dummy_num_classes, num_samples).astype(np.int64)

    features_np = np.vstack(all_features)
    labels_np = np.concatenate(all_labels)

    return features_np, labels_np


# --- OPTUNA OBJECTIVE FUNCTION (Kept for completeness, not active when N_OPTUNA_TRIALS=0) ---
# NOTE: This objective function is primarily used if you enable Optuna tuning (N_OPTUNA_TRIALS > 0).
# The main robust evaluation loop below uses fixed parameters from the TASKS_TO_RUN config.
def objective(trial: optuna.Trial, features_tensor: torch.Tensor, labels_tensor: torch.Tensor,
              num_classes: int, task_name: str, task_idx: int, total_tasks: int, run_idx: int):
    # This function is not called when N_OPTUNA_TRIALS = 0, which is the default in config.py
    # If N_OPTUNA_TRIALS > 0, you would also need to uncomment and enable relevant parts of the main loop
    # to trigger Optuna studies for specific tasks.

    print(f"\n--- Optuna Trial {trial.number}: Task {task_idx+1}/{total_tasks} ({task_name}) - Run {run_idx} ---")

    # Suggest hyperparameters using Optuna's API
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    num_epochs_trial = trial.suggest_int("num_epochs", 5, 20)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    simulation_timesteps = trial.suggest_int("simulation_timesteps", 10, 50)
    lif_beta = trial.suggest_float("lif_beta", 0.8, 0.99)
    lif_threshold = trial.suggest_float("lif_threshold", 0.5, 2.0)

    print(f"Trial Hyperparams: LR={lr:.6f}, Epochs={num_epochs_trial}, Batch={batch_size}, "
          f"Timesteps={simulation_timesteps}, Beta={lif_beta:.4f}, Threshold={lif_threshold:.4f}")

    # Prepare model parameters for the trial
    trial_model_params = configs.config.DEFAULT_MODEL_PARAMS.get("TemporalSNNClassifier", {}).copy()
    trial_model_params["simulation_timesteps"] = simulation_timesteps
    trial_model_params["input_size"] = features_tensor.shape[1] # Ensure input_size is dynamic

    trial_lif_params = configs.config.LIF_PARAMS.copy()
    trial_lif_params["beta"] = lif_beta
    trial_lif_params["threshold"] = lif_threshold

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=GLOBAL_SEED + trial.number)

    fold_accuracies = []

    # Early Stopping parameters for Optuna trials (from config.py)
    patience_optuna = configs.config.EARLY_STOPPING_PARAMS["patience"]
    min_delta_optuna = configs.config.EARLY_STOPPING_PARAMS["min_delta"]
    monitor_metric_optuna = configs.config.EARLY_STOPPING_PARAMS["monitor"]

    for fold_idx, (train_val_index, test_index) in enumerate(skf.split(features_tensor, labels_tensor)):
        train_val_features, test_features = features_tensor[train_val_index], features_tensor[test_index]
        train_val_labels, test_labels = labels_tensor[train_val_index], labels_tensor[test_index]

        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED + trial.number + fold_idx)
        train_indices, val_indices = next(skf_inner.split(train_val_features, train_val_labels))

        train_features_fold, val_features_fold = train_val_features[train_indices], train_val_features[val_indices]
        train_labels_fold, val_labels_fold = train_val_labels[train_indices], train_val_labels[val_indices]

        train_dataset = TensorDataset(train_features_fold, train_labels_fold)
        val_dataset = TensorDataset(val_features_fold, val_labels_fold)
        test_dataset = TensorDataset(test_features, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = initialize_model("TemporalSNNClassifier", trial_model_params["input_size"], num_classes,
                                 trial_model_params, trial_lif_params)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr) # Use torch.optim.Adam
        criterion = nn.CrossEntropyLoss() # Use torch.nn.CrossEntropyLoss

        best_model_state_dict, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
            train_model(model, train_loader, optimizer, criterion, device,
                        num_epochs=num_epochs_trial,
                        val_loader=val_loader,
                        patience=patience_optuna,
                        min_delta=min_delta_optuna,
                        monitor_metric=monitor_metric_optuna,
                        disable_tqdm=True)
        model.load_state_dict(best_model_state_dict)

        accuracy, _, _, _ = evaluate_model(model, test_loader, device, num_classes=num_classes, disable_tqdm=True, return_loss=True)
        fold_accuracies.append(accuracy)

    avg_accuracy = np.mean(fold_accuracies)
    print(f"Trial {trial.number} Avg K-Fold Accuracy: {avg_accuracy:.4f}")

    return avg_accuracy


# --- Main execution loop ---
final_aggregated_results = [] # This will now be loaded from file if it exists

# --- Load previous results for resume capability ---
if os.path.exists(RESULTS_SAVE_FILE):
    try:
        with open(RESULTS_SAVE_FILE, 'r') as f:
            final_aggregated_results = json.load(f)
        print(f"Resuming robust evaluation from previously saved results in: {RESULTS_SAVE_FILE}")
        print(f"Loaded {len(final_aggregated_results)} completed task-run-fold combinations.")
    except json.JSONDecodeError as e:
        print(f"Error loading {RESULTS_SAVE_FILE} for resume: {e}. Starting fresh.")
        final_aggregated_results = []
    except Exception as e:
        print(f"An unexpected error occurred while loading {RESULTS_SAVE_FILE}: {e}. Starting fresh.")
        final_aggregated_results = []
else:
    print(f"No previous results file found at {RESULTS_SAVE_FILE}. Starting robust evaluation from scratch.")


# Get the list of tasks from the config
tasks_to_run = configs.config.TASKS_TO_RUN
total_tasks = len(tasks_to_run)
print(f"Found {total_tasks} tasks to run.")


# --- Outer loop for multiple runs ---
for run_idx in range(1, N_RUNS + 1): # N_RUNS is now from config
    print(f"\n======== Starting Run {run_idx}/{N_RUNS} (Robust Evaluation) ========")
    set_seed(GLOBAL_SEED + run_idx - 1) # Set seed for reproducibility of outer loop

    for task_idx, task_config in enumerate(tasks_to_run): # Iterate over all tasks
        dataset_name = task_config["dataset"]
        full_classification_name = task_config["classification"] # Store the full name
        task_name = f"{dataset_name}_{full_classification_name}"

        # --- Check if ALL folds for this task-run combination are already completed ---
        completed_folds_for_this_task_run = [
            res for res in final_aggregated_results
            if res["run"] == run_idx and
               res["dataset"] == dataset_name and
               res["classification"] == full_classification_name # Use full_classification_name here
        ]

        if len(completed_folds_for_this_task_run) >= N_SPLITS: # N_SPLITS is now from config
            print(f"--- Task {task_idx+1}/{total_tasks}: Dataset={dataset_name}, Classification={full_classification_name} (Run {run_idx}) --- ALREADY COMPLETED ({N_SPLITS} folds found). Skipping.")
            continue


        print(f"\n--- Task {task_idx+1}/{total_tasks}: Dataset={dataset_name}, Classification={full_classification_name} (Run {run_idx}) ---")
        print(f"  Using Feature Selection: {task_config['use_feature_selection']}")
        if task_config['use_feature_selection']:
            print(f"  Method: {task_config['feature_selection_method']}, Num Features: {task_config['num_features_to_select']}")
        else:
            print("  Feature Selection: Disabled")

        # --- Data Loading and Feature Extraction for current task ---
        dataset_paths_for_loader = configs.config.DATASET_PATHS.get(dataset_name, {})

        # 🎯 CRITICAL FIX: Extract the base classification key for LABEL_MAPPINGS
        original_classification_key = full_classification_name
        if "_NO_FS" in original_classification_key:
            original_classification_key = original_classification_key.replace("_NO_FS", "")
        elif "_MI_FS_" in original_classification_key:
            # Find the start of the MI_FS suffix
            mi_suffix_start_index = original_classification_key.find("_MI_FS_")
            if mi_suffix_start_index != -1: # Ensure it's found
                original_classification_key = original_classification_key[:mi_suffix_start_index]
        elif "_RFE_FS_" in original_classification_key:
            rfe_suffix_start_index = original_classification_key.find("_RFE_FS_")
            if rfe_suffix_start_index != -1:
                original_classification_key = original_classification_key[:rfe_suffix_start_index]


        label_mapping_for_loader = configs.config.LABEL_MAPPINGS.get(original_classification_key, {}) # Corrected retrieval

        sampling_freq_for_loader = configs.config.SAMPLING_FREQ.get(dataset_name, 250.0)

        # Check if label_mapping_for_loader is empty, which would cause issues
        if not label_mapping_for_loader:
            print(f"Error: label_mapping for key '{original_classification_key}' derived from '{full_classification_name}' is empty or not found in config.LABEL_MAPPINGS. Cannot proceed with data loading for task {task_name}. Skipping this task.")
            continue # Skip to the next task if label mapping is invalid

        features_np, labels_np = load_data_and_create_labels(
            dataset_paths_for_loader, label_mapping_for_loader, sampling_freq_for_loader, dataset_name
        )

        # Handle dummy data case after load_data_and_create_labels returns
        # Check for the specific dummy data shape and all-zero labels to reliably identify it
        if features_np.shape[0] == 20 and features_np.shape[1] == configs.config.DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["input_size"] and np.all(labels_np == np.zeros(20)):
            print(f"Warning: Dummy data returned for task {task_name}. Skipping training/evaluation for this task-run.")
            # Optionally log this as a skipped task in your results if you want
            continue # Skip to the next task-run if dummy data is loaded


        num_classes = len(np.unique(labels_np))
        print(f"Real data loaded: Features shape {features_np.shape}, Labels shape {labels_np.shape}, Classes: {num_classes}")

        # --- Feature Selection (if enabled for this task) ---
        current_X_for_fs = features_np
        current_y_for_fs = labels_np
        selected_feature_indices = np.arange(features_np.shape[1]) # Default: all features

        logged_mi_scores = {}
        logged_rfe_ranking = {}
        rfe_estimator_used = "N/A"

        if task_config["use_feature_selection"]:
            fs_method = task_config["feature_selection_method"]
            num_features = task_config["num_features_to_select"]

            if fs_method == 'MI':
                print(f"Applying feature selection using MI (top {num_features})...")
                current_X_for_fs, selected_feature_indices, mi_scores = select_features_mutual_info(current_X_for_fs, current_y_for_fs, num_features)
                logged_mi_scores = {str(i): float(score) for i, score in enumerate(mi_scores)}
            elif fs_method == 'RFE':
                print(f"Applying feature selection using RFE (top {num_features})...")
                # RFE_ESTIMATOR needs to be defined. Assuming LogisticRegression as a default or from config.
                rfe_estimator_name = task_config.get("rfe_estimator", "LogisticRegression") # Can be specified in task_config if needed
                current_X_for_fs, selected_feature_indices, rfe_ranking = select_features_rfe(current_X_for_fs, current_y_for_fs, num_features, rfe_estimator_name)
                logged_rfe_ranking = {str(i): float(rank) for i, rank in enumerate(rfe_ranking)}
                rfe_estimator_used = rfe_estimator_name
            else:
                print(f"Warning: Unknown feature selection method '{fs_method}' specified. Skipping feature selection.")
        else:
            print("Feature selection is disabled for this task.")

        features_tensor = torch.from_numpy(current_X_for_fs).float()
        labels_tensor = torch.from_numpy(labels_np).long()
        input_size_for_snn = features_tensor.shape[1] # This is the actual input size after FS

        # --- K-Fold Cross-Validation (Robust Evaluation) ---
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=GLOBAL_SEED + run_idx + RANDOM_STATE_OFFSET)

        # Get task-specific parameters from config
        task_model_params = task_config["model_params"].copy()
        # Ensure model's input_size parameter correctly reflects the actual feature size after selection
        task_model_params["input_size"] = input_size_for_snn
        task_model_params["num_classes"] = num_classes # Ensure num_classes is correct for the specific task

        task_tuning_params = task_config["tuning_params"]
        
        # Extract LIF params from model_params for initialize_model
        # NOTE: Your config.py places LIF params directly in model_params for TASKS_TO_RUN.
        # So, we should extract them from task_model_params.
        lif_params_for_model = {
            "beta": task_model_params.pop("lif_beta"), # .pop() removes them from task_model_params
            "threshold": task_model_params.pop("lif_threshold"),
            "spike_grad": task_model_params.pop("spike_grad"),
            "surrogate_scale": task_model_params.pop("surrogate_scale")
        }

        # Early stopping parameters for robust evaluation (from config.py)
        patience = EARLY_STOPPING_PARAMS["patience"]
        min_delta = EARLY_STOPPING_PARAMS["min_delta"]
        monitor_metric = EARLY_STOPPING_PARAMS["monitor"]


        for fold_idx, (train_val_index, test_index) in enumerate(skf.split(features_tensor, labels_tensor)):
            # Check if this specific fold (for this run and task) has already been completed
            is_fold_completed = False
            for res in completed_folds_for_this_task_run:
                if res.get("fold_idx") == fold_idx:
                    is_fold_completed = True
                    break

            if is_fold_completed:
                print(f"  Run {run_idx} - Fold {fold_idx + 1}/{N_SPLITS} ALREADY COMPLETED. Skipping.")
                continue

            print(f"  Run {run_idx} - Fold {fold_idx + 1}/{N_SPLITS}")

            train_val_features, test_features = features_tensor[train_val_index], features_tensor[test_index]
            train_val_labels, test_labels = labels_tensor[train_val_index], labels_tensor[test_index]

            # Inner split for training and validation set
            skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED + run_idx + fold_idx + RANDOM_STATE_OFFSET)
            train_indices, val_indices = next(skf_inner.split(train_val_features, train_val_labels))

            train_features_fold, val_features_fold = train_val_features[train_indices], train_val_features[val_indices]
            train_labels_fold, val_labels_fold = train_val_labels[train_indices], train_val_labels[val_indices]

            train_dataset = TensorDataset(train_features_fold, train_labels_fold)
            val_dataset = TensorDataset(val_features_fold, val_labels_fold)
            test_dataset = TensorDataset(test_features, test_labels)

            train_loader = DataLoader(train_dataset, batch_size=task_tuning_params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=task_tuning_params["batch_size"], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=task_tuning_params["batch_size"], shuffle=False)

            model = initialize_model("TemporalSNNClassifier", task_model_params["input_size"], num_classes,
                                     task_model_params, lif_params_for_model) # Pass extracted LIF params separately
            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=task_tuning_params["lr"])
            criterion = nn.CrossEntropyLoss()

            best_model_state_dict, train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = \
                train_model(model, train_loader, optimizer, criterion, device,
                            num_epochs=task_tuning_params["num_epochs"],
                            val_loader=val_loader,
                            patience=patience,
                            min_delta=min_delta,
                            monitor_metric=monitor_metric,
                            disable_tqdm=True)
            model.load_state_dict(best_model_state_dict)

            accuracy, cm, report_dict, test_loss = evaluate_model(model, test_loader, device, num_classes=num_classes, disable_tqdm=True, return_loss=True)


            fold_result = {
                "dataset": dataset_name,
                "classification": full_classification_name, # Use full_classification_name here for logging
                "model": "TemporalSNNClassifier_Robust", # Naming convention for this run type
                "run": run_idx,
                "fold_idx": fold_idx,
                "feature_selection_used": task_config["use_feature_selection"],
                "feature_selection_method": task_config["feature_selection_method"],
                "num_features_selected": input_size_for_snn, # Actual features used by model
                "mi_scores": logged_mi_scores,
                "rfe_ranking": logged_rfe_ranking,
                "rfe_estimator_used": rfe_estimator_used,
                "tuned_params": {
                    "lr": task_tuning_params["lr"],
                    "num_epochs": task_tuning_params["num_epochs"],
                    "batch_size": task_tuning_params["batch_size"],
                    "simulation_timesteps": task_model_params.get("simulation_timesteps"),
                    "lif_beta": lif_params_for_model["beta"],
                    "lif_threshold": lif_params_for_model["threshold"]
                },
                "accuracy": accuracy,
                "confusion_matrix": cm.tolist(),
                "classification_report": report_dict,
                "test_loss": test_loss,
                "train_loss_history": train_loss_hist,
                "train_acc_history": train_acc_hist,
                "val_loss_history": val_loss_hist,
                "val_acc_history": val_acc_hist
            }
            final_aggregated_results.append(fold_result)

            # --- Save results incrementally after each fold ---
            try:
                with open(RESULTS_SAVE_FILE, 'w') as f:
                    json.dump(final_aggregated_results, f, indent=4)
                print(f"  Results for Run {run_idx}, Task '{task_name}', Fold {fold_idx+1}/{N_SPLITS} saved incrementally to {os.path.basename(RESULTS_SAVE_FILE)}.")
            except Exception as e:
                print(f"  Error saving incremental results: {e}")

        # After all folds for a task-run are done, calculate avg_accuracy_this_run
        accuracies_for_this_task_run = [
            res["accuracy"] for res in final_aggregated_results
            if res["run"] == run_idx and
               res["dataset"] == dataset_name and
               res["classification"] == full_classification_name # Use full_classification_name here
        ]
        avg_accuracy_this_run = np.mean(accuracies_for_this_task_run) if accuracies_for_this_task_run else 0.0
        print(f"Run {run_idx} Avg K-Fold Accuracy for {task_name} (across {len(accuracies_for_this_task_run)} folds): {avg_accuracy_this_run:.4f}")


print("\nAll tasks and runs completed.")

# --- Final Save of All Aggregated Results to a TIMESTAMPED JSON ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_results_filename = os.path.join(RESULTS_DIR, f"{RESULTS_FILE_PREFIX}_final_summary_{timestamp}.json")
try:
    with open(final_results_filename, 'w') as f:
        json.dump(final_aggregated_results, f, indent=4)
    print(f"\nFinal complete aggregated results saved to: {final_results_filename}")
except Exception as e:
    print(f"\nError saving final complete results to JSON: {e}")

print("\nRobust evaluation complete!")