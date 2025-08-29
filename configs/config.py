# --- configs/config.py ---

# Global experiment configurations
N_RUNS = 1 # Number of robust evaluation runs
N_SPLITS = 5 # Number of folds for cross-validation within each run
RANDOM_STATE_OFFSET = 42 # Base random state for reproducibility across runs and splits

# Define a standard segment duration in seconds for all datasets
SEGMENT_DURATION_SECONDS = 2.0

# NEW MI Percentages of features to select for MI ablation study
MI_FEATURE_SELECTION_PERCENTAGES = [0.25, 0.35, 0.5, 0.65, 0.75]

# Pre-calculated total feature counts for 2-second segments
# Based on the dynamic segment_length calculation and the feature extraction logic
TOTAL_FEATURES_PER_DATASET = {
    "bern_barcelona": 542,  
    "bonn": 203,            # For 173.6Hz, segment_length=347 samples
    "panwar": 203,          # For 173.6Hz, segment_length=347 samples
    "hauz": 230             # For 200Hz, segment_length=400 samples
}

# --- Sampling Frequencies ---
SAMPLING_FREQ = {
    "bonn": 173.61,
    "hauz": 250.0,
    "bern_barcelona": 512.0,
    "panwar": 173.6
}

# --- Dataset Paths ---
DATASET_PATHS = {
    "bonn": {
        's': r"D:\RESEARCH\DATABASE\Bonn Univ Dataset\s\S",
        'z': r"D:\RESEARCH\DATABASE\Bonn Univ Dataset\z\Z",
        'n': r"D:\RESEARCH\DATABASE\Bonn Univ Dataset\n\N",
        'o': r"D:\RESEARCH\DATABASE\Bonn Univ Dataset\o\O",
        'f': r"D:\RESEARCH\DATABASE\Bonn Univ Dataset\f\F"
    },
    "hauz": {
        'ictal': r"D:\RESEARCH\DATABASE\Neurology_Sleep_Centre_Hauz Khas\EEG Epilepsy Datasets\ictal",
        'interictal': r"D:\RESEARCH\DATABASE\Neurology_Sleep_Centre_Hauz Khas\EEG Epilepsy Datasets\interictal",
        'preictal': r"D:\RESEARCH\DATABASE\Neurology_Sleep_Centre_Hauz Khas\EEG Epilepsy Datasets\preictal"
    },
    "bern_barcelona": { # Kept paths for potential future use, but not included in TASKS_TO_RUN
        "focal": [
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_F_Ind_1_750",
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_F_Ind_751_1500",
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_F_Ind_1501_2250",
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_F_Ind_2251_3000",
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_F_Ind_3001_3750"
        ],
        "non_focal": [
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_N_Ind_1_750",
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_N_Ind_751_1500",
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_N_Ind_1501_2250",
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_N_Ind_2251_3000",
            r"D:\RESEARCH\DATABASE\Bern-Barcelona\Data_N_Ind_3001_3750"
        ]
    },
    "panwar": {
        "epileptic": [
            r"D:\RESEARCH\DATABASE\Panwar",
        ],
        "healthy": [
            r"D:\RESEARCH\DATABASE\Panwar",
        ]
    },
}

# --- Label Mappings ---
LABEL_MAPPINGS = {
    
    "bonn_s_z": {"s": 1, "z": 0},
    "bonn_multi_bonn": {"s": 0, "z": 1, "n": 2, "o": 3, "f": 4},

    "hauz_ictal_vs_interictal": {"ictal": 1, "interictal": 0},
    "hauz_ictal_preictal_vs_interictal": {"ictal": 1, "preictal": 1, "interictal": 0},
    "hauz_multi_hauz": {"ictal": 0, "interictal": 1, "preictal": 2},
    "hauz_ictal_vs_others": {"ictal": 1, "interictal": 0, "preictal": 0},

    "panwar_epileptic_healthy": {
        "epileptic": 1,
        "healthy": 0
        
    },
    "focal_non_focal_classification": {
        "focal": 0,
        "non_focal": 1
    }
}

# Define the specific classification tasks for each dataset
CLASSIFICATIONS_TO_RUN = {
    
    "bonn": [
        "bonn_s_z",
        "bonn_multi_bonn",
    ],
    "hauz": [
        "hauz_ictal_vs_interictal",
        "hauz_ictal_preictal_vs_interictal",
        "hauz_multi_hauz",
        "hauz_ictal_vs_others",
    ],

    "panwar": ["panwar_epileptic_healthy"],
    
    "bern_barcelona": ["focal_non_focal_classification"]    
}

# --- Model Defaults (used as templates for TASKS_TO_RUN) ---
LIF_PARAMS = { # These will be merged into model_params for TemporalSNNClassifier
    "beta": 0.8853097787598204,
    "threshold": 0.9941287219464874,
    "spike_grad": "atan",
    "surrogate_scale": 2.0
}

OPTIMIZER_CONFIGS = {
    "Adam": {
        "lr": 0.000430855512305473,
        "weight_decay": 0.0
    }
}

DEFAULT_MODEL_PARAMS = {
    # This is the base template for SNN models used in task generation
    "TemporalSNNClassifier_Base": {
        "hidden_neurons": 128,
        "simulation_timesteps": 25,
        "lif_beta": LIF_PARAMS["beta"],
        "lif_threshold": LIF_PARAMS["threshold"],
        "spike_grad": LIF_PARAMS["spike_grad"],
        "surrogate_scale": LIF_PARAMS["surrogate_scale"]
    },
    # The old "TemporalSNNClassifier" default is now effectively superseded
    # by the explicit task definitions below, and can be removed if not used elsewhere.
    "TemporalSNNClassifier": {
        "input_size": 119,
        "num_classes": 3,
        "hidden_size": 256,
        "output_size": None,
        "num_layers": 2,
        "simulation_timesteps": 28,
    }
}

# --- Early Stopping Configuration ---
EARLY_STOPPING_PARAMS = {
    "patience": 7,
    "min_delta": 0.001,
    "monitor": "val_loss",
    "restore_best_weights": True
}

# --- Robust Evaluation Configuration (moved from all_run.py for consistency) ---
ROBUST_EVAL_CONFIG = {
    "num_runs": N_RUNS,
    "n_splits": N_SPLITS,
    "random_state_offset": RANDOM_STATE_OFFSET
}

# --- TASKS TO RUN: Programmatically Generated ---
TASKS_TO_RUN = []




# --- Generate baseline tasks (no feature selection) ---
for dataset_name, classifications in CLASSIFICATIONS_TO_RUN.items():
    for classification_name in classifications:
        task = {
            "dataset": dataset_name,
            "classification": classification_name + "_NO_FS", # Append _NO_FS for clear identification
            "model_type": "TemporalSNNClassifier",
            "model_params": {
                "input_size": TOTAL_FEATURES_PER_DATASET[dataset_name],
                "num_classes": len(LABEL_MAPPINGS[classification_name]),
                **DEFAULT_MODEL_PARAMS["TemporalSNNClassifier_Base"]
            },
            "tuning_params": {"lr": 0.001, "num_epochs": 30, "batch_size": 64}, # Consistent tuning params
            "use_feature_selection": False,
            "feature_selection_method": "None",
            "num_features_to_select": None
        }
        TASKS_TO_RUN.append(task)

# --- Generate MI Feature Selection tasks for each dataset, classification, and percentage ---
for dataset_name, classifications in CLASSIFICATIONS_TO_RUN.items():
    for classification_name in classifications:
        for percent in MI_FEATURE_SELECTION_PERCENTAGES:
            total_features = TOTAL_FEATURES_PER_DATASET[dataset_name]
            num_selected = int(round(total_features * percent))
            # Ensure at least 1 feature is selected if percentage is > 0
            if num_selected == 0 and percent > 0:
                num_selected = 1
            
            # Construct a descriptive classification name for the task
            task_name_suffix = f"_MI_FS_{int(percent*100)}percent"

            task = {
                "dataset": dataset_name,
                "classification": classification_name + task_name_suffix,
                "model_type": "TemporalSNNClassifier",
                "model_params": {
                    "input_size": num_selected,
                    "num_classes": len(LABEL_MAPPINGS[classification_name]),
                    **DEFAULT_MODEL_PARAMS["TemporalSNNClassifier_Base"]
                },
                "tuning_params": {"lr": 0.001, "num_epochs": 30, "batch_size": 64}, # Consistent tuning params
                "use_feature_selection": True,
                "feature_selection_method": "MI",
                "num_features_to_select": num_selected
            }
            TASKS_TO_RUN.append(task)