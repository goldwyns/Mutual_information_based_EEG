# --- configs/config.py ---

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
    }
}

# --- Sampling Frequencies ---
SAMPLING_FREQ = {
    "bonn": 173.61,
    "hauz": 250.0
}

# --- Label Mappings ---
LABEL_MAPPINGS = {
    "bonn_s_z": {"s": 1, "z": 0},
    "bonn_s_n": {"s": 1, "n": 0},
    "bonn_s_o": {"s": 1, "o": 0},
    "bonn_s_f": {"s": 1, "f": 0},
    "bonn_s_vs_others": {"s": 1, "z": 0, "n": 0, "o": 0, "f": 0},
    "bonn_multi_bonn": {"s": 0, "z": 1, "n": 2, "o": 3, "f": 4},

    "hauz_ictal_vs_interictal": {"ictal": 1, "interictal": 0},
    "hauz_interictal_vs_ictal": {"interictal": 0, "ictal": 1},
    "hauz_interictal_vs_preictal": {"interictal": 0, "preictal": 1},
    "hauz_ictal_preictal_vs_interictal": {"ictal": 1, "preictal": 1, "interictal": 0},
    "hauz_multi_hauz": {"ictal": 0, "interictal": 1, "preictal": 2},
    "hauz_ictal_vs_others": {"ictal": 1, "interictal": 0, "preictal": 0},
}

# --- Model Defaults (used as templates for TASKS_TO_RUN) ---
DEFAULT_MODEL_PARAMS = {
    "TemporalSNNClassifier": {
        "input_size": 119, # Default, will be overridden by task-specific
        "num_classes": 3,  # Default, will be overridden by task-specific
        "hidden_size": 256, # From your provided config
        "output_size": None, # From your provided config
        "num_layers": 2, # From your provided config
        "simulation_timesteps": 28, # From your provided config
    }
}

OPTIMIZER_CONFIGS = {
    "Adam": {
        "lr": 0.000430855512305473,
        "weight_decay": 0.0
    }
}

LIF_PARAMS = { # These will be merged into model_params for TemporalSNNClassifier
    "beta": 0.8853097787598204,
    "threshold": 0.9941287219464874,
    "spike_grad": "atan",
    "surrogate_scale": 2.0
}

# --- Early Stopping Configuration ---
EARLY_STOPPING_PARAMS = {
    "patience": 7,
    "min_delta": 0.001,
    "monitor": "val_accuracy"
}

# --- Robust Evaluation Configuration (moved from all_run.py for consistency) ---
ROBUST_EVAL_CONFIG = {
    "num_runs": 3,
    "n_splits": 5,
    "random_state_offset": 42
}

# --- Tasks to Run (including ablation study) ---
TASKS_TO_RUN = [
    # Task 1: Bonn S-Z with MI Feature Selection (using 60 features)
    {
        "dataset": "bonn",
        "classification": "bonn_s_z",
        "use_feature_selection": True,
        "feature_selection_method": "MI",
        "num_features_to_select": 60,
        "model_params": {
            "input_size": 60, # Specific for this task due to feature selection
            "num_classes": len(LABEL_MAPPINGS["bonn_s_z"]),
            "hidden_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["hidden_size"],
            "output_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["output_size"],
            "num_layers": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["num_layers"],
            "simulation_timesteps": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["simulation_timesteps"],
            "lif_beta": LIF_PARAMS["beta"],
            "lif_threshold": LIF_PARAMS["threshold"],
            # Add other LIF params if TemporalSNNClassifier uses them
            "spike_grad": LIF_PARAMS["spike_grad"],
            "surrogate_scale": LIF_PARAMS["surrogate_scale"]
        },
        "tuning_params": {
            "num_epochs": 20,
            "batch_size": 32,
            "lr": OPTIMIZER_CONFIGS["Adam"]["lr"],
        }
    },
    # Task 2: Bonn S-Z WITHOUT Feature Selection (Ablation Study)
    {
        "dataset": "bonn",
        "classification": "bonn_s_z_NO_FS", # Distinct name for ablation
        "use_feature_selection": False,
        "feature_selection_method": "None",
        "num_features_to_select": None, # Not applicable
        "model_params": {
            "input_size": 119, # Full input size without feature selection
            "num_classes": len(LABEL_MAPPINGS["bonn_s_z"]),
            "hidden_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["hidden_size"],
            "output_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["output_size"],
            "num_layers": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["num_layers"],
            "simulation_timesteps": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["simulation_timesteps"],
            "lif_beta": LIF_PARAMS["beta"],
            "lif_threshold": LIF_PARAMS["threshold"],
            "spike_grad": LIF_PARAMS["spike_grad"],
            "surrogate_scale": LIF_PARAMS["surrogate_scale"]
        },
        "tuning_params": {
            "num_epochs": 20,
            "batch_size": 32,
            "lr": OPTIMIZER_CONFIGS["Adam"]["lr"],
        }
    },
    # --- Other Bonn Tasks (adapted to new structure) ---
    {
        "dataset": "bonn",
        "classification": "bonn_s_n",
        "use_feature_selection": True, # Assuming FS is used for other Bonn tasks by default
        "feature_selection_method": "MI",
        "num_features_to_select": 60, # Adjust if different for each task
        "model_params": {
            "input_size": 60,
            "num_classes": len(LABEL_MAPPINGS["bonn_s_n"]),
            "hidden_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["hidden_size"],
            "output_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["output_size"],
            "num_layers": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["num_layers"],
            "simulation_timesteps": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["simulation_timesteps"],
            "lif_beta": LIF_PARAMS["beta"],
            "lif_threshold": LIF_PARAMS["threshold"],
            "spike_grad": LIF_PARAMS["spike_grad"],
            "surrogate_scale": LIF_PARAMS["surrogate_scale"]
        },
        "tuning_params": {
            "num_epochs": 20,
            "batch_size": 32,
            "lr": OPTIMIZER_CONFIGS["Adam"]["lr"],
        }
    },
    # Add other bonn_s_o, bonn_s_f, bonn_multi_bonn, bonn_s_vs_others here following the pattern
    # For example, for bonn_multi_bonn (5 classes):
    {
        "dataset": "bonn",
        "classification": "bonn_multi_bonn",
        "use_feature_selection": True,
        "feature_selection_method": "MI",
        "num_features_to_select": 60, # Adjust if different
        "model_params": {
            "input_size": 60,
            "num_classes": len(LABEL_MAPPINGS["bonn_multi_bonn"]), # 5 classes
            "hidden_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["hidden_size"],
            "output_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["output_size"],
            "num_layers": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["num_layers"],
            "simulation_timesteps": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["simulation_timesteps"],
            "lif_beta": LIF_PARAMS["beta"],
            "lif_threshold": LIF_PARAMS["threshold"],
            "spike_grad": LIF_PARAMS["spike_grad"],
            "surrogate_scale": LIF_PARAMS["surrogate_scale"]
        },
        "tuning_params": {
            "num_epochs": 20,
            "batch_size": 32,
            "lr": OPTIMIZER_CONFIGS["Adam"]["lr"],
        }
    },


    # --- Hauz Tasks (adapted to new structure) ---
    {
        "dataset": "hauz",
        "classification": "hauz_ictal_vs_interictal",
        "use_feature_selection": True,
        "feature_selection_method": "MI",
        "num_features_to_select": 60, # Adjust if different for Hauz
        "model_params": {
            "input_size": 60,
            "num_classes": len(LABEL_MAPPINGS["hauz_ictal_vs_interictal"]),
            "hidden_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["hidden_size"],
            "output_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["output_size"],
            "num_layers": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["num_layers"],
            "simulation_timesteps": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["simulation_timesteps"],
            "lif_beta": LIF_PARAMS["beta"],
            "lif_threshold": LIF_PARAMS["threshold"],
            "spike_grad": LIF_PARAMS["spike_grad"],
            "surrogate_scale": LIF_PARAMS["surrogate_scale"]
        },
        "tuning_params": {
            "num_epochs": 20,
            "batch_size": 32,
            "lr": OPTIMIZER_CONFIGS["Adam"]["lr"],
        }
    },
    # Ablation Task for Hauz: ictal_vs_interictal WITHOUT Feature Selection
    {
        "dataset": "hauz",
        "classification": "hauz_ictal_vs_interictal_NO_FS",
        "use_feature_selection": False,
        "feature_selection_method": "None",
        "num_features_to_select": None,
        "model_params": {
            "input_size": 119, # IMPORTANT: Confirm the original feature size for Hauz if different from Bonn
            "num_classes": len(LABEL_MAPPINGS["hauz_ictal_vs_interictal"]),
            "hidden_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["hidden_size"],
            "output_size": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["output_size"],
            "num_layers": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["num_layers"],
            "simulation_timesteps": DEFAULT_MODEL_PARAMS["TemporalSNNClassifier"]["simulation_timesteps"],
            "lif_beta": LIF_PARAMS["beta"],
            "lif_threshold": LIF_PARAMS["threshold"],
            "spike_grad": LIF_PARAMS["spike_grad"],
            "surrogate_scale": LIF_PARAMS["surrogate_scale"]
        },
        "tuning_params": {
            "num_epochs": 20,
            "batch_size": 32,
            "lr": OPTIMIZER_CONFIGS["Adam"]["lr"],
        }
    },
    # Add other Hauz tasks (hauz_multi_hauz, hauz_ictal_vs_others etc.) here,
    # following the same comprehensive structure, including their NO_FS ablation variants
]