from models.snn_lif import SNN_LIF
from models.snn_cnn_lif import SNN_CNN_LIF
from models.temporal_snn import TemporalSNNClassifier

def initialize_model(model_type, input_size, num_classes, model_params=None, lif_params=None):
    if model_type == 'SNN_LIF':
        model = SNN_LIF(
            input_size=input_size,
            hidden_size=model_params.get('hidden_size', 128), # Get hidden_size from model_params
            output_size=num_classes,
            lif_params=lif_params # Pass the whole lif_params dictionary
        )
        print(f"Initialized SNN_LIF with input_size: {input_size}, num_classes: {num_classes}")
    elif model_type == 'SNN_CNN_LIF':
        # --- ADD SNN_CNN_LIF INITIALIZATION HERE ---
        model = SNN_CNN_LIF(
            input_size=input_size,
            conv_out=model_params.get('conv_out', 32), # Get conv_out from model_params
            hidden_size=model_params.get('hidden_size', 128), # Get hidden_size
            output_size=num_classes,
            lif_params=lif_params
        )
        print(f"Initialized SNN_CNN_LIF with input_size: {input_size}, num_classes: {num_classes}")
    elif model_type == 'TemporalSNNClassifier':
        model = TemporalSNNClassifier(
            input_features_dim=input_size,
            hidden_neurons=model_params.get('hidden_neurons', 128),
            num_classes=num_classes,
            # T should be passed from model_params/lif_params as configured in main_pipeline
            T=lif_params.get('simulation_timesteps', 25), # Using lif_params for consistency
            lif_params=lif_params # Pass the whole lif_params dictionary
        )
        print(f"Initialized TemporalSNNClassifier with input_size: {input_size}, num_classes: {num_classes}")
    else:
        raise ValueError("Unsupported model type.")
    return model
