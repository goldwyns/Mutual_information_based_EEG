from models.snn_lif import SNN_LIF
from models.snn_cnn_lif import SNN_CNN_LIF
from models.temporal_snn import TemporalSNNClassifier

def initialize_model(model_type, input_size, num_classes, model_params=None, lif_params=None):
    model_params = model_params or {}
    lif_params = lif_params or {}

    if model_type == "SNN_LIF":
        model = SNN_LIF(
            input_size=input_size,
            hidden_size=model_params.get("hidden_neurons", 128),
            output_size=num_classes,
            lif_params=lif_params
        )

    elif model_type == "SNN_CNN_LIF":
        model = SNN_CNN_LIF(
            input_size=input_size,
            conv_out=model_params.get("conv_out", 32),
            hidden_size=model_params.get("hidden_neurons", 128),
            output_size=num_classes,
            lif_params=lif_params
        )

    elif model_type == "TemporalSNNClassifier":
        model = TemporalSNNClassifier(
            input_features_dim=input_size,
            hidden_neurons=model_params.get("hidden_neurons", 128),
            num_classes=num_classes,
            T=model_params.get("simulation_timesteps", 25),
            lif_params=lif_params
        )
        print(f"Initialized TemporalSNNClassifier with input_size: {input_size}, num_classes: {num_classes}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model
