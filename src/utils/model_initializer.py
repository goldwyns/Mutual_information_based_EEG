from models.temporal_snn import TemporalSNNClassifier

def initialize_model(model_type, input_size, num_classes, model_params=None, lif_params=None):
    if model_type == 'TemporalSNNClassifier':
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
