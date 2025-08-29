# src/models/temporal_snn.py

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
        
        # This line is correct and necessary for model.num_classes access in trainer.py
        self.num_classes = num_classes 

        beta = lif_params.get("beta", 0.9)
        threshold = lif_params.get("threshold", 1.0)
        
        # Dynamically get the surrogate function (e.g., 'atan', 'fast_sigmoid')
        spike_grad_name = lif_params.get("spike_grad", "fast_sigmoid")
        spike_grad = getattr(surrogate, spike_grad_name)

        # Layers
        self.fc1 = nn.Linear(input_features_dim, hidden_neurons)
        # --- CRITICAL FIX: Ensure init_hidden=True IS ABSENT ---
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad()) 

        self.fc2 = nn.Linear(hidden_neurons, num_classes)
        # --- CRITICAL FIX: Ensure init_hidden=True IS ABSENT ---
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad()) 

    def forward(self, x):
        # Initialize membrane potentials ONCE at the beginning of each forward pass
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = [] # To record spikes from the output layer

        for step in range(self.T): # Iterate over simulation timesteps
            cur_input = x # Input features are applied at each timestep (static input)

            # Layer 1: Pass the current input and the *updated* membrane potential from the previous step
            cur_input = self.fc1(cur_input)
            spk1, mem1 = self.lif1(cur_input, mem1) # mem1 is correctly passed and updated here

            # Layer 2 (Output Layer): Pass the spikes from the previous layer and *updated* membrane potential
            cur_input_for_fc2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur_input_for_fc2, mem2) # mem2 is correctly passed and updated here
            
            spk2_rec.append(spk2)

        # Stack recorded spikes and sum them over the time dimension for classification
        # This gives the total spike count for each class, which can be used for prediction
        spk2_rec = torch.stack(spk2_rec, dim=0)
        out = spk2_rec.sum(dim=0)
        
        return out