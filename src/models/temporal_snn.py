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
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        #print(f"DEBUG (Model): Input x.requires_grad: {x.requires_grad}") # New debug
        for step in range(self.T):
            cur_input = x
            #print(f"DEBUG (Model, Step {step}): cur_input (before fc1).requires_grad: {cur_input.requires_grad}") # New debug
            cur_input = self.fc1(cur_input)
            #print(f"DEBUG (Model, Step {step}): cur_input (after fc1).requires_grad: {cur_input.requires_grad}") # New debug
            spk1, mem1 = self.lif1(cur_input, mem1)
            #print(f"DEBUG (Model, Step {step}): spk1.requires_grad: {spk1.requires_grad}") # New debug
            #print(f"DEBUG (Model, Step {step}): mem1.requires_grad: {mem1.requires_grad}") # New debug
            cur_input_for_fc2 = self.fc2(spk1)
            #print(f"DEBUG (Model, Step {step}): cur_input_for_fc2 (after fc2).requires_grad: {cur_input_for_fc2.requires_grad}") # New debug
            spk2, mem2 = self.lif2(cur_input_for_fc2, mem2)
            #print(f"DEBUG (Model, Step {step}): spk2.requires_grad: {spk2.requires_grad}") # New debug
            #print(f"DEBUG (Model, Step {step}): mem2.requires_grad: {mem2.requires_grad}") # New debug
            spk2_rec.append(spk2)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        out = spk2_rec.sum(dim=0)
        #print(f"DEBUG (Model): Final out.requires_grad: {out.requires_grad}") # New debug
        return out