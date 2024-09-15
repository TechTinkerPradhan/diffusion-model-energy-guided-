# models/energy_model.py

import torch
import torch.nn as nn

class energy(nn.Module):
    def __init__(self, input_dim):
        super(energy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),  # First hidden layer
            nn.ReLU(),
            nn.Linear(256, 128),        # Second hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),         # Third hidden layer
            nn.ReLU(),
            nn.Linear(64, 1)            # Output layer (scalar energy value)
        )
    
    def forward(self, x):
        # x is the concatenated features of all objects in the scene
        energy = self.network(x)
        return energy
