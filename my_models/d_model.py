# models/diffusion_model.py

import torch
import torch.nn as nn

class diffmodel(nn.Module):
    def __init__(self, input_dim):
        super(diffmodel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # +1 for the timestep
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x, t):
        # x is the noisy object features
        # t is the current timestep
        x_t = torch.cat([x, t], dim=-1)
        denoised_x = self.network(x_t)
        return denoised_x