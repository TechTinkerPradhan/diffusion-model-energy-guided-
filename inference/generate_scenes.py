# inference/generate_scenes.py

import sys
import os

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from my_models.d_model import diffmodel  
from my_models.e_model import energy
import numpy as np
from tqdm import tqdm

def generate_new_scene(timesteps=1000):
    # Instantiate the diffusion model
    diffusion_model = diffmodel(input_dim=10*6)
    diffusion_model.load_state_dict(torch.load('models/diffusion_model.pth'))
    diffusion_model.eval()

    # Instantiate the energy model
    energy_model = energy(input_dim=10*6)
    energy_model.load_state_dict(torch.load('models/energy_model.pth'))
    energy_model.eval()

    beta = cosine_beta_schedule(timesteps)
    beta = torch.from_numpy(beta)
    beta = beta.float()

    x = torch.randn(1, 10*6) *0.1  # Start from random noise

    for t in tqdm(reversed(range(timesteps))):
        t_tensor = torch.full((1, 1), t, dtype=torch.float32) / timesteps
        if t % 50 == 0 or t == 935:
            print(f"Timestep {t}, x stats: mean={x.mean().item()}, std={x.std().item()}, max={x.max().item()}, min={x.min().item()}")

        # Diffusion model step without gradient tracking
        with torch.no_grad():
            x = diffusion_model(x, t_tensor)

        # Check for NaNs after the diffusion model
        if torch.isnan(x).any():
            print(f"NaN detected in x after diffusion model at timestep {t}")
            break

        # Enable gradient tracking for x
        x.requires_grad_(True)

        # Energy guidance
        energy_value = energy_model(x)
        energy_grad = torch.autograd.grad(energy_value.sum(), x)[0]

        # Check for NaNs in energy gradient
        if torch.isnan(energy_grad).any():
            print(f"NaN detected in energy gradient at timestep {t}")
            break

        # Update x using the energy gradient
        x = x - beta[t] * energy_grad

        # Detach x to prevent computation graph from growing indefinitely
        x = x.detach()

        # Check for NaNs after energy guidance
        if torch.isnan(x).any():
            print(f"NaN detected in x after energy guidance at timestep {t}")
            break

    generated_scene = x.view(10, 6).numpy()
    return generated_scene
    
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)

def get_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)


if __name__ == "__main__":
    scene = generate_new_scene()
    print("Generated Scene:", scene)
