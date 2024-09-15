# training/train_diffusion.py

import sys
import os

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_models.d_model import diffmodel
from my_models.e_model import energy
from my_data.dset import SceneData
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float32)
    alphas_cumprod = np.cos(((x / float(timesteps)) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0.0001, 0.9999)
    return betas.astype(np.float32)

def train_diffusion_model(data_path, energy_model_path, epochs=10, batch_size=32, timesteps=1000, learning_rate=1e-5):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    dataset = SceneData(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the diffusion model
    input_dim = dataset[0][0].numel()  # Total number of features per scene
    diffusion_model = diffmodel(input_dim=input_dim).to(device).float()

    # Load the pre-trained energy model
    energy_model = energy(input_dim=input_dim).to(device).eval()

    # Define the optimizer with weight decay for regularization
    optimizer = Adam(diffusion_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Define the beta schedule using the cosine schedule
    beta = cosine_beta_schedule(timesteps)
    beta = torch.from_numpy(beta).to(device).float()  # Ensure beta is float32

    # Training loop
    for epoch in range(epochs):
        diffusion_model.train()
        total_loss = 0.0
        for features, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to the appropriate device
            features = features.to(device).float()

            # Flatten the features (batch_size, input_dim)
            features = features.view(features.size(0), -1)

            # Sample random timesteps for each sample in the batch
            t = torch.randint(0, timesteps, (features.size(0),), device=device).long()

            # Compute the noise for the forward diffusion process
            noise = torch.randn_like(features).to(device).float()
            beta_t = beta[t].unsqueeze(1)
            alpha_t = 1 - beta_t
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            # Generate noisy features
            noisy_features = sqrt_alpha_t * features + sqrt_one_minus_alpha_t * noise

            # Predict the denoised features
            t_normalized = t.float() / float(timesteps)
            t_normalized = t_normalized.unsqueeze(1)

            denoised_features = diffusion_model(noisy_features, t_normalized)
            denoised_features = torch.clamp(denoised_features, min=0.0, max=1.0)

            # Compute reconstruction loss between denoised features and original features
            reconstruction_loss = nn.MSELoss()(denoised_features, features)

            # Energy guidance loss
            with torch.no_grad():
                energy_real = energy_model(features)
                energy_fake = energy_model(denoised_features)

            energy_loss = torch.mean(energy_fake) - torch.mean(energy_real)

            # Total loss
            loss = reconstruction_loss + 0.1 * energy_loss  # Adjust the weighting factor as needed

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # Check for NaNs in loss
            if torch.isnan(loss):
                print("NaN detected in loss. Skipping this batch.")
                continue

            # Check for NaNs in model parameters
            for param in diffusion_model.parameters():
                if torch.isnan(param).any():
                    print("NaN detected in model parameters. Exiting training.")
                    return

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Save the model checkpoint periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(diffusion_model.state_dict(), f'models/diffusion_model_epoch{epoch+1}.pth')

    # Save the final model
    torch.save(diffusion_model.state_dict(), 'models/diffusion_model.pth')
    print("Training completed and model saved.")

if __name__ == "__main__":
    # Paths to the data and pre-trained energy model
    data_path = 'my_data/synthetic_data.npy'
    energy_model_path = 'models/energy_model.pth'

    # Training parameters
    epochs = 20
    batch_size = 64
    timesteps = 1000
    learning_rate = 1e-5  # Reduced learning rate for stability

    train_diffusion_model(data_path, energy_model_path, epochs, batch_size, timesteps, learning_rate)
