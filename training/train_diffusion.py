# training/train_diffusion.py

import sys
import os

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_models.d_model import diffmodel
from my_models.e_model import energy
from my_data.dset import SceneData
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

def train_diffusion_model(data_path, energy_model_path, epochs=10, batch_size=32, timesteps=1000, learning_rate=1e-4):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    dataset = SceneData(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the diffusion model
    input_dim = dataset[0][0].numel()  # Total number of features per scene
    diffusion_model = diffmodel(input_dim=input_dim).to(device)

    # Load the pre-trained energy model
    energy_model = energy(input_dim=input_dim).to(device)
    energy_model.load_state_dict(torch.load(energy_model_path, map_location=device))
    energy_model.eval()  # Set to evaluation mode

    # Define the optimizer
    optimizer = Adam(diffusion_model.parameters(), lr=learning_rate)

    # Define the beta schedule
    beta = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
    beta = torch.from_numpy(beta).to(device)

    # Training loop
    for epoch in range(epochs):
        diffusion_model.train()
        total_loss = 0.0
        for features, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to the appropriate device
            features = features.to(device)

            # Flatten the features (batch_size, input_dim)
            features = features.view(features.size(0), -1)

            # Sample random timesteps for each sample in the batch
            t = torch.randint(0, timesteps, (features.size(0),), device=device).long()

            # Compute the noise for the forward diffusion process
            noise = torch.randn_like(features).to(device)
            alpha = torch.sqrt(1 - beta[t]).unsqueeze(1)
            beta_t = torch.sqrt(beta[t]).unsqueeze(1)
            noisy_features = alpha * features + beta_t * noise

            # Predict the denoised features
            t_normalized = t.float() / timesteps  # Normalize timestep
            t_normalized = t_normalized.unsqueeze(1)
            denoised_features = diffusion_model(noisy_features, t_normalized)

            # Compute reconstruction loss (e.g., Mean Squared Error)
            reconstruction_loss = nn.MSELoss()(denoised_features, features)

            # Energy guidance
            denoised_features.requires_grad_(True)
            energy_value = energy_model(denoised_features).mean()
            energy_grad = torch.autograd.grad(energy_value, denoised_features, create_graph=True)[0]
            energy_loss = (energy_grad * denoised_features).mean()

            # Total loss
            loss = reconstruction_loss + 0.1 * energy_loss  # Adjust weighting factor as needed

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

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
    learning_rate = 1e-4

    train_diffusion_model(data_path, energy_model_path, epochs, batch_size, timesteps, learning_rate)
