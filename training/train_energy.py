# training/train_energy.py
import sys
import os

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_data.dset import SceneData
from my_models.e_model import energy
from torch.optim import Adam
from tqdm import tqdm
import numpy as np



def train_energy_model(data_path, epochs=10, batch_size=32, learning_rate=1e-3):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    dataset = SceneData(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the energy model
    input_dim = dataset[0][0].numel()  # Total number of features per scene
    model = energy(input_dim=input_dim).to(device)

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for features, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to the appropriate device
            features = features.to(device)
            labels = labels.to(device)

            # Flatten the features (batch_size, input_dim)
            features = features.view(features.size(0), -1)

            # Forward pass to compute energies
            energies = model(features).squeeze()

            # Separate energies for positive and negative samples
            positive_energies = energies[labels == 1]
            negative_energies = energies[labels == 0]

            # Skip batches without both positive and negative samples
            if len(positive_energies) == 0 or len(negative_energies) == 0:
                continue

            # Compute contrastive loss
            loss = (positive_energies.mean() - negative_energies.mean())

            # Regularization term (optional)
            # l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
            # loss += 1e-4 * l2_reg

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
            torch.save(model.state_dict(), f'models/energy_model_epoch{epoch+1}.pth')

    # Save the final model
    torch.save(model.state_dict(), 'models/energy_model.pth')
    print("Training completed and model saved.")

if __name__ == "__main__":
    # Path to the synthetic data
    data_path = '/home/artemis/project/composition/my_data/synthetic_data.npy'

    # Training parameters
    epochs = 20
    batch_size = 64
    learning_rate = 1e-3

    train_energy_model(data_path, epochs, batch_size, learning_rate)
