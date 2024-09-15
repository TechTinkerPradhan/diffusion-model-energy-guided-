# data/generate_data.py

import numpy as np
import os

def generate_synthetic_data(num_samples_per_class=500, num_objects=10, num_features=6):
    """
    Generates synthetic data for training.

    Args:
        num_samples_per_class (int): Number of positive and negative samples to generate.
        num_objects (int): Number of objects in each scene.
        num_features (int): Number of features per object.

    Returns:
        data (list): A list of dictionaries containing scene features and labels.
    """
    data = []

    # Generate positive samples (plausible scenes)
    for _ in range(num_samples_per_class):
        # Simulate plausible object features
        # For positive samples, objects are placed in realistic configurations
        features = np.random.uniform(low=0.0, high=1.0, size=(num_objects, num_features))
        positive_scene = {
            'features': features,  # Shape: (num_objects, num_features)
            'label': 1             # Label: 1 for positive samples
        }
        data.append(positive_scene)

    # Generate negative samples (implausible scenes)
    for _ in range(num_samples_per_class):
        # Simulate implausible object features
        # For negative samples, introduce anomalies in object configurations
        features = np.random.uniform(low=-1.0, high=2.0, size=(num_objects, num_features))
        negative_scene = {
            'features': features,  # Shape: (num_objects, num_features)
            'label': 0             # Label: 0 for negative samples
        }
        data.append(negative_scene)

    return data

if __name__ == "__main__":
    print("Starting data generation...")

    # Parameters
    num_samples_per_class = 500  # Total samples will be 2 * num_samples_per_class
    num_objects = 10             # Number of objects in each scene
    num_features = 6             # Number of features per object

    # Generate the synthetic data
    data = generate_synthetic_data(num_samples_per_class, num_objects, num_features)
    print("generation was done..")

    # Convert the data list to a NumPy array with object data type
    data_array = np.array(data, dtype=object)

    # Ensure the 'data/' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the data to a .npy file
    np.save('data/synthetic_data.npy', data_array)

    print(f"Synthetic data generated and saved to 'data/synthetic_data.npy'.")
