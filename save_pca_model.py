"""
Script to train PCA model and save parameters for oil classification GUI
This script loads the cleaned data, trains the PCA model, and saves the parameters
that the GUI needs to transform features into PC space.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# Load the cleaned data (after multicollinearity removal)
print("Loading data from Oil_Quality_Dataset_2025/cleaned_data_no_multicollinearity.csv...")
df = pd.read_csv('Oil_Quality_Dataset_2025/cleaned_data_no_multicollinearity.csv')

# Define feature columns (12 features after multicollinearity removal)
feature_columns = [
    'color_bbox_mean_r', 'color_bbox_mean_g', 'color_poly_mean_b',
    'color_poly_std_b', 'hsv_bbox_mean_h', 'hsv_bbox_mean_s',
    'hsv_poly_mean_h', 'hsv_poly_mean_s', 'hsv_poly_mean_v',
    'color_bbox_entropy', 'color_poly_entropy', 'ir_cluster'
]

# Extract features
X = df[feature_columns].values
y = df['level2_class'].values

print(f"Data shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")

# Standardize the features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA with 3 components
print("Performing PCA with 3 components...")
n_components = 3
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Print variance explained
print(f"\nVariance explained by each component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.2f}%")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Calculate centroids for each class in PCA space
print("\nCalculating centroids for each class...")
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca['Oil_Type'] = y

centroids = {}
for oil_type in np.unique(y):
    mask = df_pca['Oil_Type'] == oil_type
    centroid = df_pca[mask][[f'PC{i+1}' for i in range(n_components)]].mean().values
    centroids[oil_type] = centroid
    print(f"  {oil_type}: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")

# Prepare model data for saving
model_data = {
    'mean': scaler.mean_,           # Mean of each feature (30 values)
    'std': scaler.scale_,            # Std of each feature (30 values)
    'components': pca.components_,   # PCA components (3x30 matrix)
    'explained_variance_ratio': pca.explained_variance_ratio_,
    'centroids': centroids,
    'feature_columns': feature_columns,
    'n_components': n_components
}

# Save to pickle file
output_file = 'pca_model_params.pkl'
print(f"\nSaving PCA model parameters to {output_file}...")
with open(output_file, 'wb') as f:
    pickle.dump(model_data, f)

print(f"✓ Model parameters saved successfully!")

# Verify the saved file
print("\nVerifying saved file...")
with open(output_file, 'rb') as f:
    loaded_data = pickle.load(f)

print(f"  Mean shape: {loaded_data['mean'].shape}")
print(f"  Std shape: {loaded_data['std'].shape}")
print(f"  Components shape: {loaded_data['components'].shape}")
print(f"  Number of centroids: {len(loaded_data['centroids'])}")
print(f"  Centroid classes: {list(loaded_data['centroids'].keys())}")

print("\n✓ All done! The GUI can now load pca_model_params.pkl")
