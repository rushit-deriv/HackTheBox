import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

# Load the distance matrix
print("Loading distance matrix...")
distance_matrix = np.load('distance_matrix.npy')

# Display basic information
print("Matrix shape:", distance_matrix.shape)
print("Matrix data type:", distance_matrix.dtype)
print("Min value:", np.min(distance_matrix))
print("Max value:", np.max(distance_matrix))

# Basic distance matrix visualization
plt.figure(figsize=(12, 10))
plt.imshow(distance_matrix, cmap='viridis')
plt.colorbar(label='Distance')
plt.title('Distance Matrix Visualization')
plt.savefig('distance_matrix_viz.png')
print("Basic matrix visualization saved")

# Dimensionality reduction techniques to find hidden patterns
print("Performing dimensionality reduction to find hidden patterns...")

# Create several variations of visualizations to ensure we catch the flag
visualization_methods = [
    ("MDS", MDS(n_components=2, dissimilarity='precomputed', random_state=42), "distance_matrix_mds.png"),
    ("t-SNE", TSNE(n_components=2, metric='precomputed', random_state=42), "distance_matrix_tsne.png")
]

for name, model, filename in visualization_methods:
    print(f"Generating {name} visualization...")
    try:
        points = model.fit_transform(distance_matrix)
        
        # Generate regular scatter plot
        plt.figure(figsize=(12, 10))
        plt.scatter(points[:, 0], points[:, 1], alpha=0.6, s=5)
        plt.title(f'{name} visualization of distance matrix')
        plt.savefig(filename)
        
        # Generate density plot with contours - this often reveals hidden text
        plt.figure(figsize=(12, 10))
        sns.kdeplot(x=points[:, 0], y=points[:, 1], cmap="viridis", fill=True, thresh=0.05)
        plt.title(f'{name} density plot (may reveal hidden text)')
        plt.savefig(f'{name}_density.png')
        
        # Try different kernel density settings
        plt.figure(figsize=(12, 10))
        sns.kdeplot(x=points[:, 0], y=points[:, 1], cmap="hot", fill=True, levels=30)
        plt.title(f'{name} enhanced density plot')
        plt.savefig(f'{name}_enhanced_density.png')
        
        # Generate heatmap
        plt.figure(figsize=(12, 10))
        h = plt.hist2d(points[:, 0], points[:, 1], bins=100, cmap='viridis')
        plt.colorbar(h[3], label='Density')
        plt.title(f'{name} heatmap visualization')
        plt.savefig(f'{name}_heatmap.png')
        
        print(f"{name} visualizations completed")
    except Exception as e:
        print(f"Error with {name}: {e}")

# Try another approach: convert distance matrix to similarity and visualize
print("Generating similarity-based visualizations...")
similarity = 1 / (1 + distance_matrix)
np.fill_diagonal(similarity, 0)  # Reset diagonal

plt.figure(figsize=(12, 10))
plt.imshow(similarity, cmap='plasma')
plt.colorbar(label='Similarity')
plt.title('Similarity Matrix Visualization')
plt.savefig('similarity_matrix_viz.png')

# Apply PCA to the similarity matrix
print("Applying PCA to similarity matrix...")
pca = PCA(n_components=2)
sim_points = pca.fit_transform(similarity)

plt.figure(figsize=(12, 10))
plt.scatter(sim_points[:, 0], sim_points[:, 1], alpha=0.6, s=5)
plt.title('PCA visualization of similarity matrix')
plt.savefig('similarity_pca.png')

# Density plot that may reveal the flag
plt.figure(figsize=(12, 10))
sns.kdeplot(x=sim_points[:, 0], y=sim_points[:, 1], cmap="viridis", fill=True)
plt.title('PCA density plot (may reveal hidden flag)')
plt.savefig('similarity_pca_density.png')

# Try different color schemes for better visibility
for cmap in ['hot', 'coolwarm', 'jet']:
    plt.figure(figsize=(12, 10))
    sns.kdeplot(x=sim_points[:, 0], y=sim_points[:, 1], cmap=cmap, fill=True, levels=30)
    plt.title(f'PCA density plot with {cmap} colormap')
    plt.savefig(f'similarity_pca_density_{cmap}.png')

print("All visualizations completed. Check the generated images for the hidden flag!") 