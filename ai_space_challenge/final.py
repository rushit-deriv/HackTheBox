import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

distance_matrix = np.load('distance_matrix.npy')
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
points = mds.fit_transform(distance_matrix)

plt.figure(figsize=(20, 15))
plt.scatter(points[:, 0], points[:, 1], alpha=0.8, s=10, c='black')
plt.title('High Contrast MDS Visualization')
plt.savefig('high_contrast_flag.png', dpi=300, bbox_inches='tight')