import numpy as np
import matplotlib.pyplot as plt

# Load the distance matrix
print("Loading distance matrix...")
distance_matrix = np.load('distance_matrix.npy')

# Display basic information
print("Matrix shape:", distance_matrix.shape)
print("Matrix data type:", distance_matrix.dtype)
print("Min value:", np.min(distance_matrix))
print("Max value:", np.max(distance_matrix))

# Basic distance matrix visualization
plt.figure(figsize=(15, 12))
plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Distance')
plt.title('Distance Matrix Visualization')
plt.savefig('basic_matrix.png', dpi=300, bbox_inches='tight')
print("Basic matrix visualization saved")

# Try different colormaps that might reveal hidden text/patterns
colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'gray', 'bone', 'copper', 'pink', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

for cmap in colormaps:
    plt.figure(figsize=(15, 12))
    plt.imshow(distance_matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Distance')
    plt.title(f'Distance Matrix - {cmap} colormap')
    plt.savefig(f'matrix_{cmap}.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization with {cmap} colormap")

# Try different interpolation methods
interpolations = ['nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

for interp in interpolations:
    plt.figure(figsize=(15, 12))
    plt.imshow(distance_matrix, cmap='hot', interpolation=interp)
    plt.colorbar(label='Distance')
    plt.title(f'Distance Matrix - {interp} interpolation')
    plt.savefig(f'matrix_interp_{interp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization with {interp} interpolation")

# Try different contrast adjustments
contrasts = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0]

for contrast in contrasts:
    adjusted_matrix = distance_matrix * contrast
    plt.figure(figsize=(15, 12))
    plt.imshow(adjusted_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Distance')
    plt.title(f'Distance Matrix - contrast {contrast}')
    plt.savefig(f'matrix_contrast_{contrast}.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization with contrast {contrast}")

# Try thresholding to reveal patterns
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for thresh in thresholds:
    max_val = np.max(distance_matrix)
    threshold_val = max_val * thresh
    binary_matrix = (distance_matrix > threshold_val).astype(float)
    
    plt.figure(figsize=(15, 12))
    plt.imshow(binary_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Binary')
    plt.title(f'Binary threshold at {thresh} of max value')
    plt.savefig(f'matrix_threshold_{thresh}.png', dpi=300, bbox_inches='tight')
    print(f"Saved binary threshold visualization at {thresh}")

# Try log scaling
log_matrix = np.log1p(distance_matrix)  # log(1 + x) to handle zeros
plt.figure(figsize=(15, 12))
plt.imshow(log_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Log Distance')
plt.title('Log-scaled Distance Matrix')
plt.savefig('matrix_log_scaled.png', dpi=300, bbox_inches='tight')
print("Saved log-scaled visualization")

# Try negative values (invert)
inverted_matrix = np.max(distance_matrix) - distance_matrix
plt.figure(figsize=(15, 12))
plt.imshow(inverted_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(label='Inverted Distance')
plt.title('Inverted Distance Matrix')
plt.savefig('matrix_inverted.png', dpi=300, bbox_inches='tight')
print("Saved inverted visualization")

print("All visualizations completed! Check the generated PNG files for hidden flags or text patterns.")
print("Look especially at the different colormap and threshold visualizations - flags are often hidden as patterns in the data.") 