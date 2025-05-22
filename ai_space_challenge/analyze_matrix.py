import numpy as np
import matplotlib.pyplot as plt

# Load the distance matrix
try:
    distance_matrix = np.load('distance_matrix.npy')
    
    # Display basic information about the matrix
    print("Matrix shape:", distance_matrix.shape)
    print("Matrix data type:", distance_matrix.dtype)
    print("Matrix size in memory:", distance_matrix.nbytes / (1024 * 1024), "MB")
    print("Min value:", np.min(distance_matrix))
    print("Max value:", np.max(distance_matrix))
    
    # Check if it's a square matrix (which would be typical for a distance matrix)
    if len(distance_matrix.shape) == 2 and distance_matrix.shape[0] == distance_matrix.shape[1]:
        print("This is a square matrix, which is typical for a distance matrix between points")
        n_points = distance_matrix.shape[0]
        print(f"The matrix represents distances between {n_points} points")
    
    # Analyze the structure of the matrix
    is_symmetric = np.allclose(distance_matrix, distance_matrix.T, rtol=1e-5, atol=1e-8)
    print("Is the matrix symmetric (indicating an undirected graph):", is_symmetric)
    
    # Check if diagonal is zero (as expected in a distance matrix)
    diagonal_is_zero = np.allclose(np.diag(distance_matrix), 0, rtol=1e-5, atol=1e-8)
    print("Does the diagonal contain zeros (as expected for a distance matrix):", diagonal_is_zero)
    
    # Try to visualize the matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='viridis')
    plt.colorbar(label='Distance')
    plt.title('Distance Matrix Visualization')
    plt.savefig('distance_matrix_viz.png')
    print("Matrix visualization saved as 'distance_matrix_viz.png'")
    
    # Look for patterns - are there distinct clusters?
    print("\nLooking for patterns in the data...")
    
    # If the matrix is very large, we might want to try dimensionality reduction
    if distance_matrix.shape[0] > 1000:
        from sklearn.manifold import MDS
        # Multi-dimensional scaling to visualize the points in 2D
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        
        # If the matrix is extremely large, subsample for visualization
        max_points = 1000
        if distance_matrix.shape[0] > max_points:
            indices = np.random.choice(distance_matrix.shape[0], max_points, replace=False)
            subset_matrix = distance_matrix[np.ix_(indices, indices)]
            points = mds.fit_transform(subset_matrix)
            print(f"Visualizing a subset of {max_points} points due to large matrix size")
        else:
            points = mds.fit_transform(distance_matrix)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(points[:, 0], points[:, 1], alpha=0.6)
        plt.title('MDS visualization of distance matrix')
        plt.savefig('distance_matrix_mds.png')
        print("MDS visualization saved as 'distance_matrix_mds.png'")
    
except Exception as e:
    print(f"Error analyzing the distance matrix: {e}") 