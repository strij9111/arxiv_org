"""
Ripser to compute persistent homologies for weighted graphs and point clouds.

In both examples, a distance matrix is used as input, which represents the weighted graph.
You can also use a point cloud (list of points) as input. The examples compute persistent homology
up to dimension 2 (0-dimensional for connected components, 1-dimensional for loops, and 2-dimensional for voids).

The output persistence (GUDHI) and diagrams (Ripser) contain the persistence diagrams, which describe the birth
and death times of topological features (e.g., connected components, loops, voids) across different scales.

"""
import numpy as np
from ripser import ripser

# Define a distance matrix (weighted graph) or a point cloud
distance_matrix = np.array([[0.0, 0.5, 0.8],
                            [0.5, 0.0, 0.9],
                            [0.8, 0.9, 0.0]])

# Compute persistent homology using Ripser
result = ripser(distance_matrix, distance_matrix=True, maxdim=2)

# Extract persistence diagrams
diagrams = result['dgms']

print(diagrams)
