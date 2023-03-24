"""
TDA for weighted graphs and point clouds: You can use Python libraries like GUDHI or Ripser to compute persistent
homologies for weighted graphs and point clouds.

In both examples, a distance matrix is used as input, which represents the weighted graph.
You can also use a point cloud (list of points) as input. The examples compute persistent homology
up to dimension 2 (0-dimensional for connected components, 1-dimensional for loops, and 2-dimensional for voids).

The output persistence (GUDHI) and diagrams (Ripser) contain the persistence diagrams, which describe the birth
and death times of topological features (e.g., connected components, loops, voids) across different scales.

"""
import numpy as np
import gudhi as gd

# Define a distance matrix (weighted graph) or a point cloud
distance_matrix = np.array([[0.0, 0.5, 0.8],
                            [0.5, 0.0, 0.9],
                            [0.8, 0.9, 0.0]])

# Initialize Rips complex from the distance matrix
rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)

# Compute the simplex tree and persistent homology
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
persistence = simplex_tree.persistence()

print(persistence)
