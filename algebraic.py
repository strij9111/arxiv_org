"""
calculation of algebraic features from the attention matrices. Here's the Python code to calculate these features:
"""
import numpy as np

def get_algebraic_features(attention_matrix):
    n = attention_matrix.shape[0]

    # Calculate the sum of the upper triangular part (asymmetry)
    asymmetry = np.sum(np.triu(attention_matrix, k=1)) / (n**2)

    # Calculate mean values of the 3 longest diagonals
    main_diag_mean = np.mean(np.diag(attention_matrix))
    off_diag_1_mean = np.mean(np.diag(attention_matrix, k=1))
    off_diag_minus_1_mean = np.mean(np.diag(attention_matrix, k=-1))

    return asymmetry, main_diag_mean, off_diag_1_mean, off_diag_minus_1_mean

# Example usage
attention_matrix = np.array([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6],
                             [0.7, 0.8, 0.9]])

features = get_algebraic_features(attention_matrix)
print(features)
