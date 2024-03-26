import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_nn_similarity(data, n_neighbors=4):
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(data)
    _, indices = nn.kneighbors(data)  # each of those arrays, is of dims len(data) x n_neighbors
    indices = np.delete(indices, 0, axis=1)

    num_points = len(data)
    similarity_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in indices[i]:
            similarity_matrix[i, j] = 1
            similarity_matrix[j, i] = 1

    return similarity_matrix
