import numpy as np
from kernel_functions import *
import math
from scipy.linalg import norm
from sklearn.cluster import KMeans
from kmeans import KmeansClustering
from kernel_functions import *

class SpectralClustering:
    def __init__(self, kernel_function=compute_nn_similarity, n_clusters=19):
        self.Y = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.kernel_function = kernel_function
        self.n_clusters = n_clusters
        self.n_connected_components = None

    def fit(self, D):
        A = self.kernel_function(D)

        delta = np.diag(np.sum(A, axis=1))
        assert delta.shape == A.shape, "Degree matrix shape does not match the Data matrix shape."

        # Compute B = L_a = I - delta^-1 . A
        delta_inv = np.diag(1 / np.diagonal(delta))
        B = np.identity(D.shape[0]) - (delta_inv @ A)

        unsorted_eigenvalues, unsorted_eigenvectors = np.linalg.eig(B)
        sorted_indices = np.argsort(unsorted_eigenvalues)
        eigenvalues = np.real(unsorted_eigenvalues[sorted_indices])
        eigenvectors = np.real(unsorted_eigenvectors[:, sorted_indices])

        self.n_connected_components = np.count_nonzero(eigenvalues < 1e-12)
        selected_indices = eigenvalues > 1e-12
        selected_eigenvalues = eigenvalues[selected_indices]
        selected_eigenvectors = eigenvectors[selected_indices]

        U = selected_eigenvectors[:, :self.n_clusters - 1]
        Y = U / (norm(U, axis=1, keepdims=True) + 1e-10)

        assert Y.shape == U.shape, "Y matrix shape does not match the U matrix shape."

        # Store the results to the object fields
        self.eigenvalues = selected_eigenvalues
        self.eigenvectors = selected_eigenvectors
        self.Y = Y

    def predict(self):
        kmeans = KmeansClustering(self.n_clusters)
        kmeans.fit(self.Y)
        clusters_identified = kmeans.predict(self.Y)

        return clusters_identified
