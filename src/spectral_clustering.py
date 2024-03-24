import numpy as np
from kernel_functions import *
import math
from scipy.linalg import norm
from sklearn.cluster import KMeans
from Kmeans import KmeansClustering


class SpectralClustering:
    def __init__(self, kernel_function="nn", n_clusters=19):
        self.C = None
        self.Y = None
        self.eigenvalues = None
        self.kernel_function = kernel_function
        self.n_clusters = n_clusters

    def fit(self, D):
        A = compute_nn_similarity(D)

        delta = np.diag(np.sum(A, axis=1))
        assert delta.shape == A.shape, "Degree matrix shape does not match the Data matrix shape."

        # Compute B = L_a = I - delta^-1 . A
        B = np.identity(D.shape[0]) - (np.linalg.inv(delta) @ A)

        unsorted_eigenvalues, unsorted_eigenvectors = np.linalg.eig(B)
        sorted_indices = np.argsort(unsorted_eigenvalues)
        eigenvalues = unsorted_eigenvalues[sorted_indices]
        eigenvectors = unsorted_eigenvectors[:, sorted_indices]
        U = eigenvectors[:, :self.n_clusters]

        Y = U / norm(U, axis=1, keepdims=True)
        print(U.shape, Y.shape)
        assert Y.shape == U.shape, "Y matrix shape does not match the U matrix shape."

        # Store the results to the object fields
        self.eigenvalues = eigenvalues
        self.Y = Y

    def predict(self):
        # kmeans_builtin = KMeans(self.n_clusters)
        # kmeans_builtin.fit(self.Y)
        # clusters_identified = kmeans_builtin.labels_

        kmeans = KmeansClustering(self.n_clusters)
        kmeans.fit(self.Y)
        clusters_identified = kmeans.predict(self.Y)

        return clusters_identified
