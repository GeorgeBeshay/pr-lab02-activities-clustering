import numpy as np
import matplotlib.pyplot as plt


class KmeansClustering:
    def __init__(self, k=8, max_iterations=500, random_state=42):
        self.clusters = None
        self.centroids = None
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(range(len(X)), size=self.k, replace=False)]

        for i in range(self.max_iterations):

            new_assignment = self._assignment(X)
            new_centroids = self._compute_new_centroids(X, new_assignment)

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def _assignment(self, X):
        centroids_with_new_axis = self.centroids[:, np.newaxis]
        difference_points_centroids = X - centroids_with_new_axis
        distances = np.sqrt((difference_points_centroids ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _compute_new_centroids(self, X, new_assignment):
        centers = []
        for i in range(self.k):
            if np.sum(new_assignment == i) == 0:
                centers.append(X[np.random.choice(X.shape[0])])
            else:
                centers.append(X[new_assignment == i].mean(axis=0))
        return np.array(centers)

    def predict(self, X):
        return self._assignment(X)
