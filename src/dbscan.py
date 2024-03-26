import numpy as np
from sklearn.neighbors import NearestNeighbors


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.neighbours = NearestNeighbors(radius=self.eps)
        self.labels = None

    def fit(self, X):
        self.neighbours.fit(X)
        self.labels = np.zeros(X.shape[0])
        cluster = 0
        for i in range(X.shape[0]):
            # if it was already assigned to a cluster
            if self.labels[i] != 0:
                continue
            # if it was unassigned
            # find neighbors within eps distance
            neighbors = self._get_neighbours(X, i)
            # if it is a noise point
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
                continue
            # else it is a core point
            cluster += 1
            self._expand_cluster(X, i, neighbors, cluster)
        return self.labels

    def _expand_cluster(self, X, center_idx, neighbors, cluster):
        self.labels[center_idx] = cluster
        # expand the cluster
        for j in neighbors:
            # if it was noise
            if self.labels[j] == -1:
                self.labels[j] = cluster
            # if it was already assigned to a cluster
            if self.labels[j] != 0:
                continue
            # if it was unassigned
            else:
                self.labels[j] = cluster

            # find neighbors within eps distance to each neighbor
            neighbors2 = self._get_neighbours(X, j)
            if len(neighbors2) >= self.min_samples:
                neighbors = np.concatenate((neighbors, neighbors2))

    def _get_neighbours(self, X, center_idx):
        return self.neighbours.radius_neighbors([X[center_idx]])[1][0]
