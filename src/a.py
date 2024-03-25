


import numpy as np
from sklearn.preprocessing import StandardScaler

import data

dataLoader = data.DataLoader()

pure_dataset = dataLoader.get_pure_data()
solution_a_dataset = dataLoader.get_modified_data_solution_a()
# solution_b_dataset_before_projection = dataLoader._get_modified_data_solution_b_before_projection()
# solution_b_dataset = dataLoader.get_modified_data_solution_b()
x_train, y_train, x_test, y_test = dataLoader.get_splitted_data(solution_a=True)
# x_train2, y_train2, x_test2, y_test2 = dataLoader.get_splitted_data(solution_a=False)

print(f"Pure Dataset Dims:                           {pure_dataset.shape}")
print(f"Solution A Dataset Dims:                     {solution_a_dataset.shape}")
# print(f"Solution B (before projection) Dataset Dims: {solution_b_dataset_before_projection.shape}")
# print(f"Solution B (after projection) Dataset Dims:  {solution_b_dataset.shape}\n")
print(f"Train and Test Splits (solution A) Dims:  \n"
      f"\tx_train.shape =              {x_train.shape}\n"
      f"\ty_train.shape =              {y_train.shape}\n"
      f"\tx_test.shape =               {x_test.shape}\n"
      f"\ty_test.shape =               {y_test.shape}\n")
# print(f"Train and Test Splits (solution B) Dims:  \n"
#       f"\tx_train.shape =              {x_train2.shape}\n"
#       f"\ty_train.shape =              {y_train2.shape}\n"
#       f"\tx_test.shape =               {x_test2.shape}\n"
#       f"\ty_test.shape =               {y_test2.shape}\n")

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        self.labels_ = np.zeros(X.shape[0])
        self.cluster_id = 0

        for i in range(X.shape[0]):
            if self.labels_[i] != 0:
                continue
            neighbors = self._find_neighbors(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                self.cluster_id += 1
                self._expand_cluster(X, i, neighbors, self.cluster_id)

        return self.labels_

    def _find_neighbors(self, X, center_idx):
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(X[center_idx] - X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, center_idx, neighbors, cluster_id):
        self.labels_[center_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            elif self.labels_[neighbor_idx] == 0:
                self.labels_[neighbor_idx] = cluster_id
                new_neighbors = self._find_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)
            i += 1

# Example usage:
if __name__ == "__main__":
    # Create DBSCAN instance and fit the data
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(x_train)

    # Print accuracy
    correct = 0
    for i in range(len(labels)):
        if labels[i] == y_train[i]:
            correct += 1
    accuracy = correct / len(labels)
    print(f"Accuracy: {accuracy}")


