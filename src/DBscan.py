import numpy as np
from sklearn.neighbors import NearestNeighbors

import data

dataLoader = data.DataLoader()

pure_dataset = dataLoader.get_pure_data()
solution_a_dataset = dataLoader.get_modified_data_solution_a()
solution_b_dataset_before_projection = dataLoader._get_modified_data_solution_b_before_projection()
solution_b_dataset = dataLoader.get_modified_data_solution_b()
x_train, y_train, x_test, y_test = dataLoader.get_splitted_data(solution_a=True)
x_train2, y_train2, x_test2, y_test2 = dataLoader.get_splitted_data(solution_a=False)

print(f"Pure Dataset Dims:                           {pure_dataset.shape}")
print(f"Solution A Dataset Dims:                     {solution_a_dataset.shape}")
print(f"Solution B (before projection) Dataset Dims: {solution_b_dataset_before_projection.shape}")
print(f"Solution B (after projection) Dataset Dims:  {solution_b_dataset.shape}\n")
print(f"Train and Test Splits (solution A) Dims:  \n"
      f"\tx_train.shape =              {x_train.shape}\n"
      f"\ty_train.shape =              {y_train.shape}\n"
      f"\tx_test.shape =               {x_test.shape}\n"
      f"\ty_test.shape =               {y_test.shape}\n")
print(f"Train and Test Splits (solution B) Dims:  \n"
      f"\tx_train.shape =              {x_train2.shape}\n"
      f"\ty_train.shape =              {y_train2.shape}\n"
      f"\tx_test.shape =               {x_test2.shape}\n"
      f"\ty_test.shape =               {y_test2.shape}\n")


# DBSCAN
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


dbscan = DBSCAN(eps=2.1, min_samples=40)
labels = dbscan.fit(x_train)
labels_2 = dbscan.fit(x_train2)


# Function to map predicted labels to given training labels
def map_labels(predicted_labels, true_labels):
    # Convert true_labels to integer type
    true_labels = true_labels.astype(int)

    # Exclude noise points (label -1)
    non_noise_mask = predicted_labels != -1
    predicted_labels = predicted_labels[non_noise_mask]
    true_labels = true_labels[non_noise_mask]
    # Create a dictionary to store mapping of predicted label to true label
    label_map = {}
    unique_true_labels = np.unique(true_labels)
    for predicted_label in np.unique(predicted_labels):
        # Extract indices where predicted label matches
        indices = np.where(predicted_labels == predicted_label)[0]
        # Count occurrences of true labels in these indices
        counts = np.bincount(true_labels[indices], minlength=len(unique_true_labels))
        # Get the true label with maximum occurrences
        mapped_label = np.argmax(counts)
        # Map the predicted label to this true label
        label_map[predicted_label] = mapped_label

    # Map the predicted labels using the created dictionary
    mapped_labels = np.vectorize(label_map.get)(predicted_labels)
    return mapped_labels


# Map the predicted labels to the given training labels
mapped_labels = map_labels(labels, y_train)
mapped_labels_2 = map_labels(labels_2, y_train2)


# print accuracy
def print_accuracy(mapped_labels, y_train):
    correct = 0
    for i in range(len(mapped_labels)):
        if mapped_labels[i] == y_train[i]:
            correct += 1
    accuracy = correct / len(mapped_labels)
    print(f"Accuracy on Test Data: {accuracy * 100}")


print_accuracy(mapped_labels, y_train)
print_accuracy(mapped_labels_2, y_train2)

# implement DBscan using sklearn
from sklearn.cluster import DBSCAN as dbscan_sklearn

# Compute DBSCAN
db = dbscan_sklearn(eps=2.1, min_samples=40).fit(x_train)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

db = dbscan_sklearn(eps=2.1, min_samples=40).fit(x_train2)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels_2 = db.labels_


# Apply mapping to get training labels for predicted labels
mapped_labels = map_labels(labels, y_train)
mapped_labels_2 = map_labels(labels_2, y_train2)

# print accuracy
print_accuracy(mapped_labels, y_train)
print_accuracy(mapped_labels_2, y_train2)
