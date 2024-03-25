import numpy as np
import math


class Evaluator:
    def __init__(self, true_labels, assignment_points_to_clusters):
        self.true_labels = true_labels
        self.clusters = self._partition_clusters(assignment_points_to_clusters)
        self.true_positive = 0.0
        self.true_negative = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0
        self.confusion_matrix = np.zeros((len(self.clusters), int(np.max(true_labels)) + 1))
        self.compute()

    def _partition_clusters(self, assignment):
        clusters = [[] for _ in range(max(assignment) + 1)]
        for i in range(len(assignment)):
            clusters[int(assignment[i])].append(i)

        return clusters

    def compute(self):
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i])):
                self.confusion_matrix[i, int(self.true_labels[int(self.clusters[i][j])])] += 1

        for i in range(self.confusion_matrix.shape[0]):
            for j in range(self.confusion_matrix.shape[1]):
                if self.confusion_matrix[i, j] > 1:
                    self.true_positive += (self.confusion_matrix[i, j] * (self.confusion_matrix[i, j] - 1)) / 2

                for k in range(j + 1, self.confusion_matrix.shape[1]):
                    self.false_positive += self.confusion_matrix[i, j] * self.confusion_matrix[i, k]

        for j in range(self.confusion_matrix.shape[1]):
            for i in range(self.confusion_matrix.shape[0]):
                for k in range(i + 1, self.confusion_matrix.shape[0]):
                    self.false_negative += self.confusion_matrix[i, j] * self.confusion_matrix[k, j]

                    for m in range(self.confusion_matrix.shape[1]):
                        if m != j:
                            self.true_negative += self.confusion_matrix[i, j] * self.confusion_matrix[k, m]

    def compute_precision(self):
        if self.true_positive + self.false_positive == 0.0:
            return 0.0

        return self.true_positive / (self.true_positive + self.false_positive)

    def compute_recall(self):
        if self.true_positive + self.false_negative == 0.0:
            return 0.0

        return self.true_positive / (self.true_positive + self.false_negative)

    def compute_f1(self):
        precision = self.compute_precision()
        recall = self.compute_recall()

        if precision + recall == 0.0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def compute_conditional_entropy(self):
        conditional_entropy = 0.0

        for i in range(self.confusion_matrix.shape[0]):
            n_elements_in_cluster = sum(self.confusion_matrix[i, :])

            for j in range(self.confusion_matrix.shape[1]):
                if self.confusion_matrix[i, j] != 0:

                    conditional_entropy -= ((self.confusion_matrix[i, j] / n_elements_in_cluster) *
                                            math.log2(self.confusion_matrix[i, j] / n_elements_in_cluster))

        return conditional_entropy

    def computer_accuracy(self):
        return ((self.true_positive + self.true_negative) /
                (self.true_positive + self.true_negative + self.false_positive + self.false_negative))
