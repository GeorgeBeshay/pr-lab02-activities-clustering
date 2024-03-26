import matplotlib.pyplot as plt
from tabulate import tabulate


class Plotter:
    def __init__(self, k_values, precision_scores, recall_scores,
                 f1_scores, conditional_entropy_scores, accuracy_scores, title):
        self.k_values = k_values
        self.precision_scores = precision_scores
        self.recall_scores = recall_scores
        self.f1_scores = f1_scores
        self.conditional_entropy_scores = conditional_entropy_scores
        self.accuracy_scores = accuracy_scores
        self.title = title

    def plot_scores(self):
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle(self.title, fontsize=32)

        axs[0, 0].plot(self.k_values, self.precision_scores, marker='o')
        axs[0, 0].set_title('Precision Scores')
        axs[0, 0].set_xlabel('k_values')
        axs[0, 0].set_ylabel('Precision')

        axs[0, 1].plot(self.k_values, self.recall_scores, marker='o')
        axs[0, 1].set_title('Recall Scores')
        axs[0, 1].set_xlabel('k_values')
        axs[0, 1].set_ylabel('Recall')

        axs[1, 0].plot(self.k_values, self.f1_scores, marker='o')
        axs[1, 0].set_title('F1 Scores')
        axs[1, 0].set_xlabel('k_values')
        axs[1, 0].set_ylabel('F1')

        axs[1, 1].plot(self.k_values, self.conditional_entropy_scores, marker='o')
        axs[1, 1].set_title('Conditional Entropy Scores')
        axs[1, 1].set_xlabel('k_values')
        axs[1, 1].set_ylabel('Conditional Entropy')

        axs[2, 0].plot(self.k_values, self.accuracy_scores, marker='o')
        axs[2, 0].set_title('Accuracy Scores')
        axs[2, 0].set_xlabel('k_values')
        axs[2, 0].set_ylabel('Accuracy')

        axs[2, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def tabulate_scores(self):
        data = {
            'k_values': self.k_values,
            'Precision': self.precision_scores,
            'Recall': self.recall_scores,
            'F1': self.f1_scores,
            'Conditional Entropy': self.conditional_entropy_scores,
            'Accuracy': self.accuracy_scores
        }
        headers = ['k_values', 'Precision', 'Recall', 'F1', 'Conditional Entropy', 'Accuracy']
        table = tabulate(data, headers=headers, tablefmt='grid')
        print(table)
