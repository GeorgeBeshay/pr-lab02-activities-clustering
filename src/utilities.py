import numpy as np
from evaluator import Evaluator
from tabulate import tabulate
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import random


def show_evaluation(evaluators: List[Evaluator], model_names: List[str]):
    headers = ["Model Name", "Precision", "Recall", "F1-Score", "Conditional Entropy", "Accuracy"]
    metrics_data = []
    for model_idx, model_name in enumerate(model_names):
        metrics_data.append([
            model_name,
            evaluators[model_idx].compute_precision(),
            evaluators[model_idx].compute_recall(),
            evaluators[model_idx].compute_f1(),
            evaluators[model_idx].compute_conditional_entropy(),
            evaluators[model_idx].computer_accuracy()
        ])

    # Tabulate the metrics values
    print(tabulate(metrics_data, headers=headers, tablefmt="grid"))

    df = pd.DataFrame(metrics_data, columns=headers)

    # Set 'Model Name' column as index for easy plotting
    df.set_index('Model Name', inplace=True)

    num_metrics = len(df.columns)
    num_plots = num_metrics // 2 + num_metrics % 2  # Calculate the number of plots based on metrics

    fig, axes = plt.subplots(nrows=num_plots, ncols=2, figsize=(18, num_plots * 10))

    for idx, column in enumerate(df.columns):
        row_idx = idx // 2
        col_idx = idx % 2
        df[column].plot(kind='barh', ax=axes[row_idx, col_idx], color='red')
        axes[row_idx, col_idx].set_title(column + ' Comparison', fontsize=16)  # Set title fontsize
        axes[row_idx, col_idx].set_xlabel(column, fontsize=14)  # Set xlabel fontsize
        axes[row_idx, col_idx].set_ylabel('Model Name', fontsize=14)  # Set ylabel fontsize
        axes[row_idx, col_idx].tick_params(axis='x', rotation=90, labelsize=12)  # Set xtick fontsize

    axes[num_plots-1, 1].axis('off')
    plt.tight_layout()
    plt.show()


def plot_summary(evaluators: List[Evaluator], model_names: List[str]):
    headers = ["Model Name", "Precision", "Recall", "F1-Score", "Accuracy"]
    metrics_data = []
    for model_idx, model_name in enumerate(model_names):
        metrics_data.append([
            model_name,
            evaluators[model_idx].compute_precision(),
            evaluators[model_idx].compute_recall(),
            evaluators[model_idx].compute_f1(),
            evaluators[model_idx].computer_accuracy()
        ])

    df = pd.DataFrame(metrics_data, columns=headers)
    df.set_index('Model Name', inplace=True)

    plt.figure(figsize=(18, 10))
    for model_name in model_names:
        plt.plot(headers[1:], df.loc[model_name][0:], marker='o', linestyle='-', label=model_name)

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Evaluation Metrics')
    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.legend()
    plt.grid(True)
    plt.show()


def filter_dbscan_clusters(clusters_indices: np.ndarray):
    max_cluster_idx = np.max(clusters_indices)
    new_cluster_idx = max_cluster_idx + 1

    for i in range(len(clusters_indices)):
        if clusters_indices[i] == -1:
            clusters_indices[i] = new_cluster_idx
            new_cluster_idx += 1

    return clusters_indices


def compare_detected_clusters(evaluators: List[Evaluator], model_names: List[str]):
    selected_models_indices = [1, 3, 5, 7]
    evaluators = [evaluators[i] for i in selected_models_indices]
    model_names = [model_names[i] for i in selected_models_indices]

    headers = [f'Samples Identified In Cluster {i}' for i in range(1, 20)]
    clusters_identified = np.zeros((len(evaluators), 19))

    for model_idx, model_name in enumerate(model_names):
        temp_conf = evaluators[model_idx].confusion_matrix
        max_clusters = np.max(temp_conf, axis=0)
        if max_clusters.shape[0] > 19:
            max_clusters = np.delete(max_clusters, 0)

        clusters_identified[model_idx] = max_clusters

    plt.figure(figsize=(14, 6))
    for row in clusters_identified:
        plt.plot(range(len(row)), row)

    # Add labels and title
    plt.xlabel('Activity Index')
    plt.ylabel('Samples Identified Per Activity')
    plt.title('Samples Identified Per Activity overall Models')
    plt.xticks(range(0, 19))
    plt.grid(True)

    # Add legend indicating row number
    plt.legend(model_names)

    # Show the plot
    plt.show()

