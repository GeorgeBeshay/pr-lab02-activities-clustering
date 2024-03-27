# Pattern Recognition Assignment 2: Daily and Sports Activity Clustering

## Introduction
In today’s era of data-driven decision-making, the ability to identify patterns, group similar entities, and extract insights from large datasets has become indispensable across various domains. Clustering, a fundamental technique in unsupervised machine learning, plays a key role in organizing data into meaningful groups based on similarities. This assignment aims to explore the application of clustering algorithms, specifically K-Means, Spectral Clustering, and DBSCAN, for the detection of daily and sports activities captured using motion sensors.

## Problem Statement
The task involves analyzing motion sensor data of various daily and sports activities performed by multiple subjects. The dataset comprises motion sensor data collected over 5 minutes for 19 activities, each performed by 8 subjects. Each 5-minute signal is divided into 5-second segments, resulting in an overall 9120 data points. The objective is to preprocess the dataset, apply clustering algorithms, evaluate the clustering results, and compare different algorithms based on their ability to accurately detect activities.

## Dataset
The "Daily and Sports Activity" dataset contains motion sensor data collected from 5 Xsens MTx units placed on the torso, arms, and legs of 8 subjects. Each segment consists of 125 samples acquired from 9 sensors over 5 seconds. In our implementation, we have 2 versions of the dataset depending on the type of preprocessing made on it: Solution A involves taking the mean of each column in each segment, resulting in 45 features per data point. Solution B involves flattening all features, resulting in 5625 features per data point before dimensionality reduction. The dataset is then split into training and testing sets for evaluation. (however not all of the algorithms require 'training').

## Implementation Overview
1. **Data Preprocessing**: Analyze and preprocess the dataset to prepare it for clustering, with its two versions.
2. **K-Means Clustering**: Implement K-Means clustering on both Solution A and Solution B datasets for different values of K.
3. **Spectral Clustering**: Implement Spectral Clustering algorithm on both solutions and evaluate its performance.
4. **DBSCAN Clustering**: Apply DBSCAN algorithm on both solutions and analyze the clustering results.
5. **Evaluation**: Evaluate clustering models using metrics such as precision, recall, F1 score, and conditional entropy.
7. **Comparison and Analysis**: Compare the performance of different clustering algorithms and provide insights into their strengths and weaknesses.

## Repository Structure
- **data.py**: Module for loading and preprocessing the dataset.
- **kmeans.py**: Implementation of the K-Means clustering algorithm.
- **spectral_clustering.py**: Implementation of Spectral Clustering algorithm.
- **dbscan.py**: Implementation of DBSCAN algorithm.
- **evaluator.py**: Module for evaluating clustering models using various metrics.
- **plotter.py**: Module for visualizing clustering results.
- **utilities.py**: Utility functions used in the analysis.

## Authors
- [Mariam Aziz](https://github.com/MariamAziz0)
- [Makario Michel](https://github.com/Mak-Michel)
- [George Beshay](https://github.com/GeorgeBeshay)

## License
- [License](LICENSE)
---

This assignment is part of the `CSE 352: Pattern Recognition` course offered by the department of Computer and Systems Engineering, University of Alexandria in Spring 2023 / 2024.
