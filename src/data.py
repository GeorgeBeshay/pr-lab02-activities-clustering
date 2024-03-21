import os
import numpy as np
from sklearn.decomposition import PCA


class DataLoader:
    """
    A class to load, process and prepare the data with the desired properties (pure, solution A and solution B).

    Attributes:
    root_dir : str
        The root directory containing the dataset folders.
    num_activities : int
        The number of activities in the dataset.
    num_subjects : int
        The number of subjects for each activity.
    num_samples : int
        The number of samples for each subject.
    rows : int
        The number of rows in each sample.
    columns : int
        The number of columns in each sample.
    data : numpy.ndarray
        The loaded data, shape is (num_subjects, rows, columns).
    """
    def __init__(self, root_dir=os.getcwd() + '\\data', num_activities=19, num_subjects=8, num_samples=60,
                 rows=125, columns=45):
        """
        Initializes the DataLoader object with default or provided parameters.

        **Parameter**:\n
        root_dir : str, optional
            The root directory containing the dataset folders. Default is current working directory + '/data'.
        num_activities : int, optional
            The number of activities in the dataset. Default is 19.
        num_subjects : int, optional
            The number of subjects for each activity. Default is 8.
        num_samples : int, optional
            The number of samples for each subject. Default is 60.
        rows : int, optional
            The number of rows in each sample. Default is 125.
        columns : int, optional
            The number of columns in each sample. Default is 45.
        """
        self.root_dir = root_dir
        self.num_activities = num_activities
        self.num_subjects = num_subjects
        self.num_samples = num_samples
        self.rows = rows
        self.columns = columns
        self.data = self.load_data()

    def load_data(self):
        """
        Loads the data from the dataset folders into a 3D numpy array, the 3 dimensions
        correspond to the number of samples, number of rows per sample, and number of columns per row.

        Returns:
        numpy.ndarray:
            A 3D numpy array representing the loaded data.
        """
        data = np.zeros(
            (self.num_activities, self.num_subjects, self.num_samples, self.rows, self.columns)
        )  # 5D Data Matrix
        for activity_idx in range(1, self.num_activities + 1):
            activity_folder = os.path.join(self.root_dir, f"a{activity_idx:02d}")
            for subject_idx in range(1, self.num_subjects + 1):
                subject_folder = os.path.join(activity_folder, f"p{subject_idx}")
                for sample_idx in range(1, self.num_samples + 1):
                    sample_file = os.path.join(subject_folder, f"s{sample_idx:02d}.txt")
                    with open(sample_file, 'r') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            data[activity_idx - 1, subject_idx - 1, sample_idx - 1, i] = list(
                                map(float, line.split(',')))
        data = np.reshape(data, (self.num_activities * self.num_subjects * self.num_samples, self.rows, self.columns))
        return data

    def get_sample(self, activity_idx, subject_idx, sample_idx):
        """
        Retrieves a specific sample (1-index based) from the loaded data.

        **Parameter**:\n
        activity_idx : int
            Index of the activity.
        subject_idx : int
            Index of the subject.
        sample_idx : int
            Index of the sample.

        Returns:
        numpy.ndarray:
            The requested sample of dims 125 x 45
        """
        return self.data[
            ((activity_idx - 1) * 8 * 60) +
            ((subject_idx - 1) * 60) +
            (sample_idx - 1)]

    def get_pure_data(self):
        """
        Retrieves the loaded data without any modification (dims = 9120 x 125 x 45).

        Returns:
        numpy.ndarray:
            The loaded data.
        """
        return self.data

    def get_modified_data_solution_a(self):
        """
        Retrieves the modified data using solution A (mean of the 125 rows).

        Returns:
        numpy.ndarray:
            The modified data using solution A.
        """
        return np.mean(self.data, axis=1)       # axis 1

    def _get_modified_data_solution_b_before_projection(self):
        """
        Retrieves the modified data using solution B before projection.

        Returns:
        numpy.ndarray:
            The modified data using solution B before projection.
        """
        return np.reshape(self.data, (self.data.shape[0], -1))

    def get_modified_data_solution_b(self, n_components=0.95):
        """
        Retrieves the modified data using solution B (flattened rows + PCA).

        Parameters:
        n_components : int or float, optional
            If int, the number of components to keep. If float, it represents the target explained variance ratio.
            Default is 0.95.

        Returns:
        numpy.ndarray:
            The modified data using solution B.
        """
        pca = PCA(n_components=n_components)
        data_before_projections = self._get_modified_data_solution_b_before_projection()
        pca.fit(data_before_projections)
        return pca.transform(data_before_projections)
