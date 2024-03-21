import os
import numpy as np


class DataLoader:
    def __init__(self, root_dir=os.getcwd() + '\\data', num_activities=19, num_subjects=8, num_samples=60, rows=125, columns=45):
        self.root_dir = root_dir
        self.num_activities = num_activities
        self.num_subjects = num_subjects
        self.num_samples = num_samples
        self.rows = rows
        self.columns = columns
        self.data = self.load_data()

    def load_data(self):
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
        return self.data[
            ((activity_idx - 1) * 8 * 60) +
            ((subject_idx - 1) * 60) +
            (sample_idx - 1)]

    def get_pure_data(self):
        return self.data

    def get_modified_data_solution_a(self):
        return np.mean(self.data, axis=1)       # axis 1

    def get_modified_data_solution_b(self):
        return np.reshape(self.data, (self.data.shape[0], -1))


# TODO: Remove the following code
# if __name__ == '__main__':
#     print("Hello world!")
#     dataLoader = DataLoader()
#     pure_dataset = dataLoader.get_pure_data()    # 9120 x 125 x 45
#     solution_a_dataset = dataLoader.get_modified_data_solution_a()
#     solution_b_dataset = dataLoader.get_modified_data_solution_b()
#     print(solution_a_dataset.shape)
#     print(solution_b_dataset.shape)
#
#     original_data = np.reshape(pure_dataset, (19, 8, 60, 125, 45))
#     temp_idx = 0
#     for i in range(19):
#         for j in range(8):
#             for k in range(60):
#                 if not np.all(dataLoader.get_sample(i + 1, j + 1, k + 1) == pure_dataset[temp_idx]):
#                     print("FALSE !!!!!")
#                     print(i, j, k)
#                     exit(1)
#                 temp_idx += 1
#     print("All good ..")

