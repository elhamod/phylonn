import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):   
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, add_labels=False):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()

        labels=None
        if add_labels:
            labels_per_file = list(map(lambda path: path.split('/')[-2], paths))
            labels_set = sorted(list(set(labels_per_file)))
            labels_to_idx = {label_name: i for i, label_name in enumerate(labels_set)}
            labels = {'class': [labels_to_idx[label_name] for label_name in labels_per_file]}

        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=labels)


class CustomTest(CustomTrain):
    def __init__(self, size, test_images_list_file, add_labels=False):
        super().__init__(size, test_images_list_file, add_labels)


