#based on https://github.com/CompVis/taming-transformers

import os
from torch.utils.data import Dataset
from scripts.data.base import ImagePaths
import scripts.constants as CONSTANTS


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
    def __init__(self, size, training_images_list_file, horizontalflip=False, random_contrast=False, shiftrotate=False, add_labels=False, unique_skipped_labels=[]):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = sorted(f.read().splitlines(), key=lambda x: os.path.basename(x))
        self.paths = paths

        labels=None
        if add_labels:
            labels_per_file = list(map(lambda path: path.split('/')[-2], paths))
            labels_set = sorted(list(set(labels_per_file)))
            self.labels_to_idx = {label_name: i for i, label_name in enumerate(labels_set)}
            labels = {
                CONSTANTS.DISENTANGLER_CLASS_OUTPUT: [self.labels_to_idx[label_name] for label_name in labels_per_file],
                CONSTANTS.DATASET_CLASSNAME: labels_per_file
            }
            
        self.indx_to_label = {v: k for k, v in self.labels_to_idx.items()}

        self.data = ImagePaths(paths=paths, size=size, random_crop=False, horizontalflip=horizontalflip, random_contrast=random_contrast, shiftrotate=shiftrotate, labels=labels, unique_skipped_labels=unique_skipped_labels)


class CustomTest(CustomTrain):
    def __init__(self, size, test_images_list_file, add_labels=False, unique_skipped_labels=[]):
        super().__init__(size, test_images_list_file, add_labels=add_labels, unique_skipped_labels=unique_skipped_labels)


