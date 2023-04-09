
from ..dataset import Sample, Subset, ClassificationDataset
import os
import numpy as np
import pickle
from typing import List
from pprint import pprint

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in BGR channel order.
        '''

        if not os.path.isdir(fdir):
            raise ValueError(f"{fdir} is not a directory")

        # Define file names for the subsets
        if subset == Subset.TRAINING:
            file_names = [os.path.join(fdir, f"data_batch_{i}") for i in range(1, 4)]
        elif subset == Subset.VALIDATION:
            file_names = [os.path.join(fdir, "data_batch_5")]
        elif subset == Subset.TEST:
            file_names = [os.path.join(fdir, "test_batch")]

        # Load data from files
        data = []
        for file_name in file_names:
            with open(file_name, "rb") as f:
                batch = pickle.load(f, encoding="bytes")

            for i in range(len(batch[b"data"])):
                sample_idx = len(data)  # Index of the sample in the dataset
                sample_data = batch[b"data"][i]
                sample_label = batch[b"labels"][i]
                sample = Sample(idx=sample_idx, data=sample_data, label=sample_label)
                data.append(sample)

        for i in range(10):
            print(data[i])   

        

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        return len(self.data)


    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")

        x = self.data[idx]
        y = self.labels[idx]

        return Sample(x, y)

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        return 2
