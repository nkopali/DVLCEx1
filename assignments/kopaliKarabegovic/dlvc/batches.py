import typing
import numpy as np
from .dataset import Dataset
from .ops import Op

class Batch:
    '''
    A (mini)batch generated by the batch generator.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.data = None
        self.label = None
        self.idx = None

class BatchGenerator:
    '''
    Batch generator.
    Returned batches have the following properties:
      data: numpy array holding batch data of shape (s, SHAPE_OF_DATASET_SAMPLES).
      label: numpy array holding batch labels of shape (s, SHAPE_OF_DATASET_LABELS).
      idx: numpy array with shape (s,) encoding the indices of each sample in the original dataset.
    '''

    def __init__(self, dataset: Dataset, num: int, shuffle: bool, op: Op=None):
        '''
        Ctor.
        Dataset is the dataset to iterate over.
        num is the number of samples per batch. the number in the last batch might be smaller than that.
        shuffle controls whether the sample order should be preserved or not.
        op is an operation to apply to input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values, such as if num is > len(dataset).
        '''
        if not isinstance(dataset, Dataset):
            raise TypeError("The dataset should be an instance of Dataset class")
        if not isinstance(num, int) or num <= 0:
            raise TypeError("The number of samples per batch should be a positive integer")
        if num > len(dataset):
            raise TypeError("The number of samples per batch should not be bigger than the length of dataset")
        if not isinstance(shuffle, bool):
            raise TypeError("The shuffle argument should be boolean")

        if num > len(dataset):
            raise ValueError("The number of samples per batch should be less than or equal to the dataset size")

        self.dataset = dataset
        self.num = num
        self.shuffle = shuffle
        self.op = op

    def __len__(self) -> int:
        '''
        Returns the total number of batches the dataset is split into.
            This is identical to the total number of batches yielded every time the __iter__ method is called.
        '''
        
        return int(np.ceil(len(self.dataset) / self.num))


    def __iter__(self) -> typing.Iterable[Batch]:
        '''
        Iterate over the wrapped dataset, returning the data as batches.
        '''

        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.num):
            batch_indices = indices[i:i + self.num]
            batch = Batch()
            batch.idx = batch_indices
            batch.data = self.dataset.get_data(batch_indices)
            batch.label = self.dataset.get_label(batch_indices)
            if self.op is not None:
                batch.data = self.op(batch.data)
            yield batch
