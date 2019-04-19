import numpy as np


class Dataset(object):
    def __init__(self, data_map, shuffle=True):
        """
        Data loader that handles batches and shuffling.
        WARNING: this will alter the given data_map ordering, as dicts are mutable

        :param data_map: (dict) the input data, where every column is a key
        :param shuffle: (bool) Whether to shuffle or not the dataset
            Important: this should be disabled for recurrent policies
        """
        self.data_map = data_map
        self.shuffle = shuffle
        self.n_samples = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        if self.shuffle:
            self.shuffle_dataset()

    def shuffle_dataset(self):
        """
        Shuffles the data_map
        """
        perm = np.arange(self.n_samples)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

    def next_batch(self, batch_size):
        """
        returns a batch of data of a given size

        :param batch_size: (int) the size of the batch
        :return: (dict) a batch of the input data of size 'batch_size'
        """
        if self._next_id >= self.n_samples:
            self._next_id = 0
            if self.shuffle:
                self.shuffle_dataset()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n_samples - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        """
        generator that iterates over the dataset

        :param batch_size: (int) the size of the batch
        :return: (dict) a batch of the input data of size 'batch_size'
        """
        if self.shuffle:
            self.shuffle_dataset()

        while self._next_id <= self.n_samples - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, shuffle=True):
        """
        Return a subset of the current dataset

        :param num_elements: (int) the number of element you wish to have in the subset
        :param shuffle: (bool) Whether to shuffle or not the dataset
        :return: (Dataset) a new subset of the current Dataset object
        """
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, shuffle)


def iterbatches(arrays, *, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    """
    Iterates over arrays in batches, must provide either num_batches or batch_size, the other must be None.

    :param arrays: (tuple) a tuple of arrays
    :param num_batches: (int) the number of batches, must be None is batch_size is defined
    :param batch_size: (int) the size of the batch, must be None is num_batches is defined
    :param shuffle: (bool) enable auto shuffle
    :param include_final_partial_batch: (bool) add the last batch if not the same size as the batch_size
    :return: (tuples) a tuple of a batch of the arrays
    """
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n_samples = arrays[0].shape[0]
    assert all(a.shape[0] == n_samples for a in arrays[1:])
    inds = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(inds)
    sections = np.arange(0, n_samples, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)
