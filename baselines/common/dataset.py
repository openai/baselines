import numpy as np

class Dataset(object):
    def __init__(self, data_map, deterministic=False, shuffle=True):
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle: self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, deterministic=True):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, deterministic)


def iterbatches(arrays, *, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle: np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)
