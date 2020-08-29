import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self._neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) >> 1
        if end <= mid:
            return self._reduce_helper(start, end, node << 1, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, node << 1 | 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, node << 1, node_start, mid),
                    self._reduce_helper(mid + 1, end, node << 1 | 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx >>= 1
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[idx << 1],
                self._value[idx << 1 | 1]
            )
            idx >>= 1

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class ZKWSegmentTree(SegmentTree):
    def __init__(self, *args, **kwargs):
        super(ZKWSegmentTree, self).__init__(*args, **kwargs)

    def _reduce_helper(self, start, end, node, node_start, node_end):
        result: float = self._neutral_element

        left = start + self._capacity - 1
        right = end + self._capacity + 1

        while left ^ right ^ 1:
            if ~left & 1:  # l % 2 == 0 thus $left is left son of $left >> 1
                result = self._operation(result, self._value[left ^ 1])
            if right & 1:  # r % 2 == 1 ths $right is right son of $right >> 1
                result = self._operation(result, self._value[right ^ 1])
            left >>= 1
            right >>= 1

        return result


class SumSegmentTree(ZKWSegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[idx << 1] > prefixsum:
                idx <<= 1
            else:
                prefixsum -= self._value[idx << 1]
                idx = idx << 1 | 1
        return idx - self._capacity


class MinSegmentTree(ZKWSegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
