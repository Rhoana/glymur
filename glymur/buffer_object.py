# Wrapper for an array-like structure to be used as a buffer
import numpy as np

class ArrayAsBuffer():

    def __init__(self, arr):
        self._arr = np.ascontiguousarray(arr)
        self._pos = 0
        self._open = True

    def seek(self, idx, from_what=0):
        assert(self._open)
        if from_what == 0: # relative to the beginning of the file
            self._pos = idx
        elif from_what == 1: # relative to the current position
            self._pos += idx
        elif from_what == 2: # relative to the end of the file
            self._pos = len(self._arr) - idx
        else:
            raise ValueError()

    def read(self, len=None):
        assert(self._open)
        if len is None:
            len = len(self._arr) - self._pos

        assert(len > 0)
        res = self._arr[self._pos:self._pos + len].data[:]
        self._pos += len
        return res

    def write(self, buffer):
        assert(self._open)
        self._arr[self._pos:self._pos + len(buffer)] = buffer
        self._pos += len(buffer)

    def tell(self):
        assert(self._open)
        return self._pos

    def close(self):
        self._open = False

    @property
    def closed(self):
        return self._open == False

    def length(self):
        return len(self._arr)

