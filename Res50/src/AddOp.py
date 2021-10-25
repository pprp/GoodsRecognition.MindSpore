import mindspore
import numpy as np

class Add():
    def add(self, x, y):
        if x is not None and y is not None:
            x = x.asnumpy()
            y = y.asnumpy()
            out = x + y
            out = np.array(out, x.dtype)
            return mindspore.Tensor(out)
        return None

