import functools
import numpy as np


def _backward(trunc_array, padded_array, axis, real, scale):
    padded_array.fill(0)
    if real:
        s = [slice(0, n) for n in trunc_array.shape]
        padded_array[s] = trunc_array[:]
    else:
        N = trunc_array.shape[axis]
        su = [slice(None)]*trunc_array.ndim
        su[axis] = slice(0, N//2)
        padded_array[su] = trunc_array[su]
        su[axis] = slice(-N//2, None)
        padded_array[su] = trunc_array[su]
    padded_array *= scale


def _forward(padded_array, trunc_array, axis, real, scale):
    trunc_array.fill(0)
    N = trunc_array.shape[axis]
    if not real:
        su = [slice(None)]*trunc_array.ndim
        su[axis] = slice(0, N//2+1)
        trunc_array[su] = padded_array[su]
        su[axis] = slice(-N//2, None)
        trunc_array[su] += padded_array[su]
    else:
        s = [slice(None)]*trunc_array.ndim
        s[axis] = slice(0, N)
        trunc_array[:] = padded_array[s]
    trunc_array *= (1./scale)


class Padder(object):

    def __init__(self, padded_array, trunc_shape=(0,), axis=None,
                 real=False, scale=1.0):
        trunc_array = np.zeros(trunc_shape, dtype=padded_array.dtype)
        self.forward = functools.partial(_forward, padded_array,
                                         trunc_array, axis, real, scale)
        self.backward = functools.partial(_backward, trunc_array,
                                          padded_array, axis, real, scale)
        self.forward.input_array = self.forward.args[0]
        self.forward.output_array = self.forward.args[1]
        self.backward.input_array = self.backward.args[0]
        self.backward.output_array = self.backward.args[1]
