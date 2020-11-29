from time import time

import numpy as np
from lab import B
from matrix import AbstractMatrix
from numpy.testing import assert_array_almost_equal
from plum import dispatch

__all__ = ['benchmark', 'to_np', 'approx']


def benchmark(f, args, n=1000, get_output=False):
    """Benchmark the performance of a function `f` called with arguments
    `args` in microseconds.

    Args:
        f (function): Function to benchmark.
        args (tuple): Argument to call `f` with.
        n (int): Repetitions.
        get_output (bool): Also return final output of function.
    """
    start = time()
    for i in range(n):
        out = f(*args)
    dur = (time() - start) * 1e6 / n
    return (dur, out) if get_output else out


@dispatch(B.Numeric)
def to_np(a):
    return a


@dispatch(tuple)
def to_np(a):
    return tuple(to_np(x) for x in a)


@dispatch(AbstractMatrix)
def to_np(a):
    return B.dense(a)


@dispatch(list)
def to_np(a):
    return np.array(a)


@dispatch({B.TFNumeric, B.TorchNumeric})
def to_np(a):
    return a.numpy()


@dispatch({B.Numeric, AbstractMatrix, list},
          {B.Numeric, AbstractMatrix, list}, [object])
def approx(a, b, desc=None, atol=1e-8, rtol=1e-8):
    np.testing.assert_allclose(to_np(a), to_np(b),
                               atol=atol, rtol=rtol, err_msg=desc)


@dispatch(tuple, tuple, [object])
def approx(a, b, desc=None, atol=1e-8, rtol=1e-8):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        approx(x, y, desc, atol=atol, rtol=rtol)

