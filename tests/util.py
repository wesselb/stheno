from time import time

import numpy as np
from lab import B
from matrix import AbstractMatrix
from plum import dispatch, Union

__all__ = ["benchmark", "to_np", "approx"]


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


@dispatch
def to_np(a: B.Numeric):
    return a


@dispatch
def to_np(a: tuple):
    return tuple(to_np(x) for x in a)


@dispatch
def to_np(a: AbstractMatrix):
    return B.dense(a)


@dispatch
def to_np(a: list):
    return np.array(a)


@dispatch
def to_np(a: Union[B.TFNumeric, B.TorchNumeric]):
    return a.numpy()


@dispatch
def approx(
    a: Union[B.Numeric, AbstractMatrix, list],
    b: Union[B.Numeric, AbstractMatrix, list],
    desc=None,
    atol=1e-8,
    rtol=1e-8,
):
    np.testing.assert_allclose(to_np(a), to_np(b), atol=atol, rtol=rtol, err_msg=desc)


@dispatch
def approx(a: tuple, b: tuple, desc=None, atol=1e-8, rtol=1e-8):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        approx(x, y, desc=desc, atol=atol, rtol=rtol)
