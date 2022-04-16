from time import time

import numpy as np
from lab import B
from matrix import AbstractMatrix
from plum import dispatch, Union

from stheno import Normal

__all__ = ["benchmark", "approx"]


def benchmark(f, args, n=1000, get_output=False):
    """Benchmark the performance of a function `f` called with arguments
    `args` in microseconds.

    Args:
        f (function): Function to benchmark.
        args (tuple): Argument to call `f` with.
        n (int): Repetitions.
        get_output (bool): Also return final output of function.

    Returns:
        float: Time in seconds.
        object, optional: Output of function.
    """
    start = time()
    for i in range(n):
        out = f(*args)
    dur = (time() - start) * 1e6 / n
    return (dur, out) if get_output else out


@dispatch
def approx(
    a: Union[B.Numeric, AbstractMatrix, list],
    b: Union[B.Numeric, AbstractMatrix, list],
    desc=None,
    atol=1e-8,
    rtol=1e-8,
):
    np.testing.assert_allclose(
        B.to_numpy(a), B.to_numpy(b), atol=atol, rtol=rtol, err_msg=desc
    )


@dispatch
def approx(a: Normal, b: Normal, desc=None, atol=1e-8, rtol=1e-8):
    approx(a.mean, b.mean, desc=desc, atol=atol, rtol=rtol)
    approx(a.var, b.var, desc=desc, atol=atol, rtol=rtol)


@dispatch
def approx(a: tuple, b: tuple, desc=None, atol=1e-8, rtol=1e-8):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        approx(x, y, desc=desc, atol=atol, rtol=rtol)
