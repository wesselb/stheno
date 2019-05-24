# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from time import time

import numpy as np
from lab import B
from plum import dispatch
from numpy.testing import assert_array_almost_equal

from stheno.matrix import Dense, dense as _dense

__all__ = ['benchmark', 'dense', 'allclose', 'approx']


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
def dense(a):
    return a


@dispatch(tuple)
def dense(a):
    return tuple(dense(x) for x in a)


@dispatch(Dense)
def dense(a):
    return _dense(a)


@dispatch(list)
def dense(a):
    return np.array(a)


@dispatch({B.Numeric, Dense, list}, {B.Numeric, Dense, list}, [object])
def allclose(a, b, desc=None, atol=1e-8, rtol=1e-8):
    np.testing.assert_allclose(dense(a), dense(b), atol=atol, rtol=rtol)


@dispatch(tuple, tuple, [object])
def allclose(a, b, desc=None, atol=1e-8, rtol=1e-8):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        allclose(x, y, desc, atol=atol, rtol=rtol)


approx = assert_array_almost_equal
