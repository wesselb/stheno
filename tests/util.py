# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys
from time import time

import numpy as np
from lab import B
from nose.tools import assert_raises, assert_equal, assert_less, \
    assert_less_equal, assert_not_equal, assert_greater, \
    assert_greater_equal, ok_
from plum import dispatch

from stheno.matrix import Dense, dense as _dense

le = assert_less_equal
lt = assert_less
eq = assert_equal
neq = assert_not_equal
ge = assert_greater_equal
gt = assert_greater
raises = assert_raises
ok = ok_


def call(f, method, args=(), res=True):
    assert_equal(getattr(f, method)(*args), res)


def lam(f, args=()):
    ok_(f(*args), 'Lambda returned False.')


def eprint(*args, **kw_args):
    print(*args, file=sys.stderr, **kw_args)


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
    return B.array(a)


def allclose(a, b, desc=None, atol=1e-8, rtol=1e-8):
    return np.allclose(dense(a), dense(b), atol=atol, rtol=rtol)


@dispatch({B.Numeric, Dense, list}, {B.Numeric, Dense, list}, [object])
def assert_allclose(a, b, desc=None, atol=1e-8, rtol=1e-8):
    np.testing.assert_allclose(dense(a), dense(b), atol=atol, rtol=rtol)


@dispatch(tuple, tuple, [object])
def assert_allclose(a, b, desc=None, atol=1e-8, rtol=1e-8):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert_allclose(x, y, desc, atol=atol, rtol=rtol)


def assert_instance(a, b, desc=None):
    assert isinstance(a, b)
