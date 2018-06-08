# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import sys
from time import time
from numpy.testing import assert_allclose

from nose.tools import assert_raises, assert_equal, assert_less, \
    assert_less_equal, assert_not_equal, assert_greater, \
    assert_greater_equal, ok_

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
    ok_(f(*args))


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
