# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.testing import assert_approx_equal, assert_array_almost_equal, \
    assert_allclose

from stheno.kernel import ZeroKernel, OneKernel, Linear, EQ
from stheno.mean import ZeroMean, OneMean
from stheno.input import Component
from stheno.cache import Cache, uprank
from stheno.random import GPPrimitive
# noinspection PyUnresolvedReferences
from . import eq, neq, ok, raises, benchmark, le, eprint


def test_lab_cache():
    c = Cache()

    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    x2 = np.random.randn(10, 10)

    yield eq, id(c.pw_dists(x1, x2)), id(c.pw_dists(x1, x2))
    yield neq, id(c.pw_dists(x1, x1)), id(c.pw_dists(x1, x2))
    yield eq, id(c.matmul(x1, x2, tr_a=True)), id(c.matmul(x1, x2, tr_a=True))
    yield neq, id(c.matmul(x1, x2, tr_a=True)), id(c.matmul(x1, x2))


def test_ones_zeros():
    c = Cache()

    # Test that ones and zeros are cached and that all signatures work.
    k = ZeroKernel()
    yield eq, id(k(np.random.randn(10, 10), c)), \
          id(k(np.random.randn(10, 10), c))
    yield neq, id(k(np.random.randn(10, 10), c)), \
          id(k(np.random.randn(5, 10), c))
    yield eq, id(k(1, c)), id(k(1, c))

    k = OneKernel()
    yield eq, id(k(np.random.randn(10, 10), c)), \
          id(k(np.random.randn(10, 10), c))
    yield neq, id(k(np.random.randn(10, 10), c)), \
          id(k(np.random.randn(5, 10), c))
    yield eq, id(k(1, c)), id(k(1, c))

    # Test that ones and zeros are cached and that all signatures work.
    m = ZeroMean()
    yield eq, id(m(np.random.randn(10, 10), c)), \
          id(m(np.random.randn(10, 10), c))
    yield neq, id(m(np.random.randn(10, 10), c)), \
          id(m(np.random.randn(5, 10), c))
    yield eq, id(m(1, c)), id(m(1, c))

    m = OneKernel()
    yield eq, id(m(np.random.randn(10, 10), c)), \
          id(m(np.random.randn(10, 10), c))
    yield neq, id(m(np.random.randn(10, 10), c)), \
          id(m(np.random.randn(5, 10), c))
    yield eq, id(m(1, c)), id(m(1, c))


def test_uprank():
    yield assert_allclose, uprank(0), [[0]]
    yield assert_allclose, uprank([0]), [[0]]
    yield assert_allclose, uprank([[0]]), [[0]]
    yield eq, type(uprank(Component('test')(0))), Component('test')

    k = OneKernel()

    yield eq, k(0, 0).shape, (1, 1)
    yield eq, k(0, np.ones(5)).shape, (1, 5)
    yield eq, k(0, np.ones((5, 2))).shape, (1, 5)

    yield eq, k(np.ones(5), 0).shape, (5, 1)
    yield eq, k(np.ones(5), np.ones(5)).shape, (5, 5)
    yield eq, k(np.ones(5), np.ones((5, 2))).shape, (5, 5)

    yield eq, k(np.ones((5, 2)), 0).shape, (5, 1)
    yield eq, k(np.ones((5, 2)), np.ones(5)).shape, (5, 5)
    yield eq, k(np.ones((5, 2)), np.ones((5, 2))).shape, (5, 5)

    yield raises, ValueError, lambda: k(0, np.ones((5, 2, 1)))
    yield raises, ValueError, lambda: k(np.ones((5, 2, 1)))

    m = OneMean()

    yield eq, m(0).shape, (1, 1)
    yield eq, m(np.ones(5)).shape, (5, 1)
    yield eq, m(np.ones((5, 2))).shape, (5, 1)

    p = GPPrimitive(EQ())
    x = np.linspace(0, 10, 10)

    yield assert_approx_equal, p.condition(1, 1)(1).mean, np.array([[1]])
    yield assert_array_almost_equal, p.condition(x, x)(x).mean, x[:, None]
    yield assert_array_almost_equal, p.condition(x, x[:, None])(x).mean, \
          x[:, None]


def test_cache_performance():
    c = Cache()
    k1 = EQ()
    k2 = EQ()
    x = np.linspace(0, 1, 2000)

    dur1, y1 = benchmark(k1, (x, c), n=1, get_output=True)
    dur2, y2 = benchmark(k1, (x, c), n=1, get_output=True)
    dur3, y3 = benchmark(k2, (x, c), n=1, get_output=True)

    yield assert_allclose, y1, y2
    yield assert_allclose, y1, y3

    # Test performance of call cache.
    yield le, dur2, dur1 / 500

    # Test performance of LAB cache.
    yield le, dur3, dur1 / 20
