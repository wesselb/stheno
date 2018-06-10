# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from stheno import Cache, Component
from stheno.kernel import ZeroKernel, OneKernel, Linear
from stheno.mean import ZeroMean, OneMean
# noinspection PyUnresolvedReferences
from . import eq, neq, ok


def test_lab_cache():
    c = Cache()

    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    x2 = np.random.randn(10, 10)

    yield eq, id(c.pw_dists(x1, x2)), id(c.pw_dists(x1, x2))
    yield neq, id(c.pw_dists(x1, x1)), id(c.pw_dists(x1, x2))
    yield eq, id(c.matmul(x1, x2, tr_a=True)), id(c.matmul(x1, x2, tr_a=True))
    yield neq, id(c.matmul(x1, x2, tr_a=True)), id(c.matmul(x1, x2))


def test_reversed_hit():
    k = Linear()
    c = Cache()
    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)

    k(x1, x2, c)
    yield ok, np.allclose(k(x1, x2, c), np.transpose(k(x2, x1, c)))


def test_ones_zeros():
    c = Cache()

    # Test that ones and zeros are cached and that all signatures work.
    k = ZeroKernel()
    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    yield eq, id(k(x1, c)), id(k(x2, c))
    yield eq, id(k(x1, c)), id(k(Component('test')(x2), c))
    x1, x2 = np.random.randn(10, 10), np.random.randn(5, 10)
    yield neq, id(k(x1, c)), id(k(x2, c))
    yield eq, id(k(1, c)), id(k(1, c))

    k = OneKernel()
    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    yield eq, id(k(x1, c)), id(k(x2, c))
    x1, x2 = np.random.randn(10, 10), np.random.randn(5, 10)
    yield neq, id(k(x1, c)), id(k(x2, c))
    yield eq, id(k(1, c)), id(k(1, c))

    # Test that ones and zeros are cached and that all signatures work.
    m = ZeroMean()
    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    yield eq, id(m(x1, c)), id(m(x2, c))
    yield eq, id(m(x1, c)), id(m(Component('test')(x2), c))
    x1, x2 = np.random.randn(10, 10), np.random.randn(5, 10)
    yield neq, id(m(x1, c)), id(m(x2, c))
    yield eq, id(m(1, c)), id(m(1, c))

    m = OneKernel()
    x1, x2 = np.random.randn(10, 10), np.random.randn(10, 10)
    yield eq, id(m(x1, c)), id(m(x2, c))
    x1, x2 = np.random.randn(10, 10), np.random.randn(5, 10)
    yield neq, id(m(x1, c)), id(m(x2, c))
    yield eq, id(m(1, c)), id(m(1, c))
