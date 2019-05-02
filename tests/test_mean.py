# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
from lab import B
from numpy.testing import assert_allclose
from plum import Dispatcher
import tensorflow as tf

from stheno.cache import Cache
from stheno.input import Observed
from stheno.kernel import EQ
from stheno.matrix import matrix
from stheno.mean import TensorProductMean, ZeroMean, Mean, OneMean, \
    PosteriorMean
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, ok


def test_corner_cases():
    yield raises, NotImplementedError, lambda: Mean()(1.)


def test_construction():
    m = TensorProductMean(lambda x: x ** 2)

    x = np.random.randn(10, 1)
    c = Cache()

    yield m, x
    yield m, x, c

    yield m, Observed(x)
    yield m, Observed(x), c


def test_basic_arithmetic():
    dispatch = Dispatcher()

    @dispatch(Number)
    def f1(x): return np.array([[x ** 2]])

    @dispatch(object)
    def f1(x): return np.sum(x ** 2, axis=1)[:, None]

    @dispatch(Number)
    def f2(x): return np.array([[x ** 3]])

    @dispatch(object)
    def f2(x): return np.sum(x ** 3, axis=1)[:, None]

    m1 = TensorProductMean(f1)
    m2 = TensorProductMean(f2)
    m3 = ZeroMean()
    x1 = np.random.randn(10, 2)
    x2 = np.random.randn()

    yield ok, np.allclose((m1 * m2)(x1), m1(x1) * m2(x1)), 'prod'
    yield ok, np.allclose((m1 * m2)(x2), m1(x2) * m2(x2)), 'prod 2'
    yield ok, np.allclose((m1 + m3)(x1), m1(x1) + m3(x1)), 'sum'
    yield ok, np.allclose((m1 + m3)(x2), m1(x2) + m3(x2)), 'sum 2'
    yield ok, np.allclose((5. * m1)(x1), 5. * m1(x1)), 'prod 3'
    yield ok, np.allclose((5. * m1)(x2), 5. * m1(x2)), 'prod 4'
    yield ok, np.allclose((5. + m1)(x1), 5. + m1(x1)), 'sum 3'
    yield ok, np.allclose((5. + m1)(x2), 5. + m1(x2)), 'sum 4'


def test_posterior_mean():
    z = np.linspace(0, 1, 10)
    pcm = PosteriorMean(
        TensorProductMean(lambda x: x),
        TensorProductMean(lambda x: x ** 2),
        EQ(), z, matrix(2 * EQ()(z)), np.random.randn(10)
    )

    # Check name.
    yield eq, str(pcm), 'PosteriorMean()'

    # Check that the mean computes.
    yield lambda: pcm(z)


def test_function_mean():
    m1 = 5 * OneMean() + (lambda x: x ** 2)
    m2 = (lambda x: x ** 2) + 5 * OneMean()
    m3 = (lambda x: x ** 2) + ZeroMean()
    m4 = ZeroMean() + (lambda x: x ** 2)
    x = np.random.randn(10, 1)

    yield ok, np.allclose(m1(x), 5 + x ** 2)
    yield ok, np.allclose(m2(x), 5 + x ** 2)
    yield ok, np.allclose(m3(x), x ** 2)
    yield ok, np.allclose(m4(x), x ** 2)

    def my_function(x): pass

    yield eq, str(TensorProductMean(my_function)), 'my_function'


def test_derivative():
    s = tf.Session()

    m = TensorProductMean(lambda x: x ** 2)
    m2 = TensorProductMean(lambda x: x ** 3)
    x = tf.constant(np.random.randn(10, 1))

    yield assert_allclose, s.run(m.diff(0)(x)), s.run(2 * x)
    yield assert_allclose, s.run(m2.diff(0)(x)), s.run(3 * x ** 2)

    s.close()


def test_selected_mean():
    m = 5 * OneMean() + (lambda x: x ** 2)
    x = np.random.randn(10, 3)

    yield assert_allclose, m.select([1, 2])(x), m(x[:, [1, 2]])


def test_shifting():
    m = 5 * OneMean() + (lambda x: x ** 2)
    x = np.random.randn(10, 3)

    yield assert_allclose, m.shift(5)(x), m(x - 5)


def test_stretching():
    m = 5 * OneMean() + (lambda x: x ** 2)
    x = np.random.randn(10, 3)

    yield assert_allclose, m.stretch(5)(x), m(x / 5)


def test_input_transform():
    m = 5 * OneMean() + (lambda x: x ** 2)
    x = np.random.randn(10, 3)

    yield assert_allclose, m.transform(lambda x, c: x - 5)(x), m(x - 5)
