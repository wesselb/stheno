from numbers import Number

import lab.tensorflow as B
import numpy as np
import pytest
import tensorflow as tf
from plum import Dispatcher

from stheno.input import Input, MultiInput
from stheno.kernel import EQ
from stheno.mean import TensorProductMean, ZeroMean, Mean, OneMean, PosteriorMean
from stheno.measure import FDD
from .util import approx

_dispatch = Dispatcher()


@_dispatch
def f1(x: Number):
    return np.array([[x ** 2]])


@_dispatch
def f1(x):
    return B.sum(x ** 2, axis=1)[:, None]


@_dispatch
def f2(x: Number):
    return np.array([[x ** 3]])


@_dispatch
def f2(x):
    return B.sum(x ** 3, axis=1)[:, None]


def test_corner_cases():
    with pytest.raises(RuntimeError):
        Mean()(1.0)


@pytest.mark.parametrize("x", [B.randn(10), Input(B.randn(10)), FDD(None, B.randn(10))])
def test_construction(x):
    m = TensorProductMean(lambda y: y ** 2)
    m(x)

    # Test `MultiInput` construction.
    approx(m(MultiInput(x, x)), B.concat(m(x), m(x), axis=0))


def test_basic_arithmetic():
    m1 = TensorProductMean(f1)
    m2 = TensorProductMean(f2)
    m3 = ZeroMean()

    x1 = B.randn(10, 2)
    x2 = B.randn()

    approx((m1 * m2)(x1), m1(x1) * m2(x1))
    approx((m1 * m2)(x2), m1(x2) * m2(x2))
    approx((m1 + m3)(x1), m1(x1) + m3(x1))
    approx((m1 + m3)(x2), m1(x2) + m3(x2))
    approx((5.0 * m1)(x1), 5.0 * m1(x1))
    approx((5.0 * m1)(x2), 5.0 * m1(x2))
    approx((5.0 + m1)(x1), 5.0 + m1(x1))
    approx((5.0 + m1)(x2), 5.0 + m1(x2))


def test_sum():
    m1 = TensorProductMean(f1)
    m2 = TensorProductMean(f2)

    # Test equality.
    assert m1 + m2 == m1 + m2
    assert m1 + m2 == m2 + m1
    assert m1 + m2 != ZeroMean() + m2
    assert m1 + m2 != m1 + ZeroMean()


def test_product():
    m1 = TensorProductMean(f1)
    m2 = TensorProductMean(f2)

    # Test equality.
    assert m1 * m2 == m1 * m2
    assert m1 * m2 == m2 * m1
    assert m1 * m2 != m2 * m2
    assert m1 * m2 != m1 * m1


def test_posterior_mean():
    z = B.linspace(0, 1, 10)
    pcm = PosteriorMean(
        TensorProductMean(lambda x: x),
        TensorProductMean(lambda x: x ** 2),
        EQ(),
        z,
        2 * EQ()(z),
        B.randn(10),
    )

    # Check name.
    assert str(pcm) == "PosteriorMean()"

    # Check that the mean computes.
    pcm(z)


def test_tensor_product():
    m1 = 5 * OneMean() + (lambda x: x ** 2)
    m2 = (lambda x: x ** 2) + 5 * OneMean()
    m3 = (lambda x: x ** 2) + ZeroMean()
    m4 = ZeroMean() + (lambda x: x ** 2)

    x = B.randn(10, 1)
    assert np.allclose(m1(x), 5 + x ** 2)
    assert np.allclose(m2(x), 5 + x ** 2)
    assert np.allclose(m3(x), x ** 2)
    assert np.allclose(m4(x), x ** 2)

    def my_function(x):
        pass

    assert str(TensorProductMean(my_function)) == "my_function"


def test_derivative():
    m = TensorProductMean(lambda x: x ** 2)
    m2 = TensorProductMean(lambda x: x ** 3)

    x = B.randn(tf.float64, 10, 1)
    approx(m.diff(0)(x), 2 * x)
    approx(m2.diff(0)(x), 3 * x ** 2)


def test_selected_mean():
    m = 5 * OneMean() + (lambda x: x ** 2)
    x = B.randn(10, 3)
    approx(m.select([1, 2])(x), m(x[:, [1, 2]]))


def test_shifting():
    m = 5 * OneMean() + (lambda x: x ** 2)
    x = B.randn(10, 3)
    approx(m.shift(5)(x), m(x - 5))


def test_stretching():
    m = 5 * OneMean() + (lambda x: x ** 2)
    x = B.randn(10, 3)
    approx(m.stretch(5)(x), m(x / 5))


def test_input_transform():
    m = 5 * OneMean() + (lambda x: x ** 2)
    x = B.randn(10, 3)
    approx(m.transform(lambda x: x - 5)(x), m(x - 5))