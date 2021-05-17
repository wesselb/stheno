import numpy as np
import pytest
from lab import B
from mlkernels import (
    Linear,
    EQ,
    ZeroKernel,
    OneKernel,
    ScaledKernel,
    TensorProductMean,
    ZeroMean,
    ScaledMean,
    OneMean,
)
from plum import NotFoundLookupError

from stheno.model import Measure, GP
from stheno.random import Normal
from .util import assert_equal_gps
from ..util import approx


def test_corner_cases():
    p1 = GP(EQ())
    p2 = GP(EQ())
    x = B.randn(10, 2)

    # Test check for measure group.
    with pytest.raises(AssertionError):
        p1 + p2
    with pytest.raises(AssertionError):
        p1 * p2

    # Test incompatible operations.
    with pytest.raises(NotFoundLookupError):
        p1 + p1(x)
    with pytest.raises(NotFoundLookupError):
        p1 * p1(x)

    # Check test for prior.
    with pytest.raises(RuntimeError):
        GP().measure


def test_display():
    # Check display of GPs.
    assert str(GP()) == "GP()"
    assert str(GP(EQ())) == "GP(0, EQ())"


def test_formatting():
    p = 2 * GP(1, EQ(), measure=Measure())
    assert str(p.display(lambda x: x ** 2)) == "GP(4 * 1, 16 * EQ())"


def test_construction():
    p = GP(EQ())

    x = B.randn(10, 1)

    p.mean(x)

    p.kernel(x)
    p.kernel(x, x)

    p.kernel.elwise(x)
    p.kernel.elwise(x, x)

    # Test resolution of kernel and mean.
    k = EQ()
    m = TensorProductMean(lambda x: x ** 2)

    assert isinstance(GP(k).mean, ZeroMean)
    assert isinstance(GP(5, k).mean, ScaledMean)
    assert isinstance(GP(1, k).mean, OneMean)
    assert isinstance(GP(0, k).mean, ZeroMean)
    assert isinstance(GP(m, k).mean, TensorProductMean)
    assert isinstance(GP(k).kernel, EQ)
    assert isinstance(GP(5).kernel, ScaledKernel)
    assert isinstance(GP(1).kernel, OneKernel)
    assert isinstance(GP(0).kernel, ZeroKernel)

    # Test construction of finite-dimensional distribution without noise.
    d = GP(m, k)(x)
    approx(d.var, k(x))
    approx(d.mean, m(x))

    # Test construction of finite-dimensional distribution with noise.
    d = GP(m, k)(x, 1)
    approx(d.var, k(x) + B.eye(k(x)))
    approx(d.mean, m(x))


def test_sum_other():
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())

    def five(y):
        return 5 * B.ones(B.shape(y)[0], 1)

    x = B.randn(5, 1)
    for p_sum in [
        # Add a numeric thing.
        p + 5.0,
        5.0 + p,
        p.measure.sum(GP(), p, 5.0),
        p.measure.sum(GP(), 5.0, p),
        # Add a function.
        p + five,
        five + p,
        p.measure.sum(GP(), p, five),
        p.measure.sum(GP(), five, p),
    ]:
        approx(p.mean(x) + 5.0, p_sum.mean(x))
        approx(p.mean(x) + 5.0, p_sum.mean(x))
        approx(p.kernel(x), p_sum.kernel(x))
        approx(p.kernel(x), p_sum.kernel(x))

    # Check that a `GP` cannot be summed with a `Normal`.
    with pytest.raises(NotFoundLookupError):
        p + Normal(np.eye(3))
    with pytest.raises(NotFoundLookupError):
        Normal(np.eye(3)) + p


def test_mul_other():
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())

    def five(y):
        return 5 * B.ones(B.shape(y)[0], 1)

    x = B.randn(5, 1)
    for p_mul in [
        # Multiply numeric thing.
        p * 5.0,
        5.0 * p,
        p.measure.mul(GP(), p, 5.0),
        p.measure.mul(GP(), 5.0, p),
        # Multiply with a function.
        p * five,
        five * p,
        p.measure.mul(GP(), p, five),
        p.measure.mul(GP(), five, p),
    ]:
        approx(5.0 * p.mean(x), p_mul.mean(x))
        approx(5.0 * p.mean(x), p_mul.mean(x))
        approx(25.0 * p.kernel(x), p_mul.kernel(x))
        approx(25.0 * p.kernel(x), p_mul.kernel(x))

    # Check that a `GP` cannot be multiplied with a `Normal`.
    with pytest.raises(NotFoundLookupError):
        p * Normal(np.eye(3))
    with pytest.raises(NotFoundLookupError):
        Normal(np.eye(3)) * p


def test_stationarity():
    m = Measure()

    p1 = GP(EQ(), measure=m)
    p2 = GP(EQ().stretch(2), measure=m)
    p3 = GP(EQ().periodic(10), measure=m)

    p = p1 + 2 * p2

    assert p.stationary

    p = p3 + p

    assert p.stationary

    p = p + GP(Linear(), measure=m)

    assert not p.stationary


def test_marginals():
    p = GP(TensorProductMean(lambda x: x ** 2), EQ())
    x = B.linspace(0, 5, 10)

    # Check that `marginals` outputs the right thing.
    mean, var = p(x).marginals()
    var = B.diag(p.kernel(x))
    approx(mean, p.mean(x)[:, 0])
    approx(var, B.diag(p.kernel(x)))

    # Test correctness.
    y = p(x).sample()
    post = p.measure | (p(x), y)

    # Concentration on data:
    mean, var = post(p)(x).marginals()
    approx(mean, y[:, 0])
    approx(var, B.zeros(10), atol=1e-5)

    # Reversion to prior:
    mean, var = post(p)(x + 100).marginals()
    approx(mean, p.mean(x + 100)[:, 0])
    approx(var, B.diag(p.kernel(x + 100)))
