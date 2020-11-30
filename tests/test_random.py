import lab as B
import numpy as np
import pytest
from matrix import Dense, Zero
from scipy.stats import multivariate_normal
from plum import NotFoundLookupError

from stheno.random import Normal, RandomVector
from .util import approx


@pytest.fixture()
def normal1():
    mean = B.randn(3, 1)
    chol = B.randn(3, 3)
    var = chol @ chol.T
    return Normal(mean, var)


@pytest.fixture()
def normal2():
    mean = B.randn(3, 1)
    chol = B.randn(3, 3)
    var = chol @ chol.T
    return Normal(mean, var)


def test_normal_mean_is_zero():
    # Check zero case.
    dist = Normal(B.eye(3))
    assert dist.mean_is_zero
    approx(dist.mean, B.zeros(3, 1))

    # Check another zero case.
    dist = Normal(Zero(np.float32, 3, 1), B.eye(3))
    assert dist.mean_is_zero
    approx(dist.mean, B.zeros(3, 1))

    # Check nonzero case.
    assert not Normal(B.randn(3, 1), B.eye(3)).mean_is_zero


def test_normal_lazy_zero_mean():
    dist = Normal(lambda: B.eye(3))

    assert dist.mean_is_zero
    assert dist._mean is 0
    assert dist._var is None

    approx(dist.mean, B.zeros(3, 1))
    # At this point, the variance should be constructed, because it is used to get the
    # dimensionality and data type for the mean.
    assert dist._var is not None

    approx(dist.var, B.eye(3))


def test_normal_lazy_nonzero_mean():
    dist = Normal(lambda: B.ones(3, 1), lambda: B.eye(3))

    assert not dist.mean_is_zero
    approx(dist._mean, B.ones(3, 1))
    assert dist._var is None

    approx(dist.mean, B.ones(3, 1))
    assert dist._var is None

    approx(dist.var, B.eye(3))


def test_normal_m2(normal1):
    approx(normal1.m2, normal1.var + normal1.mean @ normal1.mean.T)


def test_normal_marginals(normal1):
    mean, lower, upper = normal1.marginals()
    approx(mean, normal1.mean.squeeze())
    approx(lower, normal1.mean.squeeze() - 1.96 * B.diag(normal1.var) ** 0.5)
    approx(upper, normal1.mean.squeeze() + 1.96 * B.diag(normal1.var) ** 0.5)


def test_normal_logpdf(normal1):
    normal1_sp = multivariate_normal(normal1.mean[:, 0], B.dense(normal1.var))
    x = B.randn(3, 10)
    approx(normal1.logpdf(x), normal1_sp.logpdf(x.T))

    # Test the the output of `logpdf` is flattened appropriately.
    assert B.shape(normal1.logpdf(B.ones(3, 1))) == ()
    assert B.shape(normal1.logpdf(B.ones(3, 2))) == (2,)


def test_normal_entropy(normal1):
    normal1_sp = multivariate_normal(normal1.mean[:, 0], B.dense(normal1.var))
    approx(normal1.entropy(), normal1_sp.entropy())


def test_normal_kl(normal1, normal2):
    assert normal1.kl(normal1) < 1e-6
    assert normal1.kl(normal2) > 0.1

    # Test against Monte Carlo estimate.
    samples = normal1.sample(50_000)
    kl_est = B.mean(normal1.logpdf(samples)) - B.mean(normal2.logpdf(samples))
    kl = normal1.kl(normal2)
    approx(kl_est, kl, rtol=0.05)


def test_normal_w2(normal1, normal2):
    assert normal1.w2(normal1) < 1e-6
    assert normal1.w2(normal2) > 0.1


def test_normal_sampling():
    for mean in [0, 1]:
        dist = Normal(mean, 3 * B.eye(np.int32, 200))

        # Sample without noise.
        samples = dist.sample(2000)
        approx(B.mean(samples), mean, atol=5e-2)
        approx(B.std(samples) ** 2, 3, atol=5e-2)

        # Sample with noise
        samples = dist.sample(2000, noise=2)
        approx(B.mean(samples), mean, atol=5e-2)
        approx(B.std(samples) ** 2, 5, atol=5e-2)


def test_normal_display(normal1):
    assert str(normal1) == RandomVector.__str__(normal1)
    assert repr(normal1) == RandomVector.__repr__(normal1)


def test_normal_arithmetic(normal1, normal2):
    a = Dense(B.randn(3, 3))
    b = 5.0

    # Test matrix multiplication.
    approx(normal1.lmatmul(a).mean, a @ normal1.mean)
    approx(normal1.lmatmul(a).var, a @ normal1.var @ a.T)
    approx(normal1.rmatmul(a).mean, a.T @ normal1.mean)
    approx(normal1.rmatmul(a).var, a.T @ normal1.var @ a)

    # Test multiplication.
    approx((normal1 * b).mean, normal1.mean * b)
    approx((normal1 * b).var, normal1.var * b ** 2)
    approx((b * normal1).mean, normal1.mean * b)
    approx((b * normal1).var, normal1.var * b ** 2)
    with pytest.raises(NotFoundLookupError):
        normal1.__mul__(normal1)
    with pytest.raises(NotFoundLookupError):
        normal1.__rmul__(normal1)

    # Test addition.
    approx((normal1 + normal2).mean, normal1.mean + normal2.mean)
    approx((normal1 + normal2).var, normal1.var + normal2.var)
    approx(normal1.__radd__(b).mean, normal1.mean + b)
    approx(normal1.__radd__(b).mean, normal1.mean + b)
    with pytest.raises(NotFoundLookupError):
        normal1.__add__(RandomVector())
    with pytest.raises(NotFoundLookupError):
        normal1.__radd__(RandomVector())

    # Test negation.
    approx((-normal1).mean, -normal1.mean)
    approx((-normal1).var, normal1.var)

    # Test substraction.
    approx((normal1 - normal2).mean, normal1.mean - normal2.mean)
    approx((normal1 - normal2).var, normal1.var + normal2.var)
    approx(normal1.__rsub__(normal2).mean, normal2.mean - normal1.mean)
    approx(normal1.__rsub__(normal2).var, normal1.var + normal2.var)

    # Test division.
    approx(normal1.__div__(b).mean, normal1.mean / b)
    approx(normal1.__div__(b).var, normal1.var / b ** 2)
    approx(normal1.__truediv__(b).mean, normal1.mean / b)
    approx(normal1.__truediv__(b).var, normal1.var / b ** 2)
