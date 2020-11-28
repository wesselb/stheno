import lab as B
import numpy as np
import pytest
from matrix import Dense, Diagonal
from scipy.stats import multivariate_normal

from stheno.graph import GP
from stheno.random import Normal, RandomVector
from .util import approx


@pytest.fixture()
def normal1():
    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol @ chol.T
    return Normal(mean, var)


@pytest.fixture()
def normal2():
    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol @ chol.T
    return Normal(mean, var)


def test_normal_mean_is_zero():
    assert Normal(np.eye(3))._mean_is_zero
    assert not Normal(np.random.randn(3, 1), np.eye(3))._mean_is_zero


def test_normal_m2(normal1):
    approx(normal1.m2, normal1.var + normal1.mean @ normal1.mean.T)


def test_normal_marginals(normal1):
    mean, lower, upper = normal1.marginals()
    approx(mean, normal1.mean.squeeze())
    approx(lower, normal1.mean.squeeze() - 1.96 * B.diag(normal1.var) ** 0.5)
    approx(upper, normal1.mean.squeeze() + 1.96 * B.diag(normal1.var) ** 0.5)


def test_normal_logpdf(normal1):
    normal1_sp = multivariate_normal(normal1.mean[:, 0], B.dense(normal1.var))
    x = np.random.randn(3, 10)
    approx(normal1.logpdf(x), normal1_sp.logpdf(x.T))

    # Test the the output of `logpdf` is flattened appropriately.
    assert np.shape(normal1.logpdf(np.ones((3, 1)))) == ()
    assert np.shape(normal1.logpdf(np.ones((3, 2)))) == (2,)


def test_normal_entropy(normal1):
    normal1_sp = multivariate_normal(normal1.mean[:, 0], B.dense(normal1.var))
    approx(normal1.entropy(), normal1_sp.entropy())


def test_normal_kl(normal1, normal2):
    assert normal1.kl(normal1) < 1e-6
    assert normal1.kl(normal2) > 0.1

    # Test against Monte Carlo estimate.
    samples = normal1.sample(50_000)
    kl_est = np.mean(normal1.logpdf(samples)) - np.mean(normal2.logpdf(samples))
    kl = normal1.kl(normal2)
    approx(kl_est, kl, rtol=0.05)


def test_normal_sampling():
    for mean in [0, 1]:
        dist = Normal(mean, 3 * np.eye(200, dtype=np.integer))

        # Sample without noise.
        samples = dist.sample(2000)
        approx(np.mean(samples), mean, atol=5e-2)
        approx(np.var(samples), 3, atol=5e-2)

        # Sample with noise
        samples = dist.sample(2000, noise=2)
        approx(np.mean(samples), mean, atol=5e-2)
        approx(np.var(samples), 5, atol=5e-2)


def test_normal_display(normal1):
    assert str(normal1) == RandomVector.__str__(normal1)
    assert repr(normal1) == RandomVector.__repr__(normal1)


def test_normal_arithmetic(normal1, normal2):
    a = Dense(np.random.randn(3, 3))
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
    with pytest.raises(NotImplementedError):
        normal1.__mul__(normal1)
    with pytest.raises(NotImplementedError):
        normal1.__rmul__(normal1)

    # Test addition.
    approx((normal1 + normal2).mean, normal1.mean + normal2.mean)
    approx((normal1 + normal2).var, normal1.var + normal2.var)
    approx((normal1.__add__(b)).mean, normal1.mean + b)
    approx((normal1.__radd__(b)).mean, normal1.mean + b)
