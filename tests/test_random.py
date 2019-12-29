import lab as B
import numpy as np
import pytest
from matrix import Diagonal
from scipy.stats import multivariate_normal

from stheno.graph import GP
from stheno.random import Normal, RandomVector
from .util import allclose


def test_normal():
    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol.dot(chol.T)

    dist = Normal(var, mean)
    dist_sp = multivariate_normal(mean[:, 0], var)

    # Test second moment.
    allclose(dist.m2, var + mean.dot(mean.T))

    # Test marginals.
    marg_mean, lower, upper = dist.marginals()
    allclose(mean.squeeze(), marg_mean)
    allclose(lower, marg_mean - 2 * np.diag(var) ** .5)
    allclose(upper, marg_mean + 2 * np.diag(var) ** .5)

    # Test `logpdf` and `entropy`.
    for _ in range(5):
        x = np.random.randn(3, 10)
        allclose(dist.logpdf(x), dist_sp.logpdf(x.T), desc='logpdf')
        allclose(dist.entropy(), dist_sp.entropy(), desc='entropy')

    # Test the the output of `logpdf` is flattened appropriately.
    assert np.shape(dist.logpdf(np.ones((3, 1)))) == ()
    assert np.shape(dist.logpdf(np.ones((3, 2)))) == (2,)

    # Test KL with Monte Carlo estimate.
    mean2 = np.random.randn(3, 1)
    chol2 = np.random.randn(3, 3)
    var2 = chol2.dot(chol2.T)
    dist2 = Normal(var2, mean2)
    samples = dist.sample(50000)
    kl_est = np.mean(dist.logpdf(samples)) - np.mean(dist2.logpdf(samples))
    kl = dist.kl(dist2)
    assert np.abs(kl_est - kl) / np.abs(kl) < 5e-2, 'kl sampled'


def test_normal_sampling():
    # Test sampling and dtype conversion.
    dist = Normal(3 * np.eye(200, dtype=np.integer))
    assert np.abs(np.std(dist.sample(1000)) ** 2 - 3) <= 5e-2, 'full'
    assert np.abs(np.std(dist.sample(1000, noise=2)) ** 2 - 5) <= 5e-2, 'full 2'

    # Test `__str__` and `__repr__`.
    assert str(dist) == RandomVector.__str__(dist)
    assert repr(dist) == RandomVector.__repr__(dist)

    # Test zero mean determination.
    assert Normal(np.eye(3))._zero_mean
    assert not Normal(np.eye(3), np.random.randn(3, 1))._zero_mean

    x = np.random.randn(3)
    assert GP(1)(x)._zero_mean
    assert not GP(1, 1)(x)._zero_mean
    assert GP(1, 0)(x)._zero_mean


def test_normal_arithmetic():
    chol = np.random.randn(3, 3)
    dist = Normal(chol.dot(chol.T), np.random.randn(3, 1))
    chol = np.random.randn(3, 3)
    dist2 = Normal(chol.dot(chol.T), np.random.randn(3, 1))

    A = np.random.randn(3, 3)
    a = np.random.randn(1, 3)
    b = 5.

    # Test matrix multiplication.
    allclose((dist.rmatmul(a)).mean, dist.mean.dot(a))
    allclose((dist.rmatmul(a)).var, a.dot(B.dense(dist.var)).dot(a.T))
    allclose((dist.lmatmul(A)).mean, A.dot(dist.mean))
    allclose((dist.lmatmul(A)).var, A.dot(B.dense(dist.var)).dot(A.T))

    # Test multiplication.
    allclose((dist * b).mean, dist.mean * b)
    allclose((dist * b).var, dist.var * b ** 2)
    allclose((b * dist).mean, dist.mean * b)
    allclose((b * dist).var, dist.var * b ** 2)
    with pytest.raises(NotImplementedError):
        dist.__mul__(dist)
    with pytest.raises(NotImplementedError):
        dist.__rmul__(dist)

    # Test addition.
    allclose((dist + dist2).mean, dist.mean + dist2.mean)
    allclose((dist + dist2).var, dist.var + dist2.var)
    allclose((dist.__add__(b)).mean, dist.mean + b)
    allclose((dist.__radd__(b)).mean, dist.mean + b)
