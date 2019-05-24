# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
from lab import B
from scipy.stats import multivariate_normal

from stheno.graph import GP
from stheno.matrix import UniformlyDiagonal, Diagonal, dense
from stheno.random import Normal, Normal1D, RandomVector
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


def test_normal_comparison():
    # Compare a diagonal normal and dense normal.
    mean = np.random.randn(3, 1)
    var_diag = np.random.randn(3) ** 2
    var = np.diag(var_diag)
    dist1 = Normal(var, mean)
    dist2 = Normal(Diagonal(var_diag), mean)
    samples = dist1.sample(100)
    allclose(dist1.logpdf(samples), dist2.logpdf(samples), desc='logpdf')
    allclose(dist1.entropy(), dist2.entropy(), desc='entropy')
    allclose(dist1.kl(dist2), 0.)
    allclose(dist1.kl(dist1), 0.)
    allclose(dist2.kl(dist2), 0.)
    allclose(dist2.kl(dist1), 0.)
    assert dist1.w2(dist1) <= 1e-3
    assert dist1.w2(dist2) <= 1e-3
    assert dist2.w2(dist1) <= 1e-3
    assert dist2.w2(dist2) <= 1e-3

    # Check a uniformly diagonal normal and dense normal.
    mean = np.random.randn(3, 1)
    var_diag_scale = np.random.randn() ** 2
    var = np.eye(3) * var_diag_scale
    dist1 = Normal(var, mean)
    dist2 = Normal(UniformlyDiagonal(var_diag_scale, 3), mean)
    samples = dist1.sample(100)
    allclose(dist1.logpdf(samples), dist2.logpdf(samples), desc='logpdf')
    allclose(dist1.entropy(), dist2.entropy(), desc='entropy')
    allclose(dist1.kl(dist2), 0.)
    allclose(dist1.kl(dist1), 0.)
    allclose(dist2.kl(dist2), 0.)
    allclose(dist2.kl(dist1), 0.)
    assert dist1.w2(dist1) <= 1e-3
    assert dist1.w2(dist2) <= 1e-3
    assert dist2.w2(dist1) <= 1e-3
    assert dist2.w2(dist2) <= 1e-3


def test_normal_sampling():
    # Test sampling and dtype conversion.
    dist = Normal(3 * np.eye(200, dtype=np.integer))
    assert np.abs(np.std(dist.sample(1000)) ** 2 - 3) <= 5e-2, 'full'
    assert np.abs(np.std(dist.sample(1000, noise=2)) ** 2 - 5) <= 5e-2, 'full 2'

    dist = Normal(Diagonal(3 * np.ones(200, dtype=np.integer)))
    assert np.abs(np.std(dist.sample(1000)) ** 2 - 3) <= 5e-2, 'diag'
    assert np.abs(np.std(dist.sample(1000, noise=2)) ** 2 - 5) <= 5e-2, 'diag 2'

    dist = Normal(UniformlyDiagonal(3, 200))
    assert np.abs(np.std(dist.sample(1000)) ** 2 - 3) <= 5e-2, 'unif'
    assert np.abs(np.std(dist.sample(1000, noise=2)) ** 2 - 5) <= 5e-2, 'unif 2'

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


def test_normal_1d():
    # Test broadcasting.
    d = Normal1D(1, 0)
    assert type(d.var) == UniformlyDiagonal
    assert B.shape(d.var) == (1, 1)
    assert B.shape(d.mean) == (1, 1)

    d = Normal1D(1, np.array([0, 0, 0]))
    assert type(d.var) == UniformlyDiagonal
    assert B.shape(d.var) == (3, 3)
    assert B.shape(d.mean) == (3, 1)

    d = Normal1D(np.array([1, 2, 3]), 0)
    assert type(d.var) == Diagonal
    assert B.shape(d.var) == (3, 3)
    assert B.shape(d.mean) == (3, 1)

    d = Normal1D(np.array([1, 2, 3]), np.array([0, 0, 0]))
    assert type(d.var) == Diagonal
    assert B.shape(d.var) == (3, 3)
    assert B.shape(d.mean) == (3, 1)

    d = Normal1D(1)
    assert type(d.var) == UniformlyDiagonal
    assert B.shape(d.var) == (1, 1)
    assert B.shape(d.mean) == (1, 1)

    d = Normal1D(np.array([1, 2, 3]))
    assert type(d.var) == Diagonal
    assert B.shape(d.var) == (3, 3)
    assert B.shape(d.mean) == (3, 1)

    with pytest.raises(ValueError):
        Normal1D(np.eye(3))
    with pytest.raises(ValueError):
        Normal1D(np.eye(3), 0)
    with pytest.raises(ValueError):
        Normal1D(1, np.ones((3, 1)))
    with pytest.raises(ValueError):
        Normal1D(np.array([1, 2]),
                 np.ones((3, 1)))


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
    allclose((dist.rmatmul(a)).var, a.dot(dense(dist.var)).dot(a.T))
    allclose((dist.lmatmul(A)).mean, A.dot(dist.mean))
    allclose((dist.lmatmul(A)).var, A.dot(dense(dist.var)).dot(A.T))

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
