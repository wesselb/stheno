# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from lab import B
from scipy.stats import multivariate_normal
from stheno.matrix import UniformlyDiagonal, Diagonal, dense
from stheno.random import Normal, Normal1D, RandomVector

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, eprint, allclose, \
    assert_allclose


def test_normal():
    mean = np.random.randn(3, 1)
    chol = np.random.randn(3, 3)
    var = chol.dot(chol.T)

    dist = Normal(var, mean)
    dist_sp = multivariate_normal(mean[:, 0], var)

    # Test second moment.
    yield assert_allclose, dist.m2, var + mean.dot(mean.T)

    # Test marginals.
    marg_mean, lower, upper = dist.marginals()
    yield assert_allclose, mean.squeeze(), marg_mean
    yield assert_allclose, lower, marg_mean - 2 * np.diag(var) ** .5
    yield assert_allclose, upper, marg_mean + 2 * np.diag(var) ** .5

    # Test `logpdf` and `entropy`.
    for _ in range(5):
        x = np.random.randn(3, 10)
        yield ok, allclose(dist.logpdf(x), dist_sp.logpdf(x.T)), 'logpdf'
        yield ok, allclose(dist.entropy(), dist_sp.entropy()), 'entropy'

    # Test the the output of `logpdf` is flattened appropriately.
    yield eq, np.shape(dist.logpdf(np.ones((3, 1)))), ()
    yield eq, np.shape(dist.logpdf(np.ones((3, 2)))), (2,)

    # Test KL with Monte Carlo estimate.
    mean2 = np.random.randn(3, 1)
    chol2 = np.random.randn(3, 3)
    var2 = chol2.dot(chol2.T)
    dist2 = Normal(var2, mean2)
    samples = dist.sample(50000)
    kl_est = np.mean(dist.logpdf(samples)) - np.mean(dist2.logpdf(samples))
    kl = dist.kl(dist2)
    yield ok, np.abs(kl_est - kl) / np.abs(kl) < 5e-2, 'kl sampled'


def test_normal_comparison():
    # Compare a diagonal normal and dense normal.
    mean = np.random.randn(3, 1)
    var_diag = np.random.randn(3) ** 2
    var = np.diag(var_diag)
    dist1 = Normal(var, mean)
    dist2 = Normal(Diagonal(var_diag), mean)
    samples = dist1.sample(100)
    yield ok, allclose(dist1.logpdf(samples),
                       dist2.logpdf(samples)), 'logpdf'
    yield ok, allclose(dist1.entropy(), dist2.entropy()), 'entropy'
    yield ok, allclose(dist1.kl(dist2), 0.), 'kl 1'
    yield ok, allclose(dist1.kl(dist1), 0.), 'kl 2'
    yield ok, allclose(dist2.kl(dist2), 0.), 'kl 3'
    yield ok, allclose(dist2.kl(dist1), 0.), 'kl 4'
    yield le, dist1.w2(dist1), 5e-4, 'w2 1'
    yield le, dist1.w2(dist2), 5e-4, 'w2 2'
    yield le, dist2.w2(dist1), 5e-4, 'w2 3'
    yield le, dist2.w2(dist2), 5e-4, 'w2 4'

    # Check a uniformly diagonal normal and dense normal.
    mean = np.random.randn(3, 1)
    var_diag_scale = np.random.randn() ** 2
    var = np.eye(3) * var_diag_scale
    dist1 = Normal(var, mean)
    dist2 = Normal(UniformlyDiagonal(var_diag_scale, 3), mean)
    samples = dist1.sample(100)
    yield ok, allclose(dist1.logpdf(samples),
                       dist2.logpdf(samples)), 'logpdf'
    yield ok, allclose(dist1.entropy(), dist2.entropy()), 'entropy'
    yield ok, allclose(dist1.kl(dist2), 0.), 'kl 1'
    yield ok, allclose(dist1.kl(dist1), 0.), 'kl 2'
    yield ok, allclose(dist2.kl(dist2), 0.), 'kl 3'
    yield ok, allclose(dist2.kl(dist1), 0.), 'kl 4'
    yield le, dist1.w2(dist1), 5e-4, 'w2 1'
    yield le, dist1.w2(dist2), 5e-4, 'w2 2'
    yield le, dist2.w2(dist1), 5e-4, 'w2 3'
    yield le, dist2.w2(dist2), 5e-4, 'w2 4'


def test_normal_sampling():
    # Test sampling and dtype conversion.
    dist = Normal(3 * np.eye(200, dtype=np.integer))
    yield le, np.abs(np.std(dist.sample(1000)) ** 2 - 3), 5e-2, 'full'
    yield le, np.abs(np.std(dist.sample(1000, noise=2)) ** 2 - 5), 5e-2, \
          'full 2'

    dist = Normal(Diagonal(3 * np.ones(200, dtype=np.integer)))
    yield le, np.abs(np.std(dist.sample(1000)) ** 2 - 3), 5e-2, 'diag'
    yield le, np.abs(np.std(dist.sample(1000, noise=2)) ** 2 - 5), 5e-2, \
          'diag 2'

    dist = Normal(UniformlyDiagonal(3, 200))
    yield le, np.abs(np.std(dist.sample(1000)) ** 2 - 3), 5e-2, 'unif'
    yield le, np.abs(np.std(dist.sample(1000, noise=2)) ** 2 - 5), 5e-2, \
          'unif 2'

    # Test `__str__` and `__repr__`.
    yield eq, str(dist), RandomVector.__str__(dist)
    yield eq, repr(dist), RandomVector.__repr__(dist)


def test_normal_1d():
    # Test broadcasting.
    d = Normal1D(1, 0)
    yield eq, type(d.var), UniformlyDiagonal
    yield eq, B.shape(d.var), (1, 1)
    yield eq, B.shape(d.mean), (1, 1)

    d = Normal1D(1, [0, 0, 0])
    yield eq, type(d.var), UniformlyDiagonal
    yield eq, B.shape(d.var), (3, 3)
    yield eq, B.shape(d.mean), (3, 1)

    d = Normal1D([1, 2, 3], 0)
    yield eq, type(d.var), Diagonal
    yield eq, B.shape(d.var), (3, 3)
    yield eq, B.shape(d.mean), (3, 1)

    d = Normal1D([1, 2, 3], [0, 0, 0])
    yield eq, type(d.var), Diagonal
    yield eq, B.shape(d.var), (3, 3)
    yield eq, B.shape(d.mean), (3, 1)

    d = Normal1D(1)
    yield eq, type(d.var), UniformlyDiagonal
    yield eq, B.shape(d.var), (1, 1)
    yield eq, B.shape(d.mean), (1, 1)

    d = Normal1D([1, 2, 3])
    yield eq, type(d.var), Diagonal
    yield eq, B.shape(d.var), (3, 3)
    yield eq, B.shape(d.mean), (3, 1)

    yield raises, ValueError, lambda: Normal1D(np.eye(3))
    yield raises, ValueError, lambda: Normal1D(np.eye(3), 0)
    yield raises, ValueError, lambda: Normal1D(1, np.ones((3, 1)))
    yield raises, ValueError, lambda: Normal1D([1, 2], np.ones((3, 1)))


def test_normal_arithmetic():
    chol = np.random.randn(3, 3)
    dist = Normal(chol.dot(chol.T), np.random.randn(3, 1))
    chol = np.random.randn(3, 3)
    dist2 = Normal(chol.dot(chol.T), np.random.randn(3, 1))

    A = np.random.randn(3, 3)
    a = np.random.randn(1, 3)
    b = 5.

    # Test matrix multiplication.
    yield ok, allclose((dist.rmatmul(a)).mean,
                       dist.mean.dot(a)), 'mean mul'
    yield ok, allclose((dist.rmatmul(a)).var,
                       a.dot(dense(dist.var)).dot(a.T)), 'var mul'
    yield ok, allclose((dist.lmatmul(A)).mean,
                       A.dot(dist.mean)), 'mean rmul'
    yield ok, allclose((dist.lmatmul(A)).var,
                       A.dot(dense(dist.var)).dot(A.T)), 'var rmul'

    # Test multiplication.
    yield ok, allclose((dist * b).mean, dist.mean * b), 'mean mul 2'
    yield ok, allclose((dist * b).var, dist.var * b ** 2), 'var mul 2'
    yield ok, allclose((b * dist).mean, dist.mean * b), 'mean rmul 2'
    yield ok, allclose((b * dist).var, dist.var * b ** 2), 'var rmul 2'
    yield raises, NotImplementedError, lambda: dist.__mul__(dist)
    yield raises, NotImplementedError, lambda: dist.__rmul__(dist)

    # Test addition.
    yield ok, allclose((dist + dist2).mean,
                       dist.mean + dist2.mean), 'mean sum'
    yield ok, allclose((dist + dist2).var,
                       dist.var + dist2.var), 'var sum'
    yield ok, allclose((dist.__add__(b)).mean, dist.mean + b), 'mean add'
    yield ok, allclose((dist.__radd__(b)).mean, dist.mean + b), 'mean radd'
